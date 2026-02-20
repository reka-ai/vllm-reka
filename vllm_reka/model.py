# ABOUTME: YasaCausalLM text-only model implementation for vLLM.
# ABOUTME: LLaMA-like architecture with GQA, RoPE, and SwiGLU MLP.
from collections.abc import Iterable
from typing import Any, Optional, Union

import regex as re
import torch
from torch import nn

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)


class YasaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class YasaAttention(nn.Module):

    def __init__(self,
                 config,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 rope_theta: float = 10000,
                 rope_scaling: Optional[dict[str, Any]] = None,
                 max_position_embeddings: int = 8192,
                 quant_config: Optional[QuantizationConfig] = None,
                 bias: bool = False,
                 bias_o_proj: bool = False,
                 cache_config: Optional[CacheConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        # Phi models introduced a partial_rotary_factor parameter in the config
        partial_rotary_factor = getattr(config, "rotary_pct", 1)
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        rope_parameters = {
            "rope_theta": rope_theta,
            "partial_rotary_factor": self.rotary_dim / self.head_dim,
        }
        if rope_scaling:
            rope_parameters.update(rope_scaling)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=is_neox_style,
        )

        if hasattr(config, "interleaved_sliding_window"):
            interleaved_sliding_window = config.interleaved_sliding_window
            if isinstance(interleaved_sliding_window, int):
                sliding_window = interleaved_sliding_window
            elif isinstance(interleaved_sliding_window, list):
                sw_idx = layer_idx % len(interleaved_sliding_window)
                sliding_window = interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(
                    f"{type(interleaved_sliding_window)} is not supported.")
        else:
            sliding_window = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class YasaDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rotary_emb_base", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        mlp_bias = config.mlp_bias
        attention_bias = config.attention_bias
        # compatibility with old gpt neox style checkpoints
        rms_norm_eps = (getattr(config, "rms_norm_eps", None)
                        or getattr(config, "layer_norm_eps", None) or 1e-5)
        self.self_attn = YasaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = YasaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=mlp_bias,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class YasaModel(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type: type[YasaDecoderLayer] = YasaDecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(config=config,
                                      cache_config=cache_config,
                                      quant_config=quant_config,
                                      prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        # compatibility with old gpt neox style checkpoints
        rms_norm_eps = (getattr(config, "rms_norm_eps", None)
                        or getattr(config, "layer_norm_eps", None) or 1e-5)
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name == "bias":
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            # Handle unpacked MLP projections (convert to merged gate_up_proj)
            if ".mlp.gate_proj." in name:
                name = name.replace(".gate_proj.", ".gate_up_proj.")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight, 0)
                loaded_params.add(name)
                continue
            if ".mlp.up_proj." in name:
                name = name.replace(".up_proj.", ".gate_up_proj.")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight, 1)
                loaded_params.add(name)
                continue

            # handle merged qkv projection
            if ".qkv_proj" in name:
                q, k, v = self._parse_qkv_tensors(loaded_weight)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, q, "q")
                weight_loader(param, k, "k")
                weight_loader(param, v, "v")
                loaded_params.add(name)
                continue

            # Handle unpacked attention projections (convert to merged qkv_proj)
            if ".self_attn.q_proj." in name:
                name = name.replace(".q_proj.", ".qkv_proj.")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight, "q")
                loaded_params.add(name)
                continue
            if ".self_attn.k_proj." in name:
                name = name.replace(".k_proj.", ".qkv_proj.")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight, "k")
                loaded_params.add(name)
                continue
            if ".self_attn.v_proj." in name:
                name = name.replace(".v_proj.", ".qkv_proj.")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight, "v")
                loaded_params.add(name)
                continue

            # handle gate and up projection
            if ".gate_up_proj" in name:
                gate_proj, up_proj = self._parse_gate_up_projection(
                    loaded_weight)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, gate_proj, 0)
                weight_loader(param, up_proj, 1)
                loaded_params.add(name)
                continue

            # Skip loading extra bias for GPTQ models.
            else:
                if name.endswith(
                        ".bias") and name not in params_dict or name.endswith(
                            ".masked_bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params

    def _parse_qkv_tensors(
        self,
        loaded_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        queries_per_group = (self.config.num_attention_heads //
                             self.config.num_query_groups)
        qkv_items_per_group = queries_per_group + 2
        qkv_items_total = qkv_items_per_group * self.config.num_query_groups
        head_size = loaded_weight.size(0) // qkv_items_total
        # this relies on the GPT Neox-style layout of doing group after group,
        # i.e. q, k, v, q, k, v, q, .. in the case of MHA or
        # q, q, q, .., k, v, q, q, q, .., k, v, q, .. in the case of GQA
        # TO DO verify efficiency of this,
        # (less important since it's weight loading time not inference)
        queries = torch.cat(
            [
                loaded_weight.narrow(
                    0,
                    query_group_id * qkv_items_per_group * head_size,
                    queries_per_group * head_size,
                ) for query_group_id in range(self.config.num_query_groups)
            ],
            dim=0,
        )

        keys = torch.cat(
            [
                loaded_weight.narrow(
                    0,
                    (query_group_id * qkv_items_per_group + queries_per_group)
                    * head_size,
                    head_size,
                ) for query_group_id in range(self.config.num_query_groups)
            ],
            dim=0,
        )
        values = torch.cat(
            [
                loaded_weight.narrow(
                    0,
                    (query_group_id * qkv_items_per_group + queries_per_group +
                     1) * head_size,
                    head_size,
                ) for query_group_id in range(self.config.num_query_groups)
            ],
            dim=0,
        )

        return queries, keys, values

    def _parse_gate_up_projection(
            self, loaded_weight: torch.Tensor) -> tuple[torch.Tensor, ...]:
        gate_proj = loaded_weight.narrow(0, 0, loaded_weight.size(0) // 2)
        up_proj = loaded_weight.narrow(0,
                                       loaded_weight.size(0) // 2,
                                       loaded_weight.size(0) // 2)
        return gate_proj, up_proj


class YasaCausalLM(nn.Module, SupportsLoRA, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.model = self._init_model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config else
                    lora_config.lora_vocab_padding_size),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def _init_model(self, vllm_config: VllmConfig, prefix: str = ""):
        return YasaModel(vllm_config=vllm_config, prefix=prefix)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, intermediate_tensors,
                                  inputs_embeds)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # Filter out causal mask weights
        # (not needed - fused in efficient inference)
        # Also handle typo variant "casual_mask" from some checkpoints
        def _filter_weights():
            for name, loaded_weight in weights:
                if "casual_mask" in name or "causal_mask" in name:
                    continue
                yield (self._maybe_map_gptneox_to_llama(name), loaded_weight)

        # Then use AutoWeightsLoader with the mapped names
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(_filter_weights())

    @staticmethod
    def _maybe_map_gptneox_to_llama(param_name: str) -> str:
        """
        Remap GPT-NeoX weight names to LLaMA style weight names.
        Skips query, key, and values will need to be handled separately
        """
        if "embed_out.weight" in param_name:
            renamed_param = param_name.replace("embed_out.weight",
                                               "lm_head.weight")
            return renamed_param
        elif param_name.startswith("lm_head."):
            return param_name
        elif "gpt_neox" not in param_name:
            return f"model.{param_name}"
        elif "gpt_neox.embed_in." in param_name:
            renamed_param = param_name.replace("gpt_neox.embed_in.",
                                               "model.embed_tokens.")
            return renamed_param
        elif "gpt_neox.final_layer_norm." in param_name:
            renamed_param = param_name.replace("gpt_neox.final_layer_norm.",
                                               "model.norm.")
            return renamed_param
        elif param_name == "gpt_neox.bias":
            renamed_param = "model.bias"
            return renamed_param
        elif param_name == "gpt_neox.rotary_emb.inv_freq":
            renamed_param = "model.rotary_emb.inv_freq"
            return renamed_param
        # All remaining names should be within the layer namespace.
        if not param_name.startswith("gpt_neox.layers"):
            raise ValueError(f"Unrecognised model weight name: {param_name}")
        # Extract layer index and the sub-name.
        group = re.match(r"gpt_neox.layers\.(\d+)\.(.*)", param_name)
        if not group:
            raise ValueError(f"Unrecognised model weight name: {param_name}")
        l_id = int(group.group(1))
        l_name = group.group(2)

        if l_name == "attention.dense.weight":
            renamed_param = f"model.layers.{l_id}.self_attn.o_proj.weight"
            return renamed_param
        elif l_name == "attention.dense.bias":
            renamed_param = f"model.layers.{l_id}.self_attn.o_proj.bias"
            return renamed_param
        # in original llama style qkv are separated, but in vllm implementatio
        # they are not. So for easier intergation
        # we keep as is here, separate and load
        elif l_name == "attention.query_key_value.weight":
            renamed_param = f"model.layers.{l_id}.self_attn.qkv_proj.weight"
            return renamed_param
        elif l_name == "attention.query_key_value.bias":
            renamed_param = f"model.layers.{l_id}.self_attn.qkv_proj.bias"
            return renamed_param
        elif l_name == "input_layernorm.weight":
            renamed_param = f"model.layers.{l_id}.input_layernorm.weight"
            return renamed_param
        elif l_name == "mlp.dense_4h_to_h.weight":
            renamed_param = f"model.layers.{l_id}.mlp.down_proj.weight"
            return renamed_param
        elif l_name == "mlp.dense_4h_to_h.bias":
            renamed_param = f"model.layers.{l_id}.mlp.down_proj.bias"
            return renamed_param
        elif l_name == "mlp.dense_h_to_4h.weight":
            renamed_param = f"model.layers.{l_id}.mlp.gate_up_proj.weight"
            return renamed_param
        elif l_name == "mlp.dense_h_to_4h.bias":
            renamed_param = f"model.layers.{l_id}.mlp.gate_up_proj.bias"
            return renamed_param
        elif l_name == "post_attention_layernorm.weight":
            renamed_param = \
                f"model.layers.{l_id}.post_attention_layernorm.weight"
            return renamed_param
        else:
            return param_name.replace("gpt_neox.", "model.")
