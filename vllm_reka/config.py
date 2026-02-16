# ABOUTME: HuggingFace PretrainedConfig subclasses for Reka models.
# ABOUTME: Defines YasaConfig (text), YasaMMLMConfig (SigLIP), YasaMMLMV2MMLMConfig (ConvNextV2).
import copy
from typing import Optional

from transformers import ConvNextV2Config, PretrainedConfig, SiglipVisionConfig


class YasaConfig(PretrainedConfig):
    """
    HF-style configuration, which is found in YasModelConfig

    TODO(kally): Maybe combine YasModelConfig with YasHFConfig,
    doesn't seem to be worth it to separate them
    """

    model_type = "yasa_model"

    def __init__(
        self,
        vocab_size: int = 50432,
        hidden_size: int = 6144,
        num_hidden_layers: int = 44,
        num_attention_heads: int = 64,
        intermediate_size: int = 24576,
        hidden_act: str = "gelu",
        rotary_pct: float = 0.25,
        rotary_emb_base: int = 10000,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        num_moe_experts: int = 1,
        moe_router_topk: Optional[int] = None,
        use_cache: bool = True,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        use_parallel_residual: bool = True,
        add_position_embedding: bool = False,
        mp_size: int = 1,
        quantization: Optional[str] = None,
        transposed_mode: bool = False,
        flash: bool = False,
        num_query_groups: Optional[int] = None,
        normalization: str = "LayerNorm",
        rotary_seq_len_interpolation_factor: Optional[int] = None,
        has_score_layer: bool = False,
        rms_norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id,
                         **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.num_moe_experts = num_moe_experts
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_parallel_residual = use_parallel_residual
        self.add_position_embedding = add_position_embedding
        self.moe_router_topk = moe_router_topk

        self.mp_size = mp_size
        self.quantization = quantization
        self.transposed_mode = transposed_mode
        self.flash = flash
        self.rotary_dim = int(self.hidden_size // self.num_attention_heads *
                              self.rotary_pct)
        self.rotate_half = True
        self.rotate_every_two = False
        self.triangular_masking = True
        self.local_attention = False
        self.window_size = 256
        self.mlp_after_attn = not self.use_parallel_residual
        MLP_ACT_FUNC_MAP = {
            "swiglu": "silu",
        }
        self.hidden_act = MLP_ACT_FUNC_MAP.get(hidden_act, "silu")
        self.num_query_groups = (num_query_groups or num_attention_heads
                                 )  # if not specified, assume MHA
        self.normalization = normalization
        self.rotary_seq_len_interpolation_factor = (
            rotary_seq_len_interpolation_factor)
        self.has_score_layer = has_score_layer
        # fix required for kv tranfer (simple_connector.py)
        self.num_key_value_heads = self.num_query_groups
        self.mlp_bias = kwargs.get("mlp_bias", False)
        self.attention_bias = kwargs.get("attention_bias", False)
        self.o_proj_bias = kwargs.get("o_proj_bias", False)


class YasaMMLMConfig(PretrainedConfig):
    model_type = "yasa_mmlm_model"
    is_composition = True

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        vision_pooling: Optional[str] = None,
        vision_tiling_method: Optional[str] = None,
        vision_max_tiles_num: Optional[int] = None,
        vision_grid_pinpoints=None,
        video_encoding_format="frames",
        video_encode_audio_track=False,
        transcription_encoding_format="segments",
        num_query_tokens=32,
        max_padded_length=384,
        dtype="bf16",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Handle case where transformers creates default instance with no args
        if text_config is None:
            text_config = {}
        if vision_config is None:
            vision_config = {}
        if vision_grid_pinpoints is None:
            vision_grid_pinpoints = [
                (2, 2),
                (1, 2),
                (2, 1),
                (1, 3),
                (3, 1),
                (1, 4),
                (4, 1),
            ]
        self.text_config = YasaConfig(**text_config)
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.vision_pooling = vision_pooling
        self.vision_tiling_method = vision_tiling_method
        self.vision_max_tiles_num = vision_max_tiles_num
        self.vision_grid_pinpoints = vision_grid_pinpoints
        self.video_encoding_format = video_encoding_format

        self.qformer_config = None
        self.audio_encoder_arch = None
        self.audio_type = None  # deprecated; backward compatibility
        self.audio_config = None
        self.clap_audio_config = None  # deprecated; backward compatibility
        self.audio_projection_no_layernorm = None
        self.tts_config = None
        self.transcription_encoding_format = transcription_encoding_format
        self.video_encode_audio_track = video_encode_audio_track
        self.num_query_tokens = num_query_tokens
        self.max_padded_length = max_padded_length
        self.dtype = dtype
        self.is_multimodal_model = True
        self.architectures = ["YasaMMLMForConditionalGeneration"]

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            Dictionary of all the attributes that make up this
            configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if self.vision_config is not None:
            output["vision_config"] = self.vision_config.to_dict()
        if self.qformer_config is not None:
            output["qformer_config"] = self.qformer_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        if self.audio_config is not None:
            output["audio_config"] = self.audio_config.to_dict()
        if self.tts_config is not None:
            output["tts_config"] = self.tts_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class YasaMMLMV2MMLMConfig(PretrainedConfig):
    """
    Configuration for YasaMMLMV2 MMLM model with ConvNextV2 vision encoder.

    This model uses ConvNextV2 (CNN-based) as the vision encoder instead of
    SigLIP (ViT-based) used in YasaMMLMConfig.
    """

    model_type = "yasa_edge_mmlm_model"
    is_composition = True

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        vision_pooling: Optional[str] = None,
        vision_tiling_method: Optional[str] = None,
        vision_max_tiles_num: Optional[int] = None,
        vision_grid_pinpoints=None,
        vision_encoder_arch: str = "convnextv2",
        video_encoding_format="frames",
        video_encode_audio_track=False,
        transcription_encoding_format="segments",
        num_query_tokens=49,  # Default for 7x7 pooled features
        max_padded_length=384,
        dtype="bf16",
        use_vision_pos_embed: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Handle case where transformers creates default instance with no args
        if text_config is None:
            text_config = {}
        if vision_config is None:
            vision_config = {}
        if vision_grid_pinpoints is None:
            vision_grid_pinpoints = [
                (2, 2),
                (1, 2),
                (2, 1),
                (1, 3),
                (3, 1),
                (1, 4),
                (4, 1),
            ]

        # Text config uses YasaConfig
        self.text_config = YasaConfig(**text_config)
        # Vision config uses ConvNextV2Config
        if isinstance(vision_config, dict):
            vision_config["num_stages"] = 4
            self.vision_config = ConvNextV2Config(**vision_config)
        else:
            # Already instantiated, patch it
            if not hasattr(vision_config, "num_stages"):
                vision_config.num_stages = 4
            self.vision_config = vision_config
        self.vision_encoder_arch = vision_encoder_arch
        self.vision_pooling = vision_pooling
        self.vision_tiling_method = vision_tiling_method
        self.vision_max_tiles_num = vision_max_tiles_num
        self.vision_grid_pinpoints = vision_grid_pinpoints
        self.video_encoding_format = video_encoding_format
        self.video_encode_audio_track = video_encode_audio_track
        self.transcription_encoding_format = transcription_encoding_format

        # Get vision hidden size from the last stage of ConvNextV2
        # Default hidden_sizes for ConvNextV2-Huge: [352, 704, 1408, 2816]
        if hasattr(self.vision_config, "hidden_sizes"):
            self.vision_hidden_size = self.vision_config.hidden_sizes[-1]
        else:
            self.vision_hidden_size = 2816  # Default for huge model

        # Audio/QFormer configs (may be None for edge model)
        self.qformer_config = None
        self.audio_encoder_arch = None
        self.audio_type = None  # deprecated; backward compatibility
        self.audio_config = None
        self.clap_audio_config = None  # deprecated; backward compatibility
        self.audio_projection_no_layernorm = None
        self.tts_config = None

        self.num_query_tokens = num_query_tokens
        self.max_padded_length = max_padded_length
        self.dtype = dtype
        self.use_vision_pos_embed = use_vision_pos_embed
        self.is_multimodal_model = True
        self.architectures = ["YasaMMLMV2ForConditionalGeneration"]

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            Dictionary of all the attributes that make up this
            configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if self.vision_config is not None:
            output["vision_config"] = self.vision_config.to_dict()
        if self.qformer_config is not None:
            output["qformer_config"] = self.qformer_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        if self.audio_config is not None:
            output["audio_config"] = self.audio_config.to_dict()
        if self.tts_config is not None:
            output["tts_config"] = self.tts_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
