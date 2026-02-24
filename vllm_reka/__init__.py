# ABOUTME: Plugin entry point that registers Reka models, tokenizer, and configs with vLLM.
# ABOUTME: Called automatically by vLLM's plugin discovery via setup.py entry_points.


def _patch_detokenizer_whitespace_stripping():
    """Patch the detokenizer to strip leading whitespace from generated text.

    Tiktoken tokens include leading whitespace (e.g., " Hello" not "Hello").
    The detokenizer preserves this in its output, but the first generated
    token's leading whitespace should be stripped for clean chat responses.
    Only applies when the tokenizer sets strip_leading_whitespace = True.
    """
    from vllm.v1.engine.detokenizer import (
        BaseIncrementalDetokenizer,
        SlowIncrementalDetokenizer,
    )

    _original = BaseIncrementalDetokenizer.get_next_output_text

    def _get_next_output_text(self, finished: bool, delta: bool) -> str:
        text = _original(self, finished, delta)
        # Only strip for tokenizers that opt in (e.g., YasaTokenizer)
        if not getattr(getattr(self, 'tokenizer', None),
                       'strip_leading_whitespace', False):
            return text
        # Non-streaming: full text is available, strip it
        if not delta:
            return text.lstrip()
        # Streaming: strip only the first non-empty chunk to remove
        # the leading space, then pass subsequent chunks through unchanged
        if not getattr(self, '_first_stripped', False) and text:
            self._first_stripped = True
            return text.lstrip()
        return text

    SlowIncrementalDetokenizer.get_next_output_text = _get_next_output_text


def _patch_xgrammar_backend():
    """Patch xgrammar to handle tiktoken-based tokenizers.

    xgrammar's TokenizerInfo.from_huggingface() expects get_vocab() to return
    string keys, but YasaTokenizer (tiktoken-based) returns bytes keys.
    This intercepts XgrammarBackend initialization to manually build the
    vocabulary list and tell xgrammar it's byte-fallback encoded.
    """
    from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

    _original_post_init = XgrammarBackend.__post_init__

    def _patched_post_init(self) -> None:
        from vllm_reka.tokenizer import YasaTokenizer

        if not isinstance(self.tokenizer, YasaTokenizer):
            return _original_post_init(self)

        import vllm.envs
        import xgrammar as xgr

        self.disable_any_whitespace = \
            self.vllm_config.structured_outputs_config.disable_any_whitespace

        vocab_dict = self.tokenizer.get_vocab()
        encoded_vocab = [
            token for token, _ in sorted(vocab_dict.items(),
                                         key=lambda x: x[1])
        ]

        stop_token_ids = None
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids = [self.tokenizer.eos_token_id]

        tokenizer_info = xgr.TokenizerInfo(
            encoded_vocab=encoded_vocab,
            vocab_type=xgr.VocabType.BYTE_FALLBACK,
            vocab_size=self.vocab_size,
            stop_token_ids=stop_token_ids,
            add_prefix_space=False,
        )

        self.compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=8,
            cache_enabled=True,
            cache_limit_bytes=vllm.envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024,
        )

        self.num_speculative_tokens = 0
        if self.vllm_config.speculative_config is not None:
            self.num_speculative_tokens = \
                self.vllm_config.speculative_config.num_speculative_tokens

    XgrammarBackend.__post_init__ = _patched_post_init


def register():
    _patch_detokenizer_whitespace_stripping()
    _patch_xgrammar_backend()

    from transformers import AutoConfig
    from vllm import ModelRegistry
    from vllm.tokenizers import TokenizerRegistry

    from .config import YasaConfig, YasaMMLMConfig, YasaMMLMV2MMLMConfig

    # Register configs with transformers AutoConfig
    AutoConfig.register("yasa_model", YasaConfig)
    AutoConfig.register("yasa_mmlm_model", YasaMMLMConfig)
    AutoConfig.register("yasa_edge_mmlm_model", YasaMMLMV2MMLMConfig)

    # Register tokenizer with both vLLM and HuggingFace AutoTokenizer
    # User passes --tokenizer-mode yasa to activate
    TokenizerRegistry.register(
        "yasa", "vllm_reka.tokenizer", "YasaTokenizer"
    )
    from transformers import AutoTokenizer
    from .tokenizer import YasaTokenizer
    AutoTokenizer.register(YasaConfig, slow_tokenizer_class=YasaTokenizer)
    AutoTokenizer.register(YasaMMLMConfig, slow_tokenizer_class=YasaTokenizer)
    AutoTokenizer.register(YasaMMLMV2MMLMConfig, slow_tokenizer_class=YasaTokenizer)

    # Register renderer for the yasa tokenizer mode
    # Reuses the HfRenderer since YasaTokenizer is a PreTrainedTokenizer
    from vllm.renderers.registry import RENDERER_REGISTRY
    RENDERER_REGISTRY.register("yasa", "vllm.renderers.hf", "HfRenderer")

    # Register model architectures with vLLM (lazy import via string)
    if "YasaCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "YasaCausalLM", "vllm_reka.model:YasaCausalLM"
        )
    if "YasaMMLMForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "YasaMMLMForConditionalGeneration",
            "vllm_reka.multimodal_model:YasaMMLMForConditionalGeneration",
        )
    if "Yasa2ForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "Yasa2ForConditionalGeneration",
            "vllm_reka.edge_model:YasaMMLMV2ForConditionalGeneration",
        )
