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

    def _get_next_output_text(self, finished, delta):
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


def register():
    _patch_detokenizer_whitespace_stripping()

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
