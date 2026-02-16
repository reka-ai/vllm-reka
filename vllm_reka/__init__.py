# ABOUTME: Plugin entry point that registers Reka models, tokenizer, and configs with vLLM.
# ABOUTME: Called automatically by vLLM's plugin discovery via setup.py entry_points.


def register():
    from transformers import AutoConfig
    from vllm import ModelRegistry
    from vllm.tokenizers import TokenizerRegistry

    from .config import YasaConfig, YasaMMLMConfig, YasaMMLMV2MMLMConfig

    # Register configs with transformers AutoConfig
    AutoConfig.register("yasa_model", YasaConfig)
    AutoConfig.register("yasa_mmlm_model", YasaMMLMConfig)
    AutoConfig.register("yasa_edge_mmlm_model", YasaMMLMV2MMLMConfig)

    # Register tokenizer mode with vLLM
    # User passes --tokenizer-mode yasa to activate
    TokenizerRegistry.register(
        "yasa", "vllm_reka.tokenizer", "YasaTokenizer"
    )

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
