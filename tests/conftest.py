# ABOUTME: Pytest conftest that stubs vllm modules when vllm is unavailable.
# ABOUTME: Allows tests to run on macOS/non-GPU systems where vllm can't be installed.

import sys
import types
from unittest.mock import MagicMock


def _install_vllm_stubs():
    """Install vllm module stubs so vllm_reka modules can be imported."""
    try:
        import vllm  # noqa: F401
        return  # vllm is available, no stubs needed
    except ImportError:
        pass

    def _make_class(name):
        """Create a stub class that supports generic subscripting (Cls[T])."""
        return type(name, (), {"__class_getitem__": classmethod(lambda cls, x: cls)})

    stub_modules = [
        "vllm",
        "vllm.config",
        "vllm.model_executor",
        "vllm.model_executor.models",
        "vllm.model_executor.models.interfaces",
        "vllm.model_executor.models.siglip",
        "vllm.model_executor.models.utils",
        "vllm.multimodal",
        "vllm.multimodal.inputs",
        "vllm.multimodal.parse",
        "vllm.multimodal.processing",
        "vllm.multimodal.video",
        "vllm.sequence",
    ]
    for mod_name in stub_modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

    # vllm.config
    sys.modules["vllm.config"].VllmConfig = _make_class("VllmConfig")

    # vllm.model_executor.models.interfaces
    sys.modules["vllm.model_executor.models.interfaces"].SupportsMultiModal = (
        _make_class("SupportsMultiModal"))
    sys.modules["vllm.model_executor.models.interfaces"].SupportsPP = (
        _make_class("SupportsPP"))

    # vllm.model_executor.models.siglip
    sys.modules["vllm.model_executor.models.siglip"].SiglipVisionModel = (
        _make_class("SiglipVisionModel"))

    # vllm.model_executor.models.utils
    utils = sys.modules["vllm.model_executor.models.utils"]
    utils._merge_multimodal_embeddings = MagicMock()
    utils.AutoWeightsLoader = MagicMock()
    utils.WeightsMapper = MagicMock()
    utils.init_vllm_registered_model = MagicMock()
    utils.maybe_prefix = MagicMock()

    # vllm.multimodal — registry that acts as a passthrough decorator
    mm_registry = MagicMock()
    mm_registry.register_processor = lambda *a, **kw: lambda cls: cls
    sys.modules["vllm.multimodal"].MULTIMODAL_REGISTRY = mm_registry
    sys.modules["vllm.multimodal"].MultiModalDataDict = MagicMock()

    # vllm.multimodal.inputs
    sys.modules["vllm.multimodal.inputs"].MultiModalFieldConfig = MagicMock()
    sys.modules["vllm.multimodal.inputs"].MultiModalKwargsItems = MagicMock()
    sys.modules["vllm.multimodal.inputs"].NestedTensors = MagicMock()

    # vllm.multimodal.parse
    sys.modules["vllm.multimodal.parse"].MultiModalDataItems = MagicMock()
    sys.modules["vllm.multimodal.parse"].MultiModalDataParser = MagicMock()

    # vllm.multimodal.processing — classes used as base classes need real types
    proc = sys.modules["vllm.multimodal.processing"]
    proc.BaseMultiModalProcessor = _make_class("BaseMultiModalProcessor")
    proc.BaseProcessingInfo = _make_class("BaseProcessingInfo")
    proc.BaseDummyInputsBuilder = _make_class("BaseDummyInputsBuilder")
    proc.InputProcessingContext = MagicMock()
    proc.PromptReplacement = MagicMock()
    proc.PromptUpdate = MagicMock()
    proc.PromptUpdateDetails = MagicMock()
    proc.ProcessorInputs = MagicMock()

    # vllm.multimodal.video — registry decorator must preserve the class
    video_registry = MagicMock()
    video_registry.register = lambda name: lambda cls: cls
    sys.modules["vllm.multimodal.video"].VIDEO_LOADER_REGISTRY = video_registry
    sys.modules["vllm.multimodal.video"].VideoLoader = _make_class("VideoLoader")

    # vllm.sequence
    sys.modules["vllm.sequence"].IntermediateTensors = MagicMock()


# Run before test collection so module-level imports succeed
_install_vllm_stubs()
