# ABOUTME: Tests for multimodal_model prompt update targets and dummy text placeholders.
# ABOUTME: Verifies video/image targets match chat template output for correct placeholder expansion.

import pytest

from vllm_reka.multimodal_utils import (
    _IMAGE_PLACEHOLDER_TOKEN_ID,
    _START_VIDEO_TOKEN,
    _END_VIDEO_TOKEN,
)
from vllm_reka.tokenizer import YasaTokenizer


@pytest.fixture()
def tokenizer():
    return YasaTokenizer(tiktoken_model_name="cl100k_base")


# ── Test group A: Video target token contract ─────────────────────────


class TestVideoTargetContract:
    """The video target in _get_prompt_updates must match the chat template.

    The chat template emits <video></video> for video items, which tokenizes
    to [_START_VIDEO_TOKEN, _END_VIDEO_TOKEN] (100284, 100285). The video
    target must be this 2-token pair — NOT _IMAGE_PLACEHOLDER_TOKEN_ID.
    """

    def test_no_video_token_alias(self):
        """multimodal_model must not define _VIDEO_TOKEN as an image placeholder alias."""
        import vllm_reka.multimodal_model as mm
        assert not hasattr(mm, "_VIDEO_TOKEN"), (
            "_VIDEO_TOKEN should not exist — video targets should use "
            "_START_VIDEO_TOKEN/_END_VIDEO_TOKEN directly.")

    def test_video_template_tokenizes_to_start_end_pair(self, tokenizer):
        """Chat template video placeholder must tokenize to [100284, 100285]."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "dummy"},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True)
        ids_list = ids[0].tolist() if hasattr(ids[0], "tolist") else list(ids)
        assert _START_VIDEO_TOKEN in ids_list
        assert _END_VIDEO_TOKEN in ids_list
        start_idx = ids_list.index(_START_VIDEO_TOKEN)
        assert ids_list[start_idx + 1] == _END_VIDEO_TOKEN

    def test_image_template_tokenizes_to_img_token(self, tokenizer):
        """Chat template image placeholder must tokenize to 100278 only."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "dummy"},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True)
        ids_list = ids[0].tolist() if hasattr(ids[0], "tolist") else list(ids)
        assert _IMAGE_PLACEHOLDER_TOKEN_ID in ids_list
        assert _START_VIDEO_TOKEN not in ids_list
        assert _END_VIDEO_TOKEN not in ids_list


# ── Test group B: Dummy text placeholders ─────────────────────────────


class TestDummyTextPlaceholders:
    """Dummy text must use the same placeholders as the chat template.

    If the chat template emits <video></video> for video items but dummy
    text uses <REKA_IMG_TOKEN>, vLLM's profiling pass will use a different
    token layout than real inference, causing mismatches.
    """

    def test_dummy_video_uses_video_tags(self, tokenizer):
        """Dummy video placeholder must be <video></video>, not <REKA_IMG_TOKEN>."""
        # What the chat template produces for video
        video_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "dummy"},
                    {"type": "text", "text": "x"},
                ],
            }
        ]
        template_prompt = tokenizer.apply_chat_template(
            video_messages, add_generation_prompt=True)
        assert "<video></video>" in template_prompt

        # Dummy text should use the same video placeholder.
        # Import the actual dummy builder's placeholder choice.
        from vllm_reka.multimodal_model import YasaDummyInputsBuilder
        # Inspect get_dummy_text source for video placeholder.
        # We test the contract: a video-only dummy text must contain
        # <video></video> and NOT <REKA_IMG_TOKEN>.
        import inspect
        source = inspect.getsource(YasaDummyInputsBuilder.get_dummy_text)
        # The method should reference <video></video> for videos
        assert "<video></video>" in source or "_VIDEO_PLACEHOLDER" in source, (
            "get_dummy_text should use <video></video> for video placeholders, "
            "matching the chat template. Found <REKA_IMG_TOKEN> instead.")

    def test_dummy_processor_inputs_video_uses_video_tags(self):
        """get_dummy_processor_inputs must use <video></video> for videos."""
        from vllm_reka.multimodal_model import YasaDummyInputsBuilder
        import inspect
        source = inspect.getsource(
            YasaDummyInputsBuilder.get_dummy_processor_inputs)
        # Video parts should use <video></video>, not <REKA_IMG_TOKEN>
        # This is a source-level check since we can't instantiate
        # the builder without a full vLLM context.
        assert "<video></video>" in source or "_VIDEO_PLACEHOLDER" in source, (
            "get_dummy_processor_inputs should use <video></video> for video "
            "placeholders, matching the chat template.")
