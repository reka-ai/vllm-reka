# ABOUTME: Tests that multimodal_model placeholders match chat template output.
# ABOUTME: Verifies dummy text, model placeholders, and chat template stay consistent.

import pytest

from vllm_reka.tokenizer import YasaTokenizer


@pytest.fixture()
def tokenizer():
    return YasaTokenizer(tiktoken_model_name="cl100k_base")


# ── Test group A: Structural guards ──────────────────────────────────


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


# ── Test group B: Dummy text matches chat template ───────────────────


class TestDummyTextPlaceholders:
    """Dummy text must use the same placeholders as the chat template.

    If the chat template emits <video></video> for video items but dummy
    text uses <REKA_IMG_TOKEN>, vLLM's profiling pass will use a different
    token layout than real inference, causing mismatches.
    """

    def test_dummy_video_text(self):
        """get_dummy_text produces <video></video> for videos."""
        from vllm_reka.multimodal_model import YasaDummyInputsBuilder
        builder = YasaDummyInputsBuilder.__new__(YasaDummyInputsBuilder)
        text = builder.get_dummy_text({"video": 1})
        assert "<video></video>" in text
        assert "<REKA_IMG_TOKEN>" not in text

    def test_dummy_image_text(self):
        """get_dummy_text produces <REKA_IMG_TOKEN> for images."""
        from vllm_reka.multimodal_model import YasaDummyInputsBuilder
        builder = YasaDummyInputsBuilder.__new__(YasaDummyInputsBuilder)
        text = builder.get_dummy_text({"image": 1})
        assert "<REKA_IMG_TOKEN>" in text
        assert "<video></video>" not in text

    def test_dummy_mixed_text(self):
        """get_dummy_text produces correct placeholders for mixed inputs."""
        from vllm_reka.multimodal_model import YasaDummyInputsBuilder
        builder = YasaDummyInputsBuilder.__new__(YasaDummyInputsBuilder)
        text = builder.get_dummy_text({"image": 2, "video": 3})
        assert text.count("<REKA_IMG_TOKEN>") == 2
        assert text.count("<video></video>") == 3

    def test_dummy_text_matches_chat_template(self, tokenizer):
        """Dummy text placeholders must match what the chat template emits."""
        from vllm_reka.multimodal_model import YasaDummyInputsBuilder
        builder = YasaDummyInputsBuilder.__new__(YasaDummyInputsBuilder)

        # Video: chat template should emit same placeholder as dummy text
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
        dummy_text = builder.get_dummy_text({"video": 1})
        # Both should use <video></video>
        assert "<video></video>" in template_prompt
        assert "<video></video>" in dummy_text

        # Image: same check
        image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "dummy"},
                    {"type": "text", "text": "x"},
                ],
            }
        ]
        template_prompt = tokenizer.apply_chat_template(
            image_messages, add_generation_prompt=True)
        dummy_text = builder.get_dummy_text({"image": 1})
        assert "<REKA_IMG_TOKEN>" in template_prompt
        assert "<REKA_IMG_TOKEN>" in dummy_text


# ── Test group C: Model placeholder strings ──────────────────────────


class TestModelPlaceholderStrings:
    """get_placeholder_str must return the same strings as the chat template."""

    def test_image_placeholder_str(self):
        from vllm_reka.multimodal_model import YasaMMLMForConditionalGeneration
        result = YasaMMLMForConditionalGeneration.get_placeholder_str("image", 0)
        assert result == "<REKA_IMG_TOKEN>"

    def test_video_placeholder_str(self):
        from vllm_reka.multimodal_model import YasaMMLMForConditionalGeneration
        result = YasaMMLMForConditionalGeneration.get_placeholder_str("video", 0)
        assert result == "<video></video>"

    def test_unknown_modality_returns_none(self):
        from vllm_reka.multimodal_model import YasaMMLMForConditionalGeneration
        result = YasaMMLMForConditionalGeneration.get_placeholder_str("audio", 0)
        assert result is None

    def test_placeholder_str_matches_chat_template(self, tokenizer):
        """Model placeholders must match what the chat template produces."""
        from vllm_reka.multimodal_model import YasaMMLMForConditionalGeneration

        image_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": "d"},
                {"type": "text", "text": "x"},
            ]}],
            add_generation_prompt=True)
        image_placeholder = YasaMMLMForConditionalGeneration.get_placeholder_str(
            "image", 0)
        assert image_placeholder in image_prompt

        video_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "video", "video": "d"},
                {"type": "text", "text": "x"},
            ]}],
            add_generation_prompt=True)
        video_placeholder = YasaMMLMForConditionalGeneration.get_placeholder_str(
            "video", 0)
        assert video_placeholder in video_prompt
