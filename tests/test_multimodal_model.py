# ABOUTME: Tests that multimodal_model placeholders match chat template output.
# ABOUTME: Verifies dummy text and model placeholders stay consistent with chat template.

import pytest

from vllm_reka.tokenizer import YasaTokenizer


@pytest.fixture()
def tokenizer():
    return YasaTokenizer(tiktoken_model_name="cl100k_base")


class TestDummyTextMatchesChatTemplate:
    """Dummy text placeholders must match what the chat template emits.

    If the chat template emits <video></video> for video items but dummy
    text uses <REKA_IMG_TOKEN>, vLLM's profiling pass will use a different
    token layout than real inference, causing mismatches.
    """

    def test_dummy_text_matches_chat_template(self, tokenizer):
        """Dummy text placeholders must match what the chat template produces."""
        from vllm_reka.multimodal_model import YasaDummyInputsBuilder

        # get_dummy_text doesn't use self, so call it as an unbound method
        dummy = YasaDummyInputsBuilder.get_dummy_text

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
        dummy_text = dummy(None, {"video": 1})
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
        dummy_text = dummy(None, {"image": 1})
        assert "<REKA_IMG_TOKEN>" in template_prompt
        assert "<REKA_IMG_TOKEN>" in dummy_text


class TestPlaceholderStrMatchesChatTemplate:
    """get_placeholder_str must return the same strings as the chat template."""

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
