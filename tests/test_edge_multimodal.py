# ABOUTME: Tests for Edge multimodal processing: video sampling, prompt updates, and chat template.
# ABOUTME: Covers _sample_indices, _get_prompt_updates video target, and bare placeholder output.

import numpy as np
import pytest

from vllm_reka.multimodal_utils import (
    YasaVideoBackend,
    _IMAGE_PLACEHOLDER_TOKEN_ID,
    _START_VIDEO_TOKEN,
    _END_VIDEO_TOKEN,
)
from vllm_reka.tokenizer import (
    DEFAULT_CHAT_TEMPLATE,
    YasaTokenizer,
)


# ── Test group A: Video chunk sampling ──────────────────────────────────


class TestSampleIndices:
    """Tests for YasaVideoBackend._sample_indices."""

    def test_uniform_mode(self):
        result = YasaVideoBackend._sample_indices(4, 100, "uniform")
        expected = np.array([0, 33, 66, 99])
        np.testing.assert_array_equal(result, expected)

    def test_chunk_mode_bounds(self):
        np.random.seed(42)
        result = YasaVideoBackend._sample_indices(4, 100, "chunk")
        assert len(result) == 4
        # Each frame should be within its chunk bounds
        chunk_size = 100 // 4
        extra = 100 % 4
        for i in range(4):
            start = i * chunk_size + min(i, extra)
            end = start + chunk_size + (1 if i < extra else 0)
            assert start <= result[i] < end, (
                f"Frame {i}: {result[i]} not in [{start}, {end})")

    def test_chunk_mode_differs_from_uniform(self):
        """Chunk sampling should produce different indices than uniform."""
        uniform = YasaVideoBackend._sample_indices(4, 100, "uniform")
        # Run chunk many times — at least one run should differ from uniform
        found_different = False
        for seed in range(100):
            np.random.seed(seed)
            chunk = YasaVideoBackend._sample_indices(4, 100, "chunk")
            if not np.array_equal(chunk, uniform):
                found_different = True
                break
        assert found_different, (
            "Chunk sampling always matched uniform across 100 seeds")

    def test_all_frames(self):
        result = YasaVideoBackend._sample_indices(-1, 10, "chunk")
        np.testing.assert_array_equal(result, np.arange(10))

    def test_fewer_than_requested(self):
        result = YasaVideoBackend._sample_indices(32, 10, "chunk")
        np.testing.assert_array_equal(result, np.arange(10))

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unsupported video sampling"):
            YasaVideoBackend._sample_indices(4, 100, "bad_mode")


# ── Test group B: Prompt update video target ────────────────────────────


class TestPromptUpdateVideoTarget:
    """Tests that the tokenized video placeholder matches the prompt update target.

    The prompt update system in _get_prompt_updates searches for a "target"
    token sequence in the tokenized prompt and replaces it with the full
    expansion. The video target must be [<video>, </video>] (token IDs
    100284, 100285) — a 2-token pair that the template emits for each video.
    """

    @pytest.fixture()
    def tokenizer(self):
        return YasaTokenizer(tiktoken_model_name="cl100k_base")

    def test_video_placeholder_tokenizes_to_start_end_pair(self, tokenizer):
        """Tokenized video message should contain [100284, 100285] as target."""
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
        # The video placeholder should tokenize to the <video></video> pair
        assert _START_VIDEO_TOKEN in ids_list
        assert _END_VIDEO_TOKEN in ids_list
        start_idx = ids_list.index(_START_VIDEO_TOKEN)
        # </video> should immediately follow <video>
        assert ids_list[start_idx + 1] == _END_VIDEO_TOKEN
        # No image placeholder token between them
        assert _IMAGE_PLACEHOLDER_TOKEN_ID not in ids_list[start_idx:start_idx + 2]

    def test_image_placeholder_tokenizes_to_img_token(self, tokenizer):
        """Tokenized image message should contain 100278, not 100284/100285."""
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
        # Image should NOT produce <video></video> tokens
        assert _START_VIDEO_TOKEN not in ids_list
        assert _END_VIDEO_TOKEN not in ids_list


# ── Test group C: Chat template output ──────────────────────────────────


class TestChatTemplateOutput:
    """Tests that the chat template emits bare placeholders."""

    @pytest.fixture()
    def tokenizer(self):
        return YasaTokenizer(tiktoken_model_name="cl100k_base")

    def test_image_message(self, tokenizer):
        """Single image + text produces exact expected prompt."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "dummy"},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True)
        assert prompt == "human: <REKA_IMG_TOKEN>describe<sep>assistant:"

    def test_video_message(self, tokenizer):
        """Single video + text produces exact expected prompt."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "dummy"},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True)
        assert prompt == "human: <video></video>describe<sep>assistant:"

    def test_image_and_video_interleaved(self, tokenizer):
        """Mixed image + video + text produces correct placeholders."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "dummy"},
                    {"type": "video", "video": "dummy"},
                    {"type": "text", "text": "compare"},
                ],
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True)
        expected = (
            "human: <REKA_IMG_TOKEN><video></video>compare<sep>assistant:")
        assert prompt == expected

    def test_jinja_template_image(self):
        """Jinja path produces bare <REKA_IMG_TOKEN> for images."""
        from transformers.utils.chat_template_utils import render_jinja_template

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "dummy"}},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        rendered, _ = render_jinja_template(
            conversations=[messages],
            tools=None,
            documents=None,
            chat_template=DEFAULT_CHAT_TEMPLATE,
            return_assistant_tokens_mask=False,
            continue_final_message=False,
            add_generation_prompt=True,
        )
        assert rendered[0] == "human: <REKA_IMG_TOKEN>describe<sep>assistant:"

    def test_jinja_template_video(self):
        """Jinja path produces <video></video> for videos."""
        from transformers.utils.chat_template_utils import render_jinja_template

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "dummy"},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        rendered, _ = render_jinja_template(
            conversations=[messages],
            tools=None,
            documents=None,
            chat_template=DEFAULT_CHAT_TEMPLATE,
            return_assistant_tokens_mask=False,
            continue_final_message=False,
            add_generation_prompt=True,
        )
        assert rendered[0] == "human: <video></video>describe<sep>assistant:"
