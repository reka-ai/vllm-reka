# ABOUTME: Tests that token ID constants match what the tokenizer actually encodes.
# ABOUTME: Catches drift between tokenizer special tokens and hardcoded constants.

import pytest

from vllm_reka.multimodal_utils import (
    _END_IMAGE_TOKEN,
    _END_VIDEO_TOKEN,
    _IMAGE_PLACEHOLDER_TOKEN_ID,
    _START_IMAGE_TOKEN,
    _START_VIDEO_TOKEN,
)
from vllm_reka.tokenizer import YasaTokenizer


@pytest.fixture()
def tokenizer():
    return YasaTokenizer(tiktoken_model_name="cl100k_base")


class TestTokenIdConsistency:
    """Hardcoded token IDs must match what the tokenizer encodes."""

    def test_image_placeholder_token_id(self, tokenizer):
        ids = tokenizer.encode("<REKA_IMG_TOKEN>", add_special_tokens=False)
        assert ids == [_IMAGE_PLACEHOLDER_TOKEN_ID]

    def test_start_image_token(self, tokenizer):
        ids = tokenizer.encode("<image>", add_special_tokens=False)
        assert ids == [_START_IMAGE_TOKEN]

    def test_end_image_token(self, tokenizer):
        ids = tokenizer.encode("</image>", add_special_tokens=False)
        assert ids == [_END_IMAGE_TOKEN]

    def test_start_video_token(self, tokenizer):
        ids = tokenizer.encode("<video>", add_special_tokens=False)
        assert ids == [_START_VIDEO_TOKEN]

    def test_end_video_token(self, tokenizer):
        ids = tokenizer.encode("</video>", add_special_tokens=False)
        assert ids == [_END_VIDEO_TOKEN]
