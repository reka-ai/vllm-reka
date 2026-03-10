# ABOUTME: Tests for image and video token counting used in KV cache allocation.
# ABOUTME: Verifies ImageProcessor and VideoProcessor compute correct token counts.

from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_reka.multimodal_utils import (
    DEFAULT_VIDEO_NUM_FRAMES,
    ImageProcessor,
    VideoProcessor,
)


@pytest.fixture(autouse=True)
def _enable_image_patching(monkeypatch):
    """Ensure USE_IMAGE_PATCHING=1 for all tests."""
    monkeypatch.setenv("USE_IMAGE_PATCHING", "1")


@pytest.fixture()
def config():
    return SimpleNamespace(
        vision_config=SimpleNamespace(image_size=384, patch_size=14),
        vision_max_tiles_num=9,
        num_query_tokens=32,
    )


@pytest.fixture()
def image_processor(config):
    return ImageProcessor(config)


@pytest.fixture()
def video_processor(config):
    return VideoProcessor(config)


class TestImageTokenCounting:

    def test_small_image_tokens(self, image_processor, config):
        """Small image (< 1 tile) → num_query_tokens."""
        image = Image.new("RGB", (200, 200))
        assert image_processor.get_num_image_tokens(image) == config.num_query_tokens

    def test_large_image_tokens(self, image_processor, config):
        """Large image → (tiles + 1) * num_query_tokens."""
        # 1024x1024 on a 384x384 grid = area_ratio ~7.1 → 8 tiles
        image = Image.new("RGB", (1024, 1024))
        tokens = image_processor.get_num_image_tokens(image)
        assert tokens > config.num_query_tokens
        assert tokens % config.num_query_tokens == 0

    def test_from_size_matches_from_image(self, image_processor):
        """get_num_image_tokens_from_size must match get_num_image_tokens for same dimensions."""
        for w, h in [(200, 200), (800, 600), (1024, 1024), (384, 384)]:
            from_image = image_processor.get_num_image_tokens(Image.new("RGB", (w, h)))
            from_size = image_processor.get_num_image_tokens_from_size(w, h)
            assert from_image == from_size, f"Mismatch for ({w}, {h})"

    def test_max_yasa_image_tokens(self, image_processor, config):
        """Max tokens = num_query_tokens * (max_tiles + 1) + 2."""
        expected = config.num_query_tokens * (config.vision_max_tiles_num + 1) + 2
        assert image_processor.get_max_yasa_image_tokens() == expected

    def test_single_tile_boundary(self, image_processor, config):
        """Image exactly at target resolution → single tile → num_query_tokens."""
        size = config.vision_config.image_size
        image = Image.new("RGB", (size, size))
        assert image_processor.get_num_image_tokens(image) == config.num_query_tokens


class TestVideoTokenCounting:

    def test_num_video_tokens(self, video_processor, config):
        """n frames → n * num_query_tokens tokens."""
        for n in [1, 8, 32]:
            assert video_processor.get_num_video_tokens(n) == n * config.num_query_tokens

    def test_max_video_tokens(self, video_processor, config):
        """Default max = DEFAULT_VIDEO_NUM_FRAMES * num_query_tokens."""
        expected = DEFAULT_VIDEO_NUM_FRAMES * config.num_query_tokens
        assert video_processor.get_max_video_tokens() == expected
