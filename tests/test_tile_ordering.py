# ABOUTME: Tests for _preprocess_anyres_image_uhd tile output ordering.
# ABOUTME: Verifies source tile position and patch sizes for multi-tile images.

import numpy as np
import pytest
from PIL import Image

from vllm_reka.multimodal_utils import ImageProcessor


@pytest.fixture(autouse=True)
def _enable_image_patching(monkeypatch):
    """Ensure USE_IMAGE_PATCHING=1 for all tests."""
    monkeypatch.setenv("USE_IMAGE_PATCHING", "1")


def _make_two_color_image():
    """Create a 1024x512 image: left half red, right half blue."""
    image = Image.new("RGB", (1024, 512))
    image.paste(Image.new("RGB", (512, 512), "red"), (0, 0))
    image.paste(Image.new("RGB", (512, 512), "blue"), (512, 0))
    return image


# ── Test group A: Raw utility output ─────────────────────────────────


class TestPreprocessAnyresImageUhd:
    """Tests for _preprocess_anyres_image_uhd tile output."""

    def test_single_tile_no_split(self):
        """Small image produces 1 tile."""
        image = Image.new("RGB", (200, 200), "red")
        tiles = ImageProcessor._preprocess_anyres_image_uhd(
            image, max_tiles=4, target_resolution=384, patch_size=14)
        assert len(tiles) == 1

    def test_multi_tile_returns_source_first(self):
        """Large image returns source as tile[0], followed by patches."""
        image = _make_two_color_image()
        tiles = ImageProcessor._preprocess_anyres_image_uhd(
            image, max_tiles=4, target_resolution=384, patch_size=14)
        assert len(tiles) > 1, f"Expected multi-tile, got {len(tiles)}"
        # tile[0] is the original image object (the source)
        assert tiles[0] is image

    def test_source_contains_full_image_content(self):
        """Source tile has content from both halves of the image."""
        image = _make_two_color_image()
        tiles = ImageProcessor._preprocess_anyres_image_uhd(
            image, max_tiles=4, target_resolution=384, patch_size=14)
        assert len(tiles) > 1

        source_arr = np.array(tiles[0])
        _, w = source_arr.shape[:2]
        left_red = source_arr[:, :w // 4, 0].mean()
        right_blue = source_arr[:, 3 * w // 4, 2].mean()
        assert left_red > 100, "Source left quarter should be reddish"
        assert right_blue > 100, "Source right quarter should be bluish"

    def test_patches_are_smaller_than_source(self):
        """Patches are sub-regions, smaller than the source."""
        image = _make_two_color_image()
        tiles = ImageProcessor._preprocess_anyres_image_uhd(
            image, max_tiles=4, target_resolution=384, patch_size=14)
        assert len(tiles) > 1
        source_pixels = tiles[0].size[0] * tiles[0].size[1]
        for patch in tiles[1:]:
            patch_pixels = patch.size[0] * patch.size[1]
            assert patch_pixels < source_pixels
