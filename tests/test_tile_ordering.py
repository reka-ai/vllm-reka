# ABOUTME: Tests for image tile ordering in the edge v2 model's image processor.
# ABOUTME: Verifies tiles are in HF ordering (patches first, source last).

from unittest.mock import MagicMock

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


# ── Test group B: Edge processor tile ordering ───────────────────────


class TestEdgeV2TileOrdering:
    """Edge v2 model must reorder tiles to [patches..., source].

    _preprocess_anyres_image_uhd returns [source, patch0, patch1, ...].
    The HF convention (image_processing_yasa2, how the model was trained)
    expects [patch0, patch1, ..., source].

    The fork applies this reordering in YasaMMLMV2ImageProcessor.preprocess():
        if len(image_tiles) > 1:
            image_tiles = list(image_tiles[1:]) + [image_tiles[0]]

    The plugin is currently missing this step.
    """

    @pytest.fixture()
    def edge_processor(self):
        from vllm_reka.edge_model import YasaMMLMV2ImageProcessor
        config = MagicMock()
        config.vision_config.image_size = 384
        config.vision_config.patch_size = 14
        config.num_query_tokens = 64
        return YasaMMLMV2ImageProcessor(config)

    @pytest.mark.xfail(
        reason="Reordering from [source, patches] to [patches, source] "
               "is not yet implemented in YasaMMLMV2ImageProcessor.preprocess",
        strict=True,
    )
    def test_edge_processor_reorders_tiles_to_hf_order(self, edge_processor):
        """Edge processor should output [patches..., source] for multi-tile images."""
        image = _make_two_color_image()

        # Capture the PIL tiles that get passed to ConvNextImageProcessor
        captured_tiles = []
        original_processor = edge_processor.image_processor

        def capture_call(images, **kwargs):
            captured_tiles.extend(images)
            return original_processor(images, **kwargs)

        edge_processor.image_processor = capture_call
        edge_processor.preprocess([image])

        assert len(captured_tiles) > 1, "Expected multi-tile output"

        # The source tile is the original image (largest, has both colors).
        # In HF order, it should be LAST.
        source_size = image.size  # (1024, 512)
        last_tile = captured_tiles[-1]
        first_tile = captured_tiles[0]
        assert last_tile.size == source_size, (
            f"Source tile (size {source_size}) should be last in HF ordering, "
            f"but last tile has size {last_tile.size}")
        assert first_tile.size != source_size, (
            f"Source tile should NOT be first in HF ordering")
