# ABOUTME: Tests for YasaMMLMV2ImageProcessor with real config values.
# ABOUTME: Verifies tile counts, reordering, and output tensor shapes.

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vllm_reka.edge_model import YasaMMLMV2ImageProcessor


@pytest.fixture(autouse=True)
def _enable_image_patching(monkeypatch):
    """Ensure USE_IMAGE_PATCHING=1 for all tests."""
    monkeypatch.setenv("USE_IMAGE_PATCHING", "1")


@pytest.fixture()
def config():
    return SimpleNamespace(
        vision_config=SimpleNamespace(image_size=224, patch_size=14),
        num_query_tokens=49,
    )


@pytest.fixture()
def processor(config):
    return YasaMMLMV2ImageProcessor(config)


class TestEdgeImageProcessor:

    def test_small_image_produces_one_tile(self, processor):
        """Image smaller than target resolution → 1 tile."""
        image = Image.new("RGB", (200, 200), "red")
        result = processor.preprocess([image])
        assert result["tiles_per_image"].tolist() == [1]

    def test_large_image_produces_multiple_tiles(self, processor):
        """Image much larger than target → multiple tiles."""
        image = Image.new("RGB", (1024, 512), "blue")
        result = processor.preprocess([image])
        tiles = result["tiles_per_image"].item()
        assert tiles > 1, f"Expected multiple tiles, got {tiles}"

    def test_output_pixel_values_shape(self, processor):
        """Output pixel_values is [N, C, H, W] with C=3."""
        image = Image.new("RGB", (400, 400), "green")
        result = processor.preprocess([image])
        pv = result["pixel_values"]
        assert pv.ndim == 4
        assert pv.shape[1] == 3  # RGB channels

    def test_tile_reordering_to_hf_order(self, processor):
        """Edge processor must reorder tiles to [patches..., source] for multi-tile images.

        _preprocess_anyres_image_uhd returns [source, patch0, patch1, ...].
        The HF convention expects [patch0, patch1, ..., source].
        """
        image = Image.new("RGB", (1024, 512))
        image.paste(Image.new("RGB", (512, 512), "red"), (0, 0))
        image.paste(Image.new("RGB", (512, 512), "blue"), (512, 0))

        # Capture the PIL tiles passed to ConvNextImageProcessor
        captured_tiles = []
        original_processor = processor.image_processor

        def capture_call(images, **kwargs):
            captured_tiles.extend(images)
            return original_processor(images, **kwargs)

        processor.image_processor = capture_call
        processor.preprocess([image])

        assert len(captured_tiles) > 1, "Expected multi-tile output"

        # Source tile is the original image (largest). In HF order it should be LAST.
        source_size = image.size  # (1024, 512)
        last_tile = captured_tiles[-1]
        first_tile = captured_tiles[0]
        assert last_tile.size == source_size, (
            f"Source tile (size {source_size}) should be last in HF ordering, "
            f"but last tile has size {last_tile.size}")
        assert first_tile.size != source_size, (
            "Source tile should NOT be first in HF ordering")

    def test_num_images_tensor(self, processor):
        """num_images should be all ones (one image per input)."""
        images = [Image.new("RGB", (100, 100)), Image.new("RGB", (300, 300))]
        result = processor.preprocess(images)
        assert result["num_images"].tolist() == [1, 1]
