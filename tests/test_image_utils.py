# ABOUTME: Tests for pure image processing utility functions in multimodal_utils.
# ABOUTME: Covers _ensure_divides, _find_best_resize, _generate_grids, _get_refine_size, _split_into_patches.

from PIL import Image

from vllm_reka.multimodal_utils import (
    _ensure_divides,
    _find_best_resize,
    _generate_grids,
    _get_refine_size,
    _split_into_patches,
)


# ── _ensure_divides ──────────────────────────────────────────────────


class TestEnsureDivides:

    def test_already_divisible(self):
        assert _ensure_divides(28, 14) == 28

    def test_rounds_up(self):
        # 15/14 = 1.07 → round(1.07) = 1 → 1*14 = 14
        assert _ensure_divides(15, 14) == 14

    def test_rounds_to_nearest(self):
        # 21/14 = 1.5 → round(1.5) = 2 → 2*14 = 28
        assert _ensure_divides(21, 14) == 28

    def test_minimum_is_patch_size(self):
        # 1/14 = 0.07 → round(0.07) = 0 → max(0, 14) = 14
        assert _ensure_divides(1, 14) == 14

    def test_zero_gives_patch_size(self):
        assert _ensure_divides(0, 14) == 14

    def test_result_always_divisible(self):
        for length in range(1, 200):
            result = _ensure_divides(length, 14)
            assert result % 14 == 0
            assert result >= 14


# ── _find_best_resize ────────────────────────────────────────────────


class TestFindBestResize:

    def test_small_image_no_upscale(self):
        # Image smaller than target resolution — should keep original aspect
        w, h = _find_best_resize((100, 100), 384, 14, allow_upscale=False)
        assert w % 14 == 0
        assert h % 14 == 0

    def test_large_image_downscale(self):
        # Image larger than target — should downscale to ~384
        w, h = _find_best_resize((2000, 1000), 384, 14, allow_upscale=False)
        assert w % 14 == 0
        assert h % 14 == 0
        # Should be in the ballpark of target resolution
        assert w * h < 2000 * 1000

    def test_upscale_allowed(self):
        w, h = _find_best_resize((100, 100), 384, 14, allow_upscale=True)
        assert w % 14 == 0
        assert h % 14 == 0
        # Should scale up toward target
        assert w >= 100

    def test_preserves_aspect_ratio_approximately(self):
        # 2:1 aspect ratio input
        w, h = _find_best_resize((800, 400), 384, 14, allow_upscale=True)
        # Result should maintain roughly 2:1 ratio
        ratio = w / h
        assert 1.5 < ratio < 2.5

    def test_result_always_divisible_by_patch(self):
        for size in [(100, 200), (500, 500), (1920, 1080), (50, 800)]:
            w, h = _find_best_resize(size, 384, 14, allow_upscale=True)
            assert w % 14 == 0, f"Width {w} not divisible by 14 for {size}"
            assert h % 14 == 0, f"Height {h} not divisible by 14 for {size}"


# ── _generate_grids ──────────────────────────────────────────────────


class TestGenerateGrids:

    def test_prime_number(self):
        # 7 is prime: only 1x7 and 7x1
        grids = _generate_grids(7)
        assert (1, 7) in grids
        assert (7, 1) in grids
        assert len(grids) == 2

    def test_square_number(self):
        grids = _generate_grids(4)
        assert (1, 4) in grids
        assert (2, 2) in grids
        assert (4, 1) in grids
        assert len(grids) == 3

    def test_nine(self):
        grids = _generate_grids(9)
        assert (1, 9) in grids
        assert (3, 3) in grids
        assert (9, 1) in grids

    def test_one(self):
        assert _generate_grids(1) == [(1, 1)]

    def test_product_equals_total(self):
        for n in range(1, 20):
            for x, y in _generate_grids(n):
                assert x * y == n


# ── _get_refine_size ─────────────────────────────────────────────────


class TestGetRefineSize:

    def test_result_divisible_by_grid_and_patch(self):
        w, h = _get_refine_size((1024, 512), (2, 2), 384, 14)
        # Each tile should be divisible by patch_size
        assert (w // 2) % 14 == 0
        assert (h // 2) % 14 == 0

    def test_different_grids_all_divisible(self):
        for grid in [(2, 1), (1, 2), (2, 2), (3, 1)]:
            w, h = _get_refine_size((1024, 512), grid, 384, 14)
            gx, gy = grid
            assert (w // gx) % 14 == 0, (
                f"Grid {grid}: tile width {w // gx} not divisible by 14")
            assert (h // gy) % 14 == 0, (
                f"Grid {grid}: tile height {h // gy} not divisible by 14")


# ── _split_into_patches ──────────────────────────────────────────────


class TestSplitIntoPatches:

    def test_2x2_grid(self):
        image = Image.new("RGB", (100, 100), "red")
        patches = _split_into_patches(image, (2, 2))
        assert len(patches) == 4
        for p in patches:
            assert p.size == (50, 50)

    def test_1x1_grid(self):
        image = Image.new("RGB", (100, 100), "red")
        patches = _split_into_patches(image, (1, 1))
        assert len(patches) == 1
        assert patches[0].size == (100, 100)

    def test_3x1_grid(self):
        image = Image.new("RGB", (300, 100), "red")
        patches = _split_into_patches(image, (3, 1))
        assert len(patches) == 3
        for p in patches:
            assert p.size == (100, 100)

    def test_patches_cover_full_image(self):
        """Patches should tile the full image area."""
        image = Image.new("RGB", (200, 100))
        patches = _split_into_patches(image, (2, 2))
        total_pixels = sum(p.size[0] * p.size[1] for p in patches)
        assert total_pixels == 200 * 100

    def test_patches_contain_correct_regions(self):
        """Each patch should contain content from its region."""
        image = Image.new("RGB", (200, 200))
        # Top-left red, top-right blue, bottom-left green, bottom-right white
        image.paste(Image.new("RGB", (100, 100), "red"), (0, 0))
        image.paste(Image.new("RGB", (100, 100), "blue"), (100, 0))
        image.paste(Image.new("RGB", (100, 100), "green"), (0, 100))
        image.paste(Image.new("RGB", (100, 100), "white"), (100, 100))

        patches = _split_into_patches(image, (2, 2))
        # Row-major order: top-left, top-right, bottom-left, bottom-right
        assert patches[0].getpixel((50, 50)) == (255, 0, 0)    # red
        assert patches[1].getpixel((50, 50)) == (0, 0, 255)    # blue
        assert patches[2].getpixel((50, 50)) == (0, 128, 0)    # green
        assert patches[3].getpixel((50, 50)) == (255, 255, 255) # white
