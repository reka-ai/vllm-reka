# ABOUTME: Tests for 2D sinusoidal positional embedding generation.
# ABOUTME: Verifies output shape, value bounds, and uniqueness across grid positions.

import torch

from vllm_reka.edge_model import get_2d_sincos_pos_embed


class TestGet2dSincosPosEmbed:

    def test_output_shape_square(self):
        """Square grid produces (H, W, embed_dim) tensor."""
        embed = get_2d_sincos_pos_embed(64, 8)
        assert embed.shape == (8, 8, 64)

    def test_output_shape_int_sizes(self):
        """Various integer sizes produce correct shapes."""
        for size in [1, 4, 16]:
            embed = get_2d_sincos_pos_embed(32, size)
            assert embed.shape == (size, size, 32)

    def test_output_shape_rectangular(self):
        """Rectangular (H, W) tuple produces correct shape."""
        embed = get_2d_sincos_pos_embed(64, (6, 10))
        assert embed.shape == (6, 10, 64)

    def test_values_bounded(self):
        """All values are in [-1, 1] since they are sin/cos outputs."""
        embed = get_2d_sincos_pos_embed(128, 16)
        assert embed.min() >= -1.0
        assert embed.max() <= 1.0

    def test_different_positions_produce_different_embeddings(self):
        """Adjacent grid positions must have distinct embeddings."""
        embed = get_2d_sincos_pos_embed(64, 4)
        # Compare (0,0) vs (0,1) vs (1,0)
        assert not torch.equal(embed[0, 0], embed[0, 1])
        assert not torch.equal(embed[0, 0], embed[1, 0])
        assert not torch.equal(embed[0, 1], embed[1, 0])

    def test_dtype_is_float32(self):
        """Output should be float32."""
        embed = get_2d_sincos_pos_embed(64, 4)
        assert embed.dtype == torch.float32

    def test_single_position(self):
        """1x1 grid still produces valid embeddings."""
        embed = get_2d_sincos_pos_embed(32, 1)
        assert embed.shape == (1, 1, 32)
        assert embed.min() >= -1.0
        assert embed.max() <= 1.0
