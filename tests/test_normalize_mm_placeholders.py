# ABOUTME: Tests for multimodal placeholder normalization in edge model processor.
# ABOUTME: Verifies _rewrite_mm_blocks correctly rewrites image/video blocks to exact counts.

import pytest

from vllm_reka.multimodal_utils import (
    _IMAGE_PLACEHOLDER_TOKEN_ID,
    _START_IMAGE_TOKEN,
    _END_IMAGE_TOKEN,
    _START_VIDEO_TOKEN,
    _END_VIDEO_TOKEN,
    _rewrite_mm_blocks,
)

P = _IMAGE_PLACEHOLDER_TOKEN_ID  # 100278 placeholder
SI = _START_IMAGE_TOKEN           # 100279 <image>
EI = _END_IMAGE_TOKEN             # 100280 </image>
SV = _START_VIDEO_TOKEN           # 100284 <video>
EV = _END_VIDEO_TOKEN             # 100285 </video>


class TestRewriteMmBlocks:
    """Tests for _rewrite_mm_blocks: rewriting multimodal placeholder blocks."""

    def test_single_image_block_exact_count(self):
        """Block with wrong count of placeholders is rewritten to exact count."""
        # Input: <image> + 64 placeholders + </image>
        ids = [10, 20, SI] + [P] * 64 + [EI, 30]
        result = _rewrite_mm_blocks(ids, SI, EI, [192])
        expected = [10, 20, SI] + [P] * 192 + [EI, 30]
        assert result == expected

    def test_single_image_block_shrink(self):
        """Block with too many placeholders is shrunk."""
        ids = [10, SI] + [P] * 200 + [EI, 30]
        result = _rewrite_mm_blocks(ids, SI, EI, [50])
        expected = [10, SI] + [P] * 50 + [EI, 30]
        assert result == expected

    def test_multiple_image_blocks(self):
        """Multiple blocks are each rewritten to their own count."""
        ids = [10, SI] + [P] * 5 + [EI, 20, SI] + [P] * 3 + [EI, 30]
        result = _rewrite_mm_blocks(ids, SI, EI, [10, 20])
        expected = [10, SI] + [P] * 10 + [EI, 20, SI] + [P] * 20 + [EI, 30]
        assert result == expected

    def test_extra_blocks_dropped(self):
        """Blocks beyond what was processed are dropped entirely."""
        ids = [10, SI] + [P] * 5 + [EI, 20, SI] + [P] * 3 + [EI, 30]
        result = _rewrite_mm_blocks(ids, SI, EI, [10])
        # Second block should be dropped
        expected = [10, SI] + [P] * 10 + [EI, 20, 30]
        assert result == expected

    def test_no_counts_passthrough(self):
        """Empty counts list returns tokens unchanged."""
        ids = [10, 20, 30]
        result = _rewrite_mm_blocks(ids, SI, EI, [])
        assert result == ids

    def test_no_blocks_passthrough(self):
        """Tokens without any blocks are returned unchanged."""
        ids = [10, 20, 30]
        result = _rewrite_mm_blocks(ids, SI, EI, [192])
        assert result == ids

    def test_missing_end_token_preserved(self):
        """Start token without matching end is kept as-is."""
        ids = [10, SI, P, P, 30]
        result = _rewrite_mm_blocks(ids, SI, EI, [192])
        assert result == ids

    def test_video_blocks(self):
        """Video blocks are rewritten with video start/end tokens."""
        ids = [10, SV, EV, 30]
        result = _rewrite_mm_blocks(ids, SV, EV, [48])
        expected = [10, SV] + [P] * 48 + [EV, 30]
        assert result == expected

    def test_empty_block_expanded(self):
        """Block with no placeholders between start/end is expanded."""
        ids = [10, SI, EI, 30]
        result = _rewrite_mm_blocks(ids, SI, EI, [192])
        expected = [10, SI] + [P] * 192 + [EI, 30]
        assert result == expected


class TestHfProcessorAppliesUpdates:
    """Tests that wrapped vs bare placeholders are detected correctly.

    Wrapped blocks (<image>...</image>) should signal that updates are
    already applied. Bare <REKA_IMG_TOKEN> placeholders should signal
    that the framework needs to apply updates.
    """

    def test_wrapped_image_detected(self):
        """Prompt with <image>...</image> wrapping is detected."""
        ids = [10, SI, P, P, EI, 30]
        has_wrapped = (SI in ids and EI in ids) or (SV in ids and EV in ids)
        assert has_wrapped is True

    def test_bare_placeholder_not_wrapped(self):
        """Prompt with bare <REKA_IMG_TOKEN> has no wrapping."""
        ids = [10, P, 30]
        has_wrapped = (SI in ids and EI in ids) or (SV in ids and EV in ids)
        assert has_wrapped is False

    def test_wrapped_video_detected(self):
        """Prompt with <video>...</video> wrapping is detected."""
        ids = [10, SV, P, P, EV, 30]
        has_wrapped = (SI in ids and EI in ids) or (SV in ids and EV in ids)
        assert has_wrapped is True
