"""Tests for the segmenter module."""

from __future__ import annotations

import numpy as np
import pytest

from px_asset_extract.segmenter import segment, segment_array, _connected_components
from px_asset_extract.utils import BBox, Segment


class TestConnectedComponents:
    """Test the connected component labeling algorithm."""

    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        labels, num = _connected_components(mask)
        assert num == 0
        assert labels.max() == 0

    def test_single_blob(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        labels, num = _connected_components(mask)
        assert num == 1
        assert labels[3, 5] == 1
        assert labels[0, 0] == 0

    def test_two_separate_blobs(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:5, 2:5] = 1  # top-left blob
        mask[15:18, 15:18] = 1  # bottom-right blob
        labels, num = _connected_components(mask)
        assert num == 2
        assert labels[3, 3] != labels[16, 16]
        assert labels[3, 3] > 0
        assert labels[16, 16] > 0

    def test_l_shaped_region(self):
        """L-shaped region should be a single component via 8-connectivity."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:5, 1:3] = 1  # vertical part
        mask[4:6, 1:6] = 1  # horizontal part
        labels, num = _connected_components(mask)
        assert num == 1

    def test_diagonal_connectivity(self):
        """Diagonal pixels should be connected (8-connectivity)."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[0, 0] = 1
        mask[1, 1] = 1
        mask[2, 2] = 1
        labels, num = _connected_components(mask)
        assert num == 1


class TestSegment:
    """Test the segment() public API."""

    def test_synthetic_image_finds_objects(self, synthetic_image):
        segments = segment(synthetic_image, min_area=30)
        assert len(segments) > 0
        assert all(isinstance(s, Segment) for s in segments)

    def test_synthetic_image_segment_properties(self, synthetic_image):
        segments = segment(synthetic_image, min_area=30)
        for seg in segments:
            assert seg.bbox.width > 0
            assert seg.bbox.height > 0
            assert seg.pixel_area > 0
            assert 0 <= seg.fill_ratio <= 1

    def test_blank_image_no_segments(self, blank_image):
        segments = segment(blank_image)
        assert len(segments) == 0

    def test_single_color_image_no_segments(self, single_color_image):
        segments = segment(single_color_image)
        assert len(segments) == 0

    def test_real_slide_image(self, slide_image):
        segments = segment(slide_image)
        assert len(segments) > 0
        # Slide should have text segments
        text_segments = [s for s in segments if s.label == "text"]
        assert len(text_segments) > 0, "Expected to find text in slide image"

    def test_real_poster_image(self, poster_image):
        segments = segment(poster_image)
        assert len(segments) > 0

    def test_real_diagram_image(self, diagram_image):
        segments = segment(diagram_image)
        assert len(segments) > 0

    def test_min_area_filter(self, synthetic_image):
        segs_low = segment(synthetic_image, min_area=10)
        segs_high = segment(synthetic_image, min_area=5000)
        assert len(segs_low) >= len(segs_high)

    def test_bg_threshold_sensitivity(self, synthetic_image):
        segs_sensitive = segment(synthetic_image, bg_threshold=5)
        segs_relaxed = segment(synthetic_image, bg_threshold=100)
        # Lower threshold = more sensitive = more or equal segments
        assert len(segs_sensitive) >= len(segs_relaxed)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            segment("/nonexistent/image.png")


class TestSegmentArray:
    """Test the segment_array() function."""

    def test_returns_all_components(self, synthetic_image):
        from px_asset_extract.utils import load_image
        image = load_image(synthetic_image)
        segments, fg_mask, labels, bg_color = segment_array(image, min_area=30)

        assert len(segments) > 0
        assert fg_mask.shape == image.shape[:2]
        assert labels.shape == image.shape[:2]
        assert len(bg_color) == 3
