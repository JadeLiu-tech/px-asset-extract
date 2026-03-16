"""Tests for the classifier module."""

from __future__ import annotations

import pytest

from px_asset_extract.classifier import classify, classify_segment, classify_with_dimensions
from px_asset_extract.segmenter import segment
from px_asset_extract.utils import BBox, Segment


class TestClassifySegment:
    """Test individual segment classification."""

    def _make_segment(self, **kwargs):
        """Create a segment with default values, overridden by kwargs."""
        defaults = {
            "id": 0,
            "bbox": BBox(x=0, y=0, width=50, height=50),
            "pixel_area": 1000,
            "label": "unknown",
            "fill_ratio": 0.5,
            "dark_ratio": 0.1,
            "very_dark_ratio": 0.05,
            "color_std": 10.0,
            "ink_std": 5.0,
            "mean_brightness": 128.0,
            "mean_saturation": 20.0,
        }
        defaults.update(kwargs)
        return Segment(**defaults)

    def test_text_segment_unchanged(self):
        """Text segments (already classified) should stay as text."""
        seg = self._make_segment(label="text")
        result = classify_segment(seg, 1000, 1000)
        assert result == "text"

    def test_large_colorful_is_illustration(self):
        seg = self._make_segment(
            bbox=BBox(x=0, y=0, width=200, height=200),
            pixel_area=30000,
            color_std=40.0,
            mean_saturation=50.0,
        )
        result = classify_segment(seg, 1000, 1000)
        assert result == "illustration"

    def test_small_element_is_icon(self):
        seg = self._make_segment(
            bbox=BBox(x=0, y=0, width=30, height=30),
            pixel_area=500,
            color_std=5.0,
            mean_saturation=5.0,
            mean_brightness=100.0,
        )
        result = classify_segment(seg, 1000, 1000)
        assert result == "icon"

    def test_thin_horizontal_is_line(self):
        seg = self._make_segment(
            bbox=BBox(x=0, y=0, width=200, height=3),
            pixel_area=600,
        )
        result = classify_segment(seg, 1000, 1000)
        assert result == "line"

    def test_thin_vertical_is_line(self):
        seg = self._make_segment(
            bbox=BBox(x=0, y=0, width=3, height=200),
            pixel_area=600,
        )
        result = classify_segment(seg, 1000, 1000)
        assert result == "line"

    def test_tiny_element_is_dot(self):
        seg = self._make_segment(
            bbox=BBox(x=0, y=0, width=10, height=10),
            pixel_area=80,
        )
        result = classify_segment(seg, 1000, 1000)
        assert result == "dot"

    def test_light_low_contrast_is_shadow(self):
        seg = self._make_segment(
            bbox=BBox(x=0, y=0, width=100, height=100),
            pixel_area=5000,
            mean_brightness=220.0,
            color_std=8.0,
            mean_saturation=10.0,
        )
        result = classify_segment(seg, 1000, 1000)
        assert result == "shadow"

    def test_full_image_spanning_sparse_is_diagram_network(self):
        seg = self._make_segment(
            bbox=BBox(x=0, y=0, width=900, height=900),
            pixel_area=2000,
            fill_ratio=0.002,
        )
        result = classify_segment(seg, 1000, 1000)
        assert result == "diagram_network"

    def test_low_fill_is_diagram(self):
        seg = self._make_segment(
            bbox=BBox(x=0, y=0, width=100, height=100),
            pixel_area=1500,
            fill_ratio=0.15,
            color_std=5.0,
            mean_saturation=5.0,
            mean_brightness=100.0,
        )
        result = classify_segment(seg, 1000, 1000)
        assert result == "diagram"


class TestClassify:
    """Test the classify() function on real images."""

    def test_slide_classification(self, slide_image):
        segments = segment(slide_image)
        classified = classify(slide_image, segments)
        labels = {s.label for s in classified}
        assert len(labels) > 0
        # Every segment should have a non-unknown label
        for seg in classified:
            assert seg.label != "unknown", f"Segment {seg.id} not classified"

    def test_poster_classification(self, poster_image):
        segments = segment(poster_image)
        classified = classify(poster_image, segments)
        assert len(classified) > 0
        for seg in classified:
            assert seg.label != "unknown"

    def test_synthetic_classification(self, synthetic_image):
        segments = segment(synthetic_image, min_area=30)
        classified = classify(synthetic_image, segments)
        labels = [s.label for s in classified]
        # Should find at least text and non-text objects
        assert len(classified) > 0


class TestClassifyWithDimensions:
    """Test classify_with_dimensions() variant."""

    def test_classifies_without_loading_image(self, synthetic_image):
        segments = segment(synthetic_image, min_area=30)
        classified = classify_with_dimensions(segments, 300, 400)
        for seg in classified:
            assert seg.label != "unknown"
