"""Tests for the extractor module."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from px_asset_extract.extractor import extract_asset, extract_crop, extract_assets_from_segments
from px_asset_extract.segmenter import segment, segment_array
from px_asset_extract.classifier import classify_with_dimensions
from px_asset_extract.utils import BBox, Segment, load_image


class TestExtractCrop:
    """Test the extract_crop() function."""

    def test_produces_rgba_output(self, synthetic_image):
        image = load_image(synthetic_image)
        h, w = image.shape[:2]

        from px_asset_extract.utils import create_foreground_mask, detect_background_color
        bg = detect_background_color(image)
        mask = create_foreground_mask(image, bg)

        bbox = BBox(x=50, y=30, width=60, height=40)
        crop = extract_crop(image, mask, bbox, padding=5)

        assert crop.ndim == 3
        assert crop.shape[2] == 4  # RGBA
        assert crop.dtype == np.uint8

    def test_crop_dimensions_include_padding(self, synthetic_image):
        image = load_image(synthetic_image)
        from px_asset_extract.utils import create_foreground_mask, detect_background_color
        bg = detect_background_color(image)
        mask = create_foreground_mask(image, bg)

        bbox = BBox(x=100, y=100, width=50, height=30)
        padding = 10
        crop = extract_crop(image, mask, bbox, padding=padding)

        # Crop should be roughly bbox size + 2*padding
        assert crop.shape[1] >= bbox.width
        assert crop.shape[0] >= bbox.height

    def test_alpha_channel_has_transparency(self, synthetic_image):
        image = load_image(synthetic_image)
        from px_asset_extract.utils import create_foreground_mask, detect_background_color
        bg = detect_background_color(image)
        mask = create_foreground_mask(image, bg)

        # Crop a region that includes some background
        bbox = BBox(x=45, y=25, width=70, height=50)
        crop = extract_crop(image, mask, bbox, padding=15)

        alpha = crop[:, :, 3]
        # Should have both transparent (0) and opaque (255) pixels
        assert alpha.min() < 128, "Expected some transparent pixels"
        assert alpha.max() > 128, "Expected some opaque pixels"

    def test_edge_bbox_clamped(self, synthetic_image):
        """Bounding box near image edge should not cause out-of-bounds."""
        image = load_image(synthetic_image)
        from px_asset_extract.utils import create_foreground_mask, detect_background_color
        bg = detect_background_color(image)
        mask = create_foreground_mask(image, bg)

        # Bbox at very edge of image
        h, w = image.shape[:2]
        bbox = BBox(x=w - 20, y=h - 20, width=30, height=30)
        crop = extract_crop(image, mask, bbox, padding=10)
        assert crop.shape[2] == 4


class TestExtractAsset:
    """Test the extract_asset() function."""

    def test_creates_png_file(self, synthetic_image, tmp_path):
        segments = segment(synthetic_image, min_area=30)
        if not segments:
            pytest.skip("No segments found in synthetic image")

        asset = extract_asset(
            synthetic_image, segments[0], str(tmp_path), padding=5,
        )

        assert asset.file_path.endswith(".png")
        full_path = tmp_path / asset.file_path
        assert full_path.exists()

    def test_output_is_rgba_png(self, synthetic_image, tmp_path):
        segments = segment(synthetic_image, min_area=30)
        if not segments:
            pytest.skip("No segments found")

        asset = extract_asset(
            synthetic_image, segments[0], str(tmp_path), padding=5,
        )

        full_path = tmp_path / asset.file_path
        img = Image.open(full_path)
        assert img.mode == "RGBA"

    def test_asset_metadata(self, synthetic_image, tmp_path):
        segments = segment(synthetic_image, min_area=30)
        if not segments:
            pytest.skip("No segments found")

        asset = extract_asset(
            synthetic_image, segments[0], str(tmp_path), padding=5,
        )

        assert asset.bbox.width > 0
        assert asset.bbox.height > 0
        assert asset.pixel_area > 0
        assert asset.source_width == 400
        assert asset.source_height == 300


class TestExtractAssetsFromSegments:
    """Test batch extraction."""

    def test_extracts_all_segments(self, synthetic_image, tmp_path):
        image = load_image(synthetic_image)
        segments, fg_mask, labels, bg = segment_array(image, min_area=30)
        segments = classify_with_dimensions(segments, *image.shape[:2])

        assets = extract_assets_from_segments(
            image, fg_mask, segments, str(tmp_path), padding=5,
        )

        assert len(assets) == len(segments)
        for asset in assets:
            full_path = tmp_path / asset.file_path
            assert full_path.exists()
            img = Image.open(full_path)
            assert img.mode == "RGBA"

    def test_creates_output_dir(self, synthetic_image, tmp_path):
        image = load_image(synthetic_image)
        segments, fg_mask, labels, bg = segment_array(image, min_area=30)

        new_dir = str(tmp_path / "new_subdir")
        assets = extract_assets_from_segments(
            image, fg_mask, segments, new_dir, padding=5,
        )

        assert (tmp_path / "new_subdir").exists()
