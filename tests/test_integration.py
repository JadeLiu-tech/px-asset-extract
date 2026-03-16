"""Integration tests -- end-to-end extraction pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from px_asset_extract import extract_assets, load_regions, ExtractionResult, BBox, Segment


class TestExtractAssets:
    """Test the extract_assets() one-call API."""

    def test_synthetic_end_to_end(self, synthetic_image, tmp_path):
        out = str(tmp_path / "output")
        result = extract_assets(synthetic_image, output_dir=out)

        assert isinstance(result, ExtractionResult)
        assert result.num_assets > 0
        assert result.source_image == "synthetic.png"
        assert result.source_size == (400, 300)
        assert len(result.background_color) == 3

    def test_manifest_created(self, synthetic_image, tmp_path):
        out = str(tmp_path / "output")
        result = extract_assets(synthetic_image, output_dir=out)

        assert result.manifest_path is not None
        manifest_path = Path(result.manifest_path)
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["num_assets"] == result.num_assets
        assert len(manifest["assets"]) == result.num_assets
        assert "source_image" in manifest
        assert "source_size" in manifest
        assert "background_color" in manifest

    def test_visualization_created(self, synthetic_image, tmp_path):
        out = str(tmp_path / "output")
        result = extract_assets(synthetic_image, output_dir=out)

        assert result.visualization_path is not None
        vis_path = Path(result.visualization_path)
        assert vis_path.exists()

        img = Image.open(vis_path)
        assert img.size[0] > 0

    def test_no_visualization_flag(self, synthetic_image, tmp_path):
        out = str(tmp_path / "output")
        result = extract_assets(synthetic_image, output_dir=out, visualization=False)
        assert result.visualization_path is None

    def test_asset_files_exist(self, synthetic_image, tmp_path):
        out = str(tmp_path / "output")
        result = extract_assets(synthetic_image, output_dir=out)

        for asset in result.assets:
            asset_path = Path(out) / asset.file_path
            assert asset_path.exists(), f"Asset file missing: {asset.file_path}"
            img = Image.open(asset_path)
            assert img.mode == "RGBA"

    def test_blank_image_no_assets(self, blank_image, tmp_path):
        out = str(tmp_path / "output")
        result = extract_assets(blank_image, output_dir=out)
        assert result.num_assets == 0

    def test_max_coverage_filter(self, synthetic_image, tmp_path):
        out1 = str(tmp_path / "normal")
        result1 = extract_assets(synthetic_image, output_dir=out1, max_coverage=0.5)

        out2 = str(tmp_path / "strict")
        result2 = extract_assets(synthetic_image, output_dir=out2, max_coverage=0.01)

        # Stricter coverage filter should produce fewer or equal assets
        assert result2.num_assets <= result1.num_assets


class TestRealImages:
    """Test on real images from test_data/."""

    def test_slide_extraction(self, slide_image, tmp_path):
        out = str(tmp_path / "slide_output")
        result = extract_assets(slide_image, output_dir=out)

        assert result.num_assets > 0
        # Slide should have multiple assets (text blocks + graphics)
        assert result.num_assets >= 3, (
            f"Expected at least 3 assets from slide, got {result.num_assets}"
        )

        # Check manifest is valid JSON
        with open(result.manifest_path) as f:
            manifest = json.load(f)
        assert manifest["num_assets"] == result.num_assets

    def test_poster_extraction(self, poster_image, tmp_path):
        out = str(tmp_path / "poster_output")
        result = extract_assets(poster_image, output_dir=out)

        assert result.num_assets > 0

    def test_diagram_extraction(self, diagram_image, tmp_path):
        out = str(tmp_path / "diagram_output")
        result = extract_assets(diagram_image, output_dir=out)

        assert result.num_assets > 0


class TestCLI:
    """Test the CLI entry point."""

    def test_cli_runs_without_error(self, synthetic_image, tmp_path):
        from px_asset_extract.cli import main

        out = str(tmp_path / "cli_output")
        main([synthetic_image, "-o", out, "--quiet"])

        assert (Path(out) / "manifest.json").exists()

    def test_cli_json_output(self, synthetic_image, tmp_path, capsys):
        from px_asset_extract.cli import main

        out = str(tmp_path / "cli_output")
        main([synthetic_image, "-o", out, "--json", "--quiet"])

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert "num_assets" in result

    def test_cli_segments_only(self, synthetic_image, capsys):
        from px_asset_extract.cli import main

        main([synthetic_image, "--segments-only", "--quiet"])

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert "segments" in result
        assert "num_segments" in result

    def test_cli_version(self, capsys):
        from px_asset_extract.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_cli_missing_file(self, tmp_path, capsys):
        from px_asset_extract.cli import main

        main(["/nonexistent/file.png", "-o", str(tmp_path), "--quiet"])
        # Should not crash, just warn

    def test_cli_batch_mode(self, synthetic_image, tmp_path):
        from px_asset_extract.cli import main

        out = str(tmp_path / "batch_output")
        main([synthetic_image, synthetic_image, "-o", out, "--batch", "--quiet"])

    def test_cli_no_visualization(self, synthetic_image, tmp_path):
        from px_asset_extract.cli import main

        out = str(tmp_path / "no_viz")
        main([synthetic_image, "-o", out, "--no-visualization", "--quiet"])

        assert (Path(out) / "manifest.json").exists()
        assert not (Path(out) / "visualization.png").exists()

    def test_cli_types_filter(self, synthetic_image, tmp_path, capsys):
        from px_asset_extract.cli import main

        out = str(tmp_path / "types_output")
        main([synthetic_image, "-o", out, "--types", "text", "--quiet", "--json"])

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        for asset in result.get("assets", []):
            assert "text" in asset["label"]

    def test_cli_exclude_types(self, synthetic_image, tmp_path, capsys):
        from px_asset_extract.cli import main

        out = str(tmp_path / "exclude_output")
        main([synthetic_image, "-o", out, "--exclude-types", "text,line", "--quiet", "--json"])

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        for asset in result.get("assets", []):
            assert "text" not in asset["label"]
            assert "line" not in asset["label"]

    def test_cli_types_and_exclude_types_conflict(self, synthetic_image, tmp_path):
        from px_asset_extract.cli import main

        with pytest.raises(SystemExit):
            main([synthetic_image, "-o", str(tmp_path), "--types", "text", "--exclude-types", "icon", "--quiet"])

    def test_cli_invalid_type(self, synthetic_image, tmp_path):
        from px_asset_extract.cli import main

        with pytest.raises(SystemExit):
            main([synthetic_image, "-o", str(tmp_path), "--types", "bogus", "--quiet"])

    def test_cli_regions(self, synthetic_image, tmp_path, capsys):
        from px_asset_extract.cli import main

        regions_file = tmp_path / "regions.json"
        regions_file.write_text(json.dumps([
            {"x": 50, "y": 30, "width": 60, "height": 40, "label": "red_box"},
            {"x": 200, "y": 150, "width": 20, "height": 20, "label": "blue_box"},
        ]))

        out = str(tmp_path / "regions_output")
        main([synthetic_image, "-o", out, "--regions", str(regions_file), "--quiet", "--json"])

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["num_assets"] == 2

    def test_cli_segments_only_with_types(self, synthetic_image, capsys):
        from px_asset_extract.cli import main

        main([synthetic_image, "--segments-only", "--exclude-types", "text", "--quiet"])

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        for seg in result["segments"]:
            assert seg["label"] != "text"


class TestTypeFiltering:
    """Test type-based filtering in extract_assets()."""

    def test_types_include(self, synthetic_image, tmp_path):
        # Get all assets first
        out_all = str(tmp_path / "all")
        result_all = extract_assets(synthetic_image, output_dir=out_all)

        # Filter to only text
        out_text = str(tmp_path / "text_only")
        result_text = extract_assets(
            synthetic_image, output_dir=out_text, types=["text"]
        )

        assert result_text.num_assets <= result_all.num_assets
        for asset in result_text.assets:
            assert asset.label == "text"

    def test_types_exclude(self, synthetic_image, tmp_path):
        out = str(tmp_path / "no_text")
        result = extract_assets(
            synthetic_image, output_dir=out, exclude_types=["text"]
        )

        for asset in result.assets:
            assert asset.label != "text"

    def test_types_and_exclude_types_error(self, synthetic_image, tmp_path):
        out = str(tmp_path / "error")
        with pytest.raises(ValueError, match="Cannot use both"):
            extract_assets(
                synthetic_image,
                output_dir=out,
                types=["text"],
                exclude_types=["icon"],
            )

    def test_types_empty_result(self, synthetic_image, tmp_path):
        out = str(tmp_path / "none")
        result = extract_assets(
            synthetic_image, output_dir=out, types=["diagram_network"]
        )
        # Synthetic image has no diagram_network segments
        assert result.num_assets == 0


class TestRegions:
    """Test pre-computed region extraction."""

    def test_regions_xywh_format(self, synthetic_image, tmp_path):
        regions = [
            Segment(id=0, bbox=BBox(50, 30, 60, 40), pixel_area=2400, label="red_box"),
            Segment(id=1, bbox=BBox(200, 150, 20, 20), pixel_area=400, label="blue_box"),
        ]

        out = str(tmp_path / "regions")
        result = extract_assets(synthetic_image, output_dir=out, regions=regions)

        assert result.num_assets == 2
        assert result.assets[0].label == "red_box"
        assert result.assets[1].label == "blue_box"

        for asset in result.assets:
            asset_path = Path(out) / asset.file_path
            assert asset_path.exists()
            img = Image.open(asset_path)
            assert img.mode == "RGBA"

    def test_regions_with_type_filter(self, synthetic_image, tmp_path):
        regions = [
            Segment(id=0, bbox=BBox(50, 30, 60, 40), pixel_area=2400, label="chart"),
            Segment(id=1, bbox=BBox(200, 150, 20, 20), pixel_area=400, label="logo"),
        ]

        out = str(tmp_path / "filtered")
        result = extract_assets(
            synthetic_image, output_dir=out, regions=regions, types=["chart"]
        )

        assert result.num_assets == 1
        assert result.assets[0].label == "chart"

    def test_load_regions_flat_list(self, tmp_path):
        regions_file = tmp_path / "regions.json"
        regions_file.write_text(json.dumps([
            {"x": 10, "y": 20, "width": 100, "height": 50, "label": "chart"},
            {"x": 200, "y": 100, "width": 80, "height": 80},
        ]))

        regions = load_regions(str(regions_file))
        assert len(regions) == 2
        assert regions[0].label == "chart"
        assert regions[0].bbox.x == 10
        assert regions[1].label == "region"  # default label

    def test_load_regions_wrapped(self, tmp_path):
        regions_file = tmp_path / "regions.json"
        regions_file.write_text(json.dumps({
            "regions": [
                {"x": 10, "y": 20, "width": 100, "height": 50, "label": "logo"},
            ]
        }))

        regions = load_regions(str(regions_file))
        assert len(regions) == 1
        assert regions[0].label == "logo"

    def test_load_regions_x1y1x2y2_format(self, tmp_path):
        regions_file = tmp_path / "regions.json"
        regions_file.write_text(json.dumps([
            {"x1": 10, "y1": 20, "x2": 110, "y2": 70, "label": "box"},
        ]))

        regions = load_regions(str(regions_file))
        assert len(regions) == 1
        assert regions[0].bbox.x == 10
        assert regions[0].bbox.y == 20
        assert regions[0].bbox.width == 100
        assert regions[0].bbox.height == 50

    def test_load_regions_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_regions(str(tmp_path / "nope.json"))

    def test_load_regions_bad_format(self, tmp_path):
        regions_file = tmp_path / "bad.json"
        regions_file.write_text(json.dumps({"not_regions": []}))

        with pytest.raises(ValueError, match="'regions' key"):
            load_regions(str(regions_file))

    def test_load_regions_missing_keys(self, tmp_path):
        regions_file = tmp_path / "missing.json"
        regions_file.write_text(json.dumps([{"label": "oops"}]))

        with pytest.raises(ValueError, match="must have"):
            load_regions(str(regions_file))
