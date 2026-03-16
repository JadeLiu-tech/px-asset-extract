"""px-asset-extract -- Extract individual assets from images.

Decomposes posters, infographics, slides, and diagrams into individual
transparent PNG assets with a JSON manifest. Zero ML models, zero LLM
tokens -- pure classical computer vision using Pillow and numpy.

Quick start::

    from px_asset_extract import extract_assets

    result = extract_assets("poster.png", output_dir="assets/")
    print(f"Extracted {result.num_assets} assets")

Or use individual steps::

    from px_asset_extract import segment, classify, extract_asset

    segments = segment("poster.png")
    classified = classify("poster.png", segments)
    for seg in classified:
        extract_asset("poster.png", seg, "assets/")
"""

from __future__ import annotations

__version__ = "0.1.0"

from pathlib import Path
from typing import List, Optional

from .classifier import classify, classify_segment, classify_with_dimensions
from .extractor import extract_asset, extract_assets_from_segments, extract_crop
from .manifest import build_manifest, save_manifest
from .segmenter import segment, segment_array
from .utils import Asset, BBox, ExtractionResult, Segment
from .visualizer import create_visualization, create_visualization_from_array


def load_regions(regions_path: str) -> List["Segment"]:
    """Load pre-computed bounding boxes from a JSON file.

    Supports multiple formats:

    - Flat list::

        [{"x": 100, "y": 50, "width": 400, "height": 300, "label": "chart"}]

    - Wrapped in a ``regions`` key::

        {"regions": [{"x": 100, "y": 50, "width": 400, "height": 300}]}

    - x1/y1/x2/y2 format::

        [{"x1": 100, "y1": 50, "x2": 500, "y2": 350, "label": "logo"}]

    The ``label`` field is optional and defaults to ``"region"``.

    Args:
        regions_path: Path to JSON file.

    Returns:
        List of Segment objects with bounding boxes and labels set.
    """
    import json

    path = Path(regions_path)
    if not path.exists():
        raise FileNotFoundError(f"Regions file not found: {regions_path}")

    with open(path) as f:
        data = json.load(f)

    # Unwrap {"regions": [...]} wrapper if present
    if isinstance(data, dict):
        if "regions" in data:
            items = data["regions"]
        else:
            raise ValueError(
                "Regions JSON object must contain a 'regions' key"
            )
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("Regions JSON must be a list or object with 'regions' key")

    segments = []
    for idx, item in enumerate(items):
        label = item.get("label", "region")

        # Support x/y/width/height or x1/y1/x2/y2
        if "width" in item and "height" in item:
            bbox = BBox(
                x=int(item["x"]),
                y=int(item["y"]),
                width=int(item["width"]),
                height=int(item["height"]),
            )
        elif "x1" in item and "y1" in item and "x2" in item and "y2" in item:
            x1, y1 = int(item["x1"]), int(item["y1"])
            x2, y2 = int(item["x2"]), int(item["y2"])
            bbox = BBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
        else:
            raise ValueError(
                f"Region {idx} must have (x, y, width, height) or "
                f"(x1, y1, x2, y2) keys"
            )

        segments.append(Segment(
            id=idx,
            bbox=bbox,
            pixel_area=bbox.area,
            label=label,
        ))

    return segments


# Valid segment type labels for --types / --exclude-types filtering
VALID_TYPES = frozenset({
    "text", "illustration", "icon", "graphic", "line",
    "dot", "diagram", "diagram_network", "shadow", "element",
})


def extract_assets(
    image_path: str,
    output_dir: str = "assets",
    *,
    bg_threshold: float = 22.0,
    min_area: int = 60,
    dilation: int = 2,
    text_dark_ratio: float = 0.4,
    text_max_height: int = 200,
    line_gap: int = 35,
    padding: int = 10,
    visualization: bool = True,
    max_coverage: float = 0.5,
    types: Optional[List[str]] = None,
    exclude_types: Optional[List[str]] = None,
    regions: Optional[List["Segment"]] = None,
) -> ExtractionResult:
    """Extract all assets from an image in one call.

    This is the primary entry point. It runs the full pipeline:
    1. Segment the image into connected components
    2. Classify each segment (text, illustration, icon, etc.)
    3. Filter by type (if ``types`` or ``exclude_types`` given)
    4. Filter out segments covering > max_coverage of the image
    5. Extract each segment as a transparent PNG
    6. Generate a JSON manifest and optional visualization

    Args:
        image_path: Path to the input image.
        output_dir: Directory to save extracted assets, manifest, and
            visualization. Created if it does not exist.
        bg_threshold: Euclidean color distance threshold for background
            detection. Lower values detect more foreground.
        min_area: Minimum pixel area for a segment to be extracted.
        dilation: MaxFilter passes to bridge character gaps.
        text_dark_ratio: Minimum dark_ratio for text classification.
        text_max_height: Maximum pixel height for text segments.
        line_gap: Maximum horizontal gap for text-line merging.
        padding: Extra pixels around each extracted bounding box.
        visualization: Whether to generate a bounding box visualization.
        max_coverage: Maximum fraction of image area a single segment
            can cover. Segments exceeding this are filtered out (removes
            full-image artifacts).
        types: Only extract segments matching these types.
            Valid types: text, illustration, icon, graphic, line, dot,
            diagram, diagram_network, shadow, element.
        exclude_types: Skip segments matching these types.
            Cannot be combined with ``types``.
        regions: Pre-computed bounding boxes to extract instead of
            running segmentation. Use ``load_regions()`` to load from
            a JSON file (e.g. output from px-ground).

    Returns:
        ExtractionResult with asset list, manifest path, and
        visualization path.

    Example:
        >>> result = extract_assets("poster.png", output_dir="assets/")
        >>> print(f"Found {result.num_assets} assets")
        >>> for asset in result.assets:
        ...     print(f"  {asset.id}: {asset.label} ({asset.bbox.width}x{asset.bbox.height})")

        >>> # Extract only illustrations and icons
        >>> result = extract_assets("slide.png", types=["illustration", "icon"])

        >>> # Extract from pre-computed regions (e.g. from px-ground)
        >>> regions = load_regions("grounded_regions.json")
        >>> result = extract_assets("slide.png", regions=regions)
    """
    from .utils import (
        create_foreground_mask,
        deduplicate_bboxes,
        detect_background_color,
        load_image,
        smooth_mask,
        tighten_bbox,
    )

    if types is not None and exclude_types is not None:
        raise ValueError("Cannot use both 'types' and 'exclude_types'")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load image
    image = load_image(image_path)
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w

    if regions is not None:
        # -- Regions mode: skip segmentation, use provided bounding boxes --
        segments = regions
        bg_color = detect_background_color(image)
        fg_mask = create_foreground_mask(image, bg_color, threshold=bg_threshold)
        fg_clean = smooth_mask(fg_mask)
    else:
        # -- Normal mode: full segmentation pipeline --
        # Segment
        segments, fg_mask, labels, bg_color = segment_array(
            image,
            bg_threshold=bg_threshold,
            dilation=dilation,
            min_area=min_area,
            text_dark_ratio=text_dark_ratio,
            text_max_height=text_max_height,
            line_gap=line_gap,
        )

        # Classify non-text segments
        segments = classify_with_dimensions(segments, img_h, img_w)

        # Clean up the foreground mask for extraction
        fg_clean = smooth_mask(fg_mask)

        # Tighten bounding boxes to actual content
        for seg in segments:
            seg.bbox = tighten_bbox(fg_clean, seg.bbox)

        # Filter out segments that cover too much of the image
        segments = [
            seg for seg in segments
            if seg.bbox.area < img_area * max_coverage
            and seg.bbox.width > 5
            and seg.bbox.height > 5
        ]

        # Deduplicate overlapping segments
        segments = deduplicate_bboxes(segments)

    # Apply type filter
    if types is not None:
        type_set = set(types)
        segments = [s for s in segments if s.label in type_set]
    elif exclude_types is not None:
        exclude_set = set(exclude_types)
        segments = [s for s in segments if s.label not in exclude_set]

    # Sort top-to-bottom, left-to-right for consistent ordering
    segments.sort(key=lambda s: (s.bbox.y, s.bbox.x))

    # Re-index after sorting
    for idx, seg in enumerate(segments):
        seg.id = idx

    # Extract assets
    assets = extract_assets_from_segments(
        image, fg_clean, segments, output_dir, padding=padding,
    )

    # Build result
    result = ExtractionResult(
        source_image=Path(image_path).name,
        source_size=(img_w, img_h),
        background_color=tuple(int(c) for c in bg_color),
        assets=assets,
    )

    # Save manifest
    manifest_path = str(out_path / "manifest.json")
    save_manifest(result, manifest_path)
    result.manifest_path = manifest_path

    # Create visualization
    if visualization:
        vis_path = str(out_path / "visualization.png")
        create_visualization_from_array(image, segments, vis_path)
        result.visualization_path = vis_path

    return result


__all__ = [
    # Primary API
    "extract_assets",
    "load_regions",
    "segment",
    "classify",
    "extract_asset",
    "extract_crop",
    # Constants
    "VALID_TYPES",
    # Data types
    "Asset",
    "BBox",
    "ExtractionResult",
    "Segment",
    # Secondary API
    "build_manifest",
    "classify_segment",
    "classify_with_dimensions",
    "create_visualization",
    "create_visualization_from_array",
    "extract_assets_from_segments",
    "save_manifest",
    "segment_array",
]
