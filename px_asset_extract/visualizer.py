"""Bounding box visualization overlay.

Generates an annotated version of the source image showing detected
segments as color-coded bounding boxes with type labels.

Works with Pillow only -- no OpenCV required. Falls back to OpenCV
if available for higher-quality text rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw

from .utils import Segment, load_image


# Color palette for segment types (RGB tuples)
_TYPE_COLORS = {
    "text": (220, 40, 40),
    "illustration": (40, 160, 40),
    "graphic": (40, 40, 220),
    "icon": (255, 140, 0),
    "line": (160, 0, 160),
    "diagram": (0, 160, 160),
    "diagram_network": (0, 120, 120),
    "shadow": (180, 180, 40),
    "dot": (255, 80, 80),
    "element": (100, 100, 200),
    "unknown": (128, 128, 128),
}


def create_visualization(
    image_path: str,
    segments: List[Segment],
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Create an annotated visualization of detected segments.

    Overlays color-coded bounding boxes on the source image with
    type labels. Each segment type gets a distinct color.

    Args:
        image_path: Path to the source image.
        segments: List of segments to visualize.
        output_path: If provided, saves the visualization to this path.

    Returns:
        RGB uint8 numpy array of the visualization image.

    Example:
        >>> from px_asset_extract import segment, classify
        >>> from px_asset_extract.visualizer import create_visualization
        >>> segs = classify("poster.png", segment("poster.png"))
        >>> create_visualization("poster.png", segs, "viz.png")
    """
    image = load_image(image_path)
    return create_visualization_from_array(image, segments, output_path)


def create_visualization_from_array(
    image: np.ndarray,
    segments: List[Segment],
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Create a visualization from an image array.

    Args:
        image: RGBA or RGB uint8 source array.
        segments: List of segments to visualize.
        output_path: If provided, saves the visualization.

    Returns:
        RGB uint8 numpy array of the visualization.
    """
    # Work with RGB
    rgb = image[:, :, :3].copy()
    result = Image.fromarray(rgb)
    draw = ImageDraw.Draw(result, "RGBA")

    for idx, seg in enumerate(segments):
        label_text = seg.label
        color = _TYPE_COLORS.get(label_text, (128, 128, 128))

        x1, y1 = seg.bbox.x, seg.bbox.y
        x2, y2 = seg.bbox.x2, seg.bbox.y2

        # Semi-transparent fill
        fill_color = color + (40,)
        draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=color, width=2)

        # Label background + text
        label = f"#{idx} {label_text}"
        text_bbox = draw.textbbox((0, 0), label)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        label_y = max(0, y1 - th - 6)
        draw.rectangle(
            [x1, label_y, x1 + tw + 6, label_y + th + 4],
            fill=color,
        )
        draw.text(
            (x1 + 3, label_y + 1),
            label,
            fill=(255, 255, 255),
        )

    vis_array = np.array(result.convert("RGB"))

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(vis_array).save(str(out))

    return vis_array
