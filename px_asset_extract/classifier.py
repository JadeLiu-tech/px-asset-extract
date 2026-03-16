"""Segment classification based on visual features.

Classifies each segment into one of these types:
- **text**: Dark pixels with uniform ink color (letters, numbers, labels)
- **illustration**: Large, colorful region (photos, drawings, charts)
- **icon**: Small, distinct graphical element
- **graphic**: Medium-sized colored element (buttons, badges, shapes)
- **line**: Thin horizontal or vertical element (separators, rules)
- **dot**: Very small element (bullet points, decorators)
- **diagram**: Low-fill structural element (wireframes, flowchart boxes)
- **diagram_network**: Image-spanning low-fill structure (connector network)
- **shadow**: Light, low-contrast region
- **element**: Catch-all for unclassified non-text objects

Classification uses simple heuristics on pre-computed features:
dark_ratio, color_std, fill_ratio, saturation, brightness, and
aspect ratio. No ML models required.
"""

from __future__ import annotations

from typing import List

from .utils import Segment


def classify_segment(
    segment: Segment,
    image_height: int,
    image_width: int,
) -> str:
    """Classify a single segment by its visual features.

    Text segments (already labeled by the segmenter) are returned as-is.
    Non-text segments are classified by size, color, shape, and fill ratio.

    Args:
        segment: The segment to classify.
        image_height: Source image height in pixels.
        image_width: Source image width in pixels.

    Returns:
        Classification label string.

    Example:
        >>> label = classify_segment(seg, 1080, 1920)
        >>> print(label)  # "illustration", "icon", "text", etc.
    """
    # Text segments are already classified by the segmenter
    if segment.label == "text":
        return "text"

    w = segment.bbox.width
    h = segment.bbox.height
    area = segment.pixel_area
    fill = segment.fill_ratio
    brightness = segment.mean_brightness
    color_std = segment.color_std
    saturation = segment.mean_saturation
    img_area = image_height * image_width
    aspect = segment.bbox.aspect_ratio

    # Full-image spanning thin elements = diagram/connector network
    if w > image_width * 0.8 and h > image_height * 0.8 and fill < 0.05:
        return "diagram_network"

    # Thin lines (separators, rules)
    if min(w, h) <= 5 and max(w, h) > 15:
        return "line"
    if min(w, h) <= 8 and (aspect > 8 or aspect < 0.125):
        return "line"

    # Small dots (bullet points, decorators)
    if area < 150 and max(w, h) < 20:
        return "dot"

    # Icons (small, distinct elements)
    if area < 3000 and max(w, h) < 60:
        return "icon"

    # Shadows (very light, low contrast)
    if brightness > 200 and color_std < 12 and saturation < 15:
        return "shadow"

    # Large illustrations (colorful and big)
    if w * h > img_area * 0.01 and (color_std > 25 or saturation > 30):
        return "illustration"

    # Medium graphics (colored shapes, buttons)
    if saturation > 15 or color_std > 15:
        return "graphic"

    # Diagram elements (low fill, structural)
    if fill < 0.25:
        return "diagram"

    return "element"


def classify(
    image_path: str,
    segments: List[Segment],
) -> List[Segment]:
    """Classify all segments in an image.

    Loads image dimensions from the file and applies ``classify_segment``
    to each segment, updating the ``label`` field in place.

    Args:
        image_path: Path to the source image (used for dimensions).
        segments: List of segments from ``segment()``.

    Returns:
        The same list of segments with updated labels.

    Example:
        >>> from px_asset_extract import segment, classify
        >>> segs = segment("poster.png")
        >>> classified = classify("poster.png", segs)
        >>> for seg in classified:
        ...     print(f"{seg.label}: {seg.bbox.width}x{seg.bbox.height}")
    """
    from .utils import load_image

    image = load_image(image_path)
    img_h, img_w = image.shape[:2]

    return classify_with_dimensions(segments, img_h, img_w)


def classify_with_dimensions(
    segments: List[Segment],
    image_height: int,
    image_width: int,
) -> List[Segment]:
    """Classify all segments given known image dimensions.

    Like ``classify()`` but avoids re-loading the image when you already
    have the dimensions.

    Args:
        segments: List of segments to classify.
        image_height: Source image height.
        image_width: Source image width.

    Returns:
        The same list of segments with updated labels.
    """
    for seg in segments:
        seg.label = classify_segment(seg, image_height, image_width)
    return segments
