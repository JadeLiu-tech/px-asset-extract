"""Image segmentation via connected components and text-line merging.

Finds meaningful objects in an image using classical computer vision:
1. Detect background color from border pixels
2. Create foreground mask via Euclidean color distance
3. Dilate mask to bridge character gaps
4. Label connected components (two-pass union-find, 8-connectivity)
5. Analyze each component (dark_ratio, color stats, fill ratio)
6. Merge text-like components into text lines (union-find)

No ML models, no LLM tokens -- pure Pillow + numpy.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .utils import (
    BBox,
    Segment,
    create_foreground_mask,
    detect_background_color,
    dilate_mask,
    load_image,
)


# ---------------------------------------------------------------------------
# Connected component labeling (two-pass union-find, 8-connectivity)
# ---------------------------------------------------------------------------


def _connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label connected components in a binary mask.

    Uses a two-pass algorithm with union-find and 8-connectivity
    (each pixel connects to all 8 neighbors). Implemented in pure
    numpy/Python -- no OpenCV required.

    Args:
        mask: Binary uint8 mask (H, W) with values 0 or 1.

    Returns:
        Tuple of (labels, num_components):
        - labels: int32 array (H, W) with component IDs (1-indexed).
        - num_components: total number of components found.
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]  # index 0 unused; labels start at 1
    next_label = 1

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Neighbor offsets for 8-connectivity (only backward neighbors)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

    # First pass: assign provisional labels and record equivalences
    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0:
                continue

            neighbors = []
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] > 0:
                    neighbors.append(labels[ny, nx])

            if not neighbors:
                labels[y, x] = next_label
                parent.append(next_label)
                next_label += 1
            else:
                min_label = min(neighbors)
                labels[y, x] = min_label
                for n in neighbors:
                    union(min_label, n)

    # Second pass: remap to consecutive IDs
    remap = {}
    new_id = 0
    for y in range(h):
        for x in range(w):
            if labels[y, x] > 0:
                root = find(labels[y, x])
                if root not in remap:
                    new_id += 1
                    remap[root] = new_id
                labels[y, x] = remap[root]

    return labels, new_id


# ---------------------------------------------------------------------------
# Component analysis
# ---------------------------------------------------------------------------

# Minimum pixels to consider a component (noise filter)
_MIN_NOISE_PIXELS = 30
_MIN_OBJECT_DIM = 4


def _analyze_components(
    labels: np.ndarray,
    num_labels: int,
    image: np.ndarray,
    foreground_mask: np.ndarray,
) -> List[dict]:
    """Compute per-component statistics for classification.

    For each component, computes: bounding box, pixel area, fill ratio,
    mean color, color standard deviation, brightness, dark_ratio (fraction
    of pixels with brightness < 120), very_dark_ratio (< 80), saturation,
    and ink_std (color std of the darkest 50% of pixels, which ignores
    anti-alias fringe for cleaner text detection).

    Args:
        labels: Component label array from _connected_components().
        num_labels: Number of labeled components.
        image: RGBA uint8 source array (H, W, 4).
        foreground_mask: Undilated binary foreground mask (H, W).

    Returns:
        List of component dictionaries with computed statistics.
    """
    components = []

    for i in range(1, num_labels + 1):
        ys, xs = np.where(labels == i)
        if len(ys) < _MIN_NOISE_PIXELS:
            continue

        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        pixel_area = len(ys)
        comp_w = x2 - x1 + 1
        comp_h = y2 - y1 + 1

        if comp_w < _MIN_OBJECT_DIM or comp_h < _MIN_OBJECT_DIM:
            continue

        # Get pixel colors -- prefer undilated foreground pixels for accuracy
        component_mask = labels == i
        real_fg = component_mask & (foreground_mask > 0)
        use_mask = real_fg if real_fg.sum() > 5 else component_mask
        pixels = image[use_mask][:, :3].astype(np.float32)

        mean_color = pixels.mean(axis=0)
        color_std = float(np.std(pixels, axis=0).mean())
        mean_brightness = float(mean_color.mean())

        # Dark ratio: fraction of pixels with average brightness < 120
        brightness_per_pixel = np.mean(pixels, axis=1)
        dark_count = np.sum(brightness_per_pixel < 120)
        dark_ratio = dark_count / len(pixels) if len(pixels) > 0 else 0.0

        # Very dark ratio: fraction with brightness < 80
        very_dark_count = np.sum(brightness_per_pixel < 80)
        very_dark_ratio = very_dark_count / len(pixels) if len(pixels) > 0 else 0.0

        # Saturation: mean (max_channel - min_channel) per pixel
        max_channel = pixels.max(axis=1)
        min_channel = pixels.min(axis=1)
        mean_saturation = float(np.mean(max_channel - min_channel))

        # Ink std: color std of only the darkest 50% of pixels.
        # Ignores anti-alias fringe for cleaner text detection.
        if len(brightness_per_pixel) > 10:
            median_brightness = np.median(brightness_per_pixel)
            dark_half = pixels[brightness_per_pixel <= median_brightness]
            ink_std = float(np.std(dark_half, axis=0).mean()) if len(dark_half) > 5 else color_std
        else:
            ink_std = color_std

        components.append({
            "id": i,
            "bbox": (x1, y1, x2, y2),  # (x1, y1, x2, y2) format internally
            "pixel_area": pixel_area,
            "width": comp_w,
            "height": comp_h,
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
            "mean_color": mean_color,
            "color_std": color_std,
            "mean_brightness": mean_brightness,
            "fill_ratio": pixel_area / max(comp_w * comp_h, 1),
            "dark_ratio": float(dark_ratio),
            "very_dark_ratio": float(very_dark_ratio),
            "mean_saturation": mean_saturation,
            "ink_std": ink_std,
        })

    return components


# ---------------------------------------------------------------------------
# Text classification
# ---------------------------------------------------------------------------


def _is_text_component(
    comp: dict,
    img_h: int,
    img_w: int,
    dark_ratio_threshold: float = 0.4,
    max_text_height: int = 200,
) -> bool:
    """Classify a component as text using dark_ratio as the primary signal.

    Anti-aliased text has high color_std due to edge pixels blending with
    the background. ``dark_ratio`` (fraction of truly dark pixels) correctly
    identifies text regardless of anti-aliasing.

    Criteria:
    - Height must be <= max_text_height
    - Bounding box area must be <= 5% of image area
    - Strong text: dark_ratio > threshold, moderate fill, low ink_std
    - Very dark text: very_dark_ratio > 0.5, moderate fill
    - Small text: dark_ratio > 0.3, short height, reasonable aspect ratio

    Args:
        comp: Component dictionary from _analyze_components().
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        dark_ratio_threshold: Minimum dark_ratio for standard text detection.
        max_text_height: Maximum pixel height for text components.

    Returns:
        True if the component is classified as text.
    """
    w, h = comp["width"], comp["height"]
    dark_r = comp["dark_ratio"]
    vdark_r = comp["very_dark_ratio"]
    fill = comp["fill_ratio"]
    ink_std = comp["ink_std"]
    bbox_area = w * h
    img_area = img_h * img_w

    # Size constraints
    if h > max_text_height:
        return False
    if bbox_area > img_area * 0.05:
        return False

    # Strong text: mostly dark pixels with uniform ink color
    if dark_r > dark_ratio_threshold and 0.15 < fill < 0.95:
        if ink_std < 35:
            return True

    # Very dark text (black)
    if vdark_r > 0.5 and fill > 0.15:
        return True

    # Small text with moderate darkness
    if dark_r > 0.3 and h < 80 and w > h * 0.3 and ink_std < 25:
        return True

    return False


# ---------------------------------------------------------------------------
# Text-line merging (union-find)
# ---------------------------------------------------------------------------

# Maximum vertical center offset as fraction of average height
_TEXT_LINE_MAX_VOFF_RATIO = 0.5


def _merge_text_lines(
    text_components: List[dict], line_gap: int = 35
) -> List[dict]:
    """Merge word-level text components into text lines using union-find.

    Transitively merges text components that are horizontally aligned
    (similar vertical center) with similar font size (similar height).

    Args:
        text_components: List of text component dictionaries.
        line_gap: Maximum horizontal gap between words on the same line.

    Returns:
        List of merged text-line component dictionaries.
    """
    if len(text_components) <= 1:
        return text_components

    # Sort top-to-bottom, left-to-right
    text_components.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))

    n = len(text_components)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        ci = text_components[i]
        for j in range(i + 1, n):
            cj = text_components[j]

            hi, hj = ci["height"], cj["height"]
            # Similar height (same font size)?
            if max(hi, hj) > min(hi, hj) * 2.5:
                continue

            # Vertical center alignment
            avg_h = (hi + hj) / 2
            v_off = abs(ci["center_y"] - cj["center_y"])
            if v_off > avg_h * _TEXT_LINE_MAX_VOFF_RATIO:
                continue

            # Horizontal gap: distance between right edge of left box
            # and left edge of right box
            gap = max(
                0,
                max(ci["bbox"][0], cj["bbox"][0])
                - min(ci["bbox"][2], cj["bbox"][2]),
            )
            if gap < line_gap:
                union(i, j)

    # Group by root
    groups: dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Build merged components
    merged = []
    for indices in groups.values():
        parts = [text_components[i] for i in indices]
        x1 = min(p["bbox"][0] for p in parts)
        y1 = min(p["bbox"][1] for p in parts)
        x2 = max(p["bbox"][2] for p in parts)
        y2 = max(p["bbox"][3] for p in parts)
        total_area = sum(p["pixel_area"] for p in parts)
        ids = [p["id"] for p in parts]

        merged.append({
            "id": ids[0],
            "component_ids": ids,
            "bbox": (x1, y1, x2, y2),
            "pixel_area": total_area,
            "width": x2 - x1 + 1,
            "height": y2 - y1 + 1,
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
            "mean_color": np.mean([p["mean_color"] for p in parts], axis=0),
            "color_std": float(np.mean([p["color_std"] for p in parts])),
            "mean_brightness": float(np.mean([p["mean_brightness"] for p in parts])),
            "fill_ratio": total_area / max((x2 - x1 + 1) * (y2 - y1 + 1), 1),
            "dark_ratio": float(np.mean([p["dark_ratio"] for p in parts])),
            "very_dark_ratio": float(np.mean([p["very_dark_ratio"] for p in parts])),
            "mean_saturation": float(np.mean([p["mean_saturation"] for p in parts])),
            "ink_std": float(np.mean([p["ink_std"] for p in parts])),
            "label": "text",
            "num_parts": len(indices),
        })

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment(
    image_path: str,
    *,
    bg_threshold: float = 22.0,
    dilation: int = 2,
    min_area: int = 60,
    text_dark_ratio: float = 0.4,
    text_max_height: int = 200,
    line_gap: int = 35,
) -> List[Segment]:
    """Segment an image into individual objects.

    Detects background, creates a foreground mask, finds connected
    components, classifies text vs non-text, and merges text components
    into text lines.

    Args:
        image_path: Path to the input image.
        bg_threshold: Euclidean color distance threshold for foreground
            detection. Lower values are more sensitive.
        dilation: Number of MaxFilter passes to bridge character gaps.
        min_area: Minimum pixel area for a segment to be kept.
        text_dark_ratio: Minimum dark_ratio for text classification.
        text_max_height: Maximum pixel height for text components.
        line_gap: Maximum horizontal gap for text-line merging.

    Returns:
        List of Segment objects, sorted by area (largest first).

    Example:
        >>> segments = segment("poster.png")
        >>> for seg in segments:
        ...     print(f"{seg.label}: {seg.bbox.width}x{seg.bbox.height}")
    """
    image = load_image(image_path)
    h, w = image.shape[:2]

    # Step 1: Background detection
    bg_color = detect_background_color(image)

    # Step 2: Foreground mask
    fg_mask = create_foreground_mask(image, bg_color, threshold=bg_threshold)

    # Step 3: Dilation to bridge character gaps
    dilated = dilate_mask(fg_mask, radius=dilation)

    # Step 4: Connected component labeling
    labels, num_components = _connected_components(dilated)

    # Step 5: Component analysis
    components = _analyze_components(labels, num_components, image, fg_mask)

    # Step 6: Text classification
    text_comps = []
    nontext_comps = []
    for comp in components:
        if _is_text_component(
            comp, h, w,
            dark_ratio_threshold=text_dark_ratio,
            max_text_height=text_max_height,
        ):
            comp["label"] = "text"
            text_comps.append(comp)
        else:
            nontext_comps.append(comp)

    # Step 7: Text-line merging
    text_lines = _merge_text_lines(text_comps, line_gap=line_gap)

    # Step 8: Add component_ids for non-text
    for comp in nontext_comps:
        comp["component_ids"] = [comp["id"]]
        comp["label"] = "unknown"  # classified later by classifier module

    # Combine all objects
    all_objects = text_lines + nontext_comps

    # Filter by minimum area
    all_objects = [
        obj for obj in all_objects
        if obj["pixel_area"] >= min_area
        and obj["width"] >= _MIN_OBJECT_DIM
        and obj["height"] >= _MIN_OBJECT_DIM
    ]

    # Sort by area (largest first)
    all_objects.sort(key=lambda c: c["pixel_area"], reverse=True)

    # Convert to Segment objects
    segments = []
    for idx, obj in enumerate(all_objects):
        x1, y1, x2, y2 = obj["bbox"]
        bbox = BBox(x=x1, y=y1, width=x2 - x1 + 1, height=y2 - y1 + 1)

        seg = Segment(
            id=idx,
            bbox=bbox,
            pixel_area=obj["pixel_area"],
            label=obj.get("label", "unknown"),
            component_ids=obj.get("component_ids", [obj["id"]]),
            fill_ratio=obj["fill_ratio"],
            dark_ratio=obj["dark_ratio"],
            very_dark_ratio=obj["very_dark_ratio"],
            color_std=obj["color_std"],
            ink_std=obj["ink_std"],
            mean_brightness=obj["mean_brightness"],
            mean_saturation=obj["mean_saturation"],
            mean_color=obj.get("mean_color"),
        )
        segments.append(seg)

    return segments


def segment_array(
    image: np.ndarray,
    *,
    bg_threshold: float = 22.0,
    dilation: int = 2,
    min_area: int = 60,
    text_dark_ratio: float = 0.4,
    text_max_height: int = 200,
    line_gap: int = 35,
) -> Tuple[List[Segment], np.ndarray, np.ndarray, np.ndarray]:
    """Segment an image array and return intermediate results.

    Like ``segment()`` but accepts a numpy array directly and also returns
    the foreground mask, component labels, and background color needed
    for downstream extraction.

    Args:
        image: RGBA uint8 numpy array (H, W, 4).
        bg_threshold: Foreground color distance threshold.
        dilation: MaxFilter passes for character gap bridging.
        min_area: Minimum pixel area for segments.
        text_dark_ratio: dark_ratio threshold for text detection.
        text_max_height: Max pixel height for text components.
        line_gap: Max horizontal gap for text-line merging.

    Returns:
        Tuple of (segments, foreground_mask, labels, background_color).
    """
    h, w = image.shape[:2]

    bg_color = detect_background_color(image)
    fg_mask = create_foreground_mask(image, bg_color, threshold=bg_threshold)
    dilated = dilate_mask(fg_mask, radius=dilation)
    labels, num_components = _connected_components(dilated)
    components = _analyze_components(labels, num_components, image, fg_mask)

    text_comps = []
    nontext_comps = []
    for comp in components:
        if _is_text_component(
            comp, h, w,
            dark_ratio_threshold=text_dark_ratio,
            max_text_height=text_max_height,
        ):
            comp["label"] = "text"
            text_comps.append(comp)
        else:
            nontext_comps.append(comp)

    text_lines = _merge_text_lines(text_comps, line_gap=line_gap)

    for comp in nontext_comps:
        comp["component_ids"] = [comp["id"]]
        comp["label"] = "unknown"

    all_objects = text_lines + nontext_comps
    all_objects = [
        obj for obj in all_objects
        if obj["pixel_area"] >= min_area
        and obj["width"] >= _MIN_OBJECT_DIM
        and obj["height"] >= _MIN_OBJECT_DIM
    ]
    all_objects.sort(key=lambda c: c["pixel_area"], reverse=True)

    segments = []
    for idx, obj in enumerate(all_objects):
        x1, y1, x2, y2 = obj["bbox"]
        bbox = BBox(x=x1, y=y1, width=x2 - x1 + 1, height=y2 - y1 + 1)
        seg = Segment(
            id=idx,
            bbox=bbox,
            pixel_area=obj["pixel_area"],
            label=obj.get("label", "unknown"),
            component_ids=obj.get("component_ids", [obj["id"]]),
            fill_ratio=obj["fill_ratio"],
            dark_ratio=obj["dark_ratio"],
            very_dark_ratio=obj["very_dark_ratio"],
            color_std=obj["color_std"],
            ink_std=obj["ink_std"],
            mean_brightness=obj["mean_brightness"],
            mean_saturation=obj["mean_saturation"],
            mean_color=obj.get("mean_color"),
        )
        segments.append(seg)

    return segments, fg_mask, labels, bg_color
