"""Asset extraction -- crop segments as transparent PNGs.

Extracts each detected segment from the source image as an individual
RGBA PNG with a transparent background. Uses the foreground mask to
produce clean alpha channels with anti-aliased edges.

Two implementations are provided:
- **Pillow-only** (default): Uses Pillow's GaussianBlur for soft alpha edges.
- **OpenCV** (optional): Uses cv2 morphological operations for higher-quality
  anti-aliased edges with dilation + Gaussian blur.

The OpenCV path is used automatically when opencv-python is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageFilter

from .utils import Asset, BBox, Segment


# ---------------------------------------------------------------------------
# Try to import OpenCV (optional dependency)
# ---------------------------------------------------------------------------

try:
    import cv2
    _HAS_OPENCV = True
except ImportError:
    _HAS_OPENCV = False


# ---------------------------------------------------------------------------
# Alpha extraction
# ---------------------------------------------------------------------------


def _extract_with_alpha_pillow(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: BBox,
    padding: int = 10,
) -> np.ndarray:
    """Crop a segment with transparent background using Pillow only.

    Args:
        image: RGBA uint8 source array (H, W, 4).
        mask: Binary uint8 foreground mask (H, W), values 0 or 1.
        bbox: Bounding box of the segment.
        padding: Extra pixels around the bounding box.

    Returns:
        RGBA uint8 array of the cropped segment.
    """
    img_h, img_w = image.shape[:2]
    x1 = max(0, bbox.x - padding)
    y1 = max(0, bbox.y - padding)
    x2 = min(img_w, bbox.x2 + padding)
    y2 = min(img_h, bbox.y2 + padding)

    crop = image[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2].copy()

    # Convert mask to 0-255 range
    mask_255 = (mask_crop * 255).astype(np.uint8)

    # Soften edges with Gaussian blur for anti-aliasing
    mask_pil = Image.fromarray(mask_255, mode="L")
    # Dilate slightly first to include edge pixels
    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(3))
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=1))
    mask_soft = np.array(mask_pil)

    # Build RGBA output
    if crop.shape[2] >= 4:
        result = crop[:, :, :4].copy()
        result[:, :, 3] = mask_soft
    else:
        result = np.dstack([crop[:, :, :3], mask_soft])

    return result


def _extract_with_alpha_opencv(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: BBox,
    padding: int = 10,
) -> np.ndarray:
    """Crop a segment with transparent background using OpenCV.

    Uses morphological dilation + Gaussian blur for higher-quality
    anti-aliased alpha edges.

    Args:
        image: RGBA uint8 source array (H, W, 4).
        mask: Binary uint8 foreground mask (H, W), values 0 or 1.
        bbox: Bounding box of the segment.
        padding: Extra pixels around the bounding box.

    Returns:
        RGBA uint8 array of the cropped segment.
    """
    img_h, img_w = image.shape[:2]
    x1 = max(0, bbox.x - padding)
    y1 = max(0, bbox.y - padding)
    x2 = min(img_w, bbox.x2 + padding)
    y2 = min(img_h, bbox.y2 + padding)

    crop = image[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2].copy()

    mask_255 = (mask_crop * 255).astype(np.uint8)

    # Dilate for anti-aliased edges, blur for soft alpha
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(mask_255, kernel, iterations=1)
    mask_soft = cv2.GaussianBlur(mask_dilated, (3, 3), 0)

    if crop.shape[2] >= 4:
        result = crop[:, :, :4].copy()
        result[:, :, 3] = mask_soft
    else:
        result = np.dstack([crop[:, :, :3], mask_soft])

    return result


def extract_crop(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: BBox,
    padding: int = 10,
) -> np.ndarray:
    """Crop a segment as an RGBA array with transparent background.

    Automatically uses OpenCV if available for better edge quality,
    otherwise falls back to Pillow-only implementation.

    Args:
        image: RGBA uint8 source array (H, W, 4).
        mask: Binary uint8 foreground mask (H, W), values 0 or 1.
        bbox: Bounding box of the segment.
        padding: Extra pixels around the bounding box.

    Returns:
        RGBA uint8 numpy array of the cropped segment.
    """
    if _HAS_OPENCV:
        return _extract_with_alpha_opencv(image, mask, bbox, padding)
    else:
        return _extract_with_alpha_pillow(image, mask, bbox, padding)


def extract_asset(
    image_path: str,
    segment: Segment,
    output_dir: str,
    *,
    padding: int = 10,
    bg_threshold: float = 22.0,
) -> Asset:
    """Extract a single segment as a transparent PNG file.

    Loads the image, creates a foreground mask, crops the segment region,
    and saves as an RGBA PNG.

    Args:
        image_path: Path to the source image.
        segment: Segment to extract.
        output_dir: Directory to save the PNG file.
        padding: Extra pixels around the bounding box.
        bg_threshold: Background detection threshold.

    Returns:
        Asset object with metadata and file path.

    Example:
        >>> from px_asset_extract import segment, extract_asset
        >>> segs = segment("poster.png")
        >>> asset = extract_asset("poster.png", segs[0], "output/")
    """
    from .utils import (
        create_foreground_mask,
        detect_background_color,
        load_image,
        smooth_mask,
    )

    image = load_image(image_path)
    img_h, img_w = image.shape[:2]

    bg_color = detect_background_color(image)
    fg_mask = create_foreground_mask(image, bg_color, threshold=bg_threshold)
    fg_mask = smooth_mask(fg_mask)

    return _save_asset(image, fg_mask, segment, output_dir, padding, img_w, img_h)


def extract_assets_from_segments(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    segments: List[Segment],
    output_dir: str,
    *,
    padding: int = 10,
) -> List[Asset]:
    """Extract all segments as transparent PNGs.

    Lower-level function that accepts pre-computed image and mask arrays.
    Used internally by ``extract_assets()`` to avoid re-loading the image.

    Args:
        image: RGBA uint8 source array (H, W, 4).
        foreground_mask: Binary uint8 foreground mask (H, W).
        segments: List of segments to extract.
        output_dir: Directory to save PNG files.
        padding: Extra pixels around bounding boxes.

    Returns:
        List of Asset objects with metadata and file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_h, img_w = image.shape[:2]
    assets = []

    for seg in segments:
        asset = _save_asset(image, foreground_mask, seg, output_dir, padding, img_w, img_h)
        assets.append(asset)

    return assets


def _save_asset(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    segment: Segment,
    output_dir: str,
    padding: int,
    img_w: int,
    img_h: int,
) -> Asset:
    """Save a single segment as a transparent PNG and return the Asset."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    crop = extract_crop(image, foreground_mask, segment.bbox, padding=padding)

    asset_id = f"asset_{segment.id:03d}_{segment.label}"
    filename = f"{asset_id}.png"
    filepath = out_path / filename

    Image.fromarray(crop, "RGBA").save(str(filepath), optimize=True)

    return Asset(
        id=asset_id,
        label=segment.label,
        bbox=segment.bbox,
        file_path=filename,
        pixel_area=segment.pixel_area,
        source_width=img_w,
        source_height=img_h,
    )
