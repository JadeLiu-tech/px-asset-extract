"""Shared utilities for image loading, bounding box operations, and color distance.

This module provides foundational operations used across the segmentation,
classification, and extraction pipeline. All functions work with numpy arrays
and Pillow images -- no OpenCV required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class BBox:
    """Axis-aligned bounding box in (x, y, width, height) format."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self):
        """Ensure all values are native Python ints (not numpy int64)."""
        self.x = int(self.x)
        self.y = int(self.y)
        self.width = int(self.width)
        self.height = int(self.height)

    @property
    def x2(self) -> int:
        """Right edge (exclusive)."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge (exclusive)."""
        return self.y + self.height

    @property
    def area(self) -> int:
        """Bounding box area in pixels."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Center point (cx, cy)."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def aspect_ratio(self) -> float:
        """Width / height ratio."""
        return self.width / max(self.height, 1)

    def intersection_area(self, other: "BBox") -> int:
        """Compute intersection area with another bounding box."""
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        if ix2 > ix1 and iy2 > iy1:
            return (ix2 - ix1) * (iy2 - iy1)
        return 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)


@dataclass
class Segment:
    """A detected region within an image.

    Represents a connected component or group of merged components with
    computed statistics for classification.
    """

    id: int
    bbox: BBox
    pixel_area: int
    label: str = "unknown"

    # Component IDs that make up this segment (for merged text lines)
    component_ids: List[int] = field(default_factory=list)

    # Classification features
    fill_ratio: float = 0.0
    dark_ratio: float = 0.0
    very_dark_ratio: float = 0.0
    color_std: float = 0.0
    ink_std: float = 0.0
    mean_brightness: float = 0.0
    mean_saturation: float = 0.0
    mean_color: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialize to dictionary (for JSON manifest)."""
        return {
            "id": int(self.id),
            "label": self.label,
            "bbox": self.bbox.to_dict(),
            "pixel_area": int(self.pixel_area),
            "fill_ratio": round(float(self.fill_ratio), 3),
        }


@dataclass
class Asset:
    """An extracted asset with its metadata and file path."""

    id: str
    label: str
    bbox: BBox
    file_path: str
    pixel_area: int
    source_width: int
    source_height: int

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "file": self.file_path,
            "position": self.bbox.to_dict(),
            "pixel_area": int(self.pixel_area),
        }


@dataclass
class ExtractionResult:
    """Complete result from an asset extraction run."""

    source_image: str
    source_size: Tuple[int, int]
    background_color: Tuple[int, int, int]
    assets: List[Asset]
    manifest_path: Optional[str] = None
    visualization_path: Optional[str] = None

    @property
    def num_assets(self) -> int:
        return len(self.assets)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as an RGBA numpy array.

    Args:
        path: Path to the image file.

    Returns:
        RGBA uint8 numpy array of shape (H, W, 4).

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file cannot be opened as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        img = Image.open(path).convert("RGBA")
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Cannot open image: {path} ({e})") from e


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image as an RGB numpy array.

    Args:
        path: Path to the image file.

    Returns:
        RGB uint8 numpy array of shape (H, W, 3).
    """
    rgba = load_image(path)
    return rgba[:, :, :3]


# ---------------------------------------------------------------------------
# Background detection
# ---------------------------------------------------------------------------


def detect_background_color(
    image: np.ndarray, margin: int = 15
) -> np.ndarray:
    """Detect the dominant background color by sampling border pixels.

    Samples pixels from all four edges of the image and computes the
    median color, which is robust to outliers (e.g., content near edges).

    Args:
        image: RGBA or RGB uint8 array of shape (H, W, 3+).
        margin: Number of pixels to sample from each edge.

    Returns:
        Background color as uint8 array of shape (3,) in RGB.
    """
    h, w = image.shape[:2]
    m = min(margin, h // 4, w // 4)
    if m < 1:
        m = 1

    edges = np.concatenate([
        image[:m, :, :3].reshape(-1, 3),
        image[-m:, :, :3].reshape(-1, 3),
        image[:, :m, :3].reshape(-1, 3),
        image[:, -m:, :3].reshape(-1, 3),
    ], axis=0)

    return np.median(edges, axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Foreground mask
# ---------------------------------------------------------------------------


def create_foreground_mask(
    image: np.ndarray,
    background_color: np.ndarray,
    threshold: float = 22.0,
) -> np.ndarray:
    """Create a binary foreground mask using Euclidean color distance.

    Each pixel is classified as foreground if its Euclidean distance from
    the background color exceeds the threshold.

    Args:
        image: RGBA or RGB uint8 array of shape (H, W, 3+).
        background_color: Background RGB as uint8 array of shape (3,).
        threshold: Minimum Euclidean distance to count as foreground.

    Returns:
        Binary uint8 mask (H, W) where 1 = foreground, 0 = background.
    """
    diff = image[:, :, :3].astype(np.float32) - background_color.astype(np.float32)
    distance = np.sqrt(np.sum(diff ** 2, axis=2))
    return (distance > threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# Morphological operations (Pillow-based, no OpenCV required)
# ---------------------------------------------------------------------------


def dilate_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Dilate a binary mask using Pillow's MaxFilter.

    This bridges small gaps between nearby components (e.g., characters
    in a word) without requiring OpenCV.

    Args:
        mask: Binary uint8 mask (H, W) with values 0 or 1.
        radius: Number of MaxFilter(3) passes to apply.

    Returns:
        Dilated binary uint8 mask (H, W).
    """
    img = Image.fromarray(mask * 255, mode="L")
    for _ in range(radius):
        img = img.filter(ImageFilter.MaxFilter(3))
    return (np.array(img) > 127).astype(np.uint8)


def erode_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Erode a binary mask using Pillow's MinFilter.

    Args:
        mask: Binary uint8 mask (H, W) with values 0 or 1.
        radius: Number of MinFilter(3) passes to apply.

    Returns:
        Eroded binary uint8 mask (H, W).
    """
    img = Image.fromarray(mask * 255, mode="L")
    for _ in range(radius):
        img = img.filter(ImageFilter.MinFilter(3))
    return (np.array(img) > 127).astype(np.uint8)


def smooth_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological close + open to clean up a mask.

    Close fills small holes, open removes small noise spots.
    Implemented with Pillow filters (no OpenCV needed).

    Args:
        mask: Binary uint8 mask (H, W) with values 0 or 1.

    Returns:
        Cleaned binary uint8 mask (H, W).
    """
    # Close: dilate then erode (fills small gaps)
    closed = dilate_mask(mask, radius=1)
    closed = erode_mask(closed, radius=1)
    # Open: erode then dilate (removes small noise)
    opened = erode_mask(closed, radius=1)
    opened = dilate_mask(opened, radius=1)
    return opened


# ---------------------------------------------------------------------------
# Bounding box operations
# ---------------------------------------------------------------------------


def tighten_bbox(
    mask: np.ndarray, bbox: BBox, padding: int = 4
) -> BBox:
    """Tighten a bounding box to the actual foreground content within it.

    Scans the mask within the bounding box region and shrinks to fit
    the actual non-zero pixels, with optional padding.

    Args:
        mask: Binary uint8 mask (H, W).
        bbox: Original bounding box.
        padding: Pixels of padding to add around tight bounds.

    Returns:
        Tightened bounding box.
    """
    img_h, img_w = mask.shape[:2]
    x1 = max(0, bbox.x)
    y1 = max(0, bbox.y)
    x2 = min(img_w, bbox.x2)
    y2 = min(img_h, bbox.y2)

    sub = mask[y1:y2, x1:x2]
    rows = np.any(sub > 0, axis=1)
    cols = np.any(sub > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        return bbox

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    nx = max(0, x1 + cmin - padding)
    ny = max(0, y1 + rmin - padding)
    nw = min(img_w - nx, (cmax - cmin) + 2 * padding + 1)
    nh = min(img_h - ny, (rmax - rmin) + 2 * padding + 1)

    return BBox(x=nx, y=ny, width=nw, height=nh)


def deduplicate_bboxes(
    segments: List[Segment], overlap_threshold: float = 0.3
) -> List[Segment]:
    """Remove segments whose bounding box is significantly contained within a larger one.

    Processes segments from largest to smallest. A smaller segment is
    removed if the intersection with any kept segment exceeds
    ``overlap_threshold`` of the smaller segment's area.

    Args:
        segments: List of segments to deduplicate.
        overlap_threshold: Fraction of overlap needed to mark as duplicate.

    Returns:
        Deduplicated list of segments.
    """
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s.bbox.area, reverse=True)
    keep: List[Segment] = []

    for seg in sorted_segs:
        is_dup = False
        for kept in keep:
            inter = seg.bbox.intersection_area(kept.bbox)
            if inter > seg.bbox.area * overlap_threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(seg)

    return keep
