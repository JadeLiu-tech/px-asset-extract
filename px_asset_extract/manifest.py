"""JSON manifest generation for extracted assets.

Produces a structured JSON file describing all extracted assets,
their positions, types, and file paths. The manifest is designed
to be consumed by downstream tools for layout reconstruction,
asset management, or further processing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import Asset, ExtractionResult


def build_manifest(result: ExtractionResult) -> Dict[str, Any]:
    """Build a manifest dictionary from an extraction result.

    Args:
        result: The extraction result containing assets and metadata.

    Returns:
        Dictionary ready for JSON serialization.

    Example:
        >>> manifest = build_manifest(result)
        >>> print(json.dumps(manifest, indent=2))
    """
    return {
        "source_image": result.source_image,
        "source_size": {
            "width": result.source_size[0],
            "height": result.source_size[1],
        },
        "background_color": list(result.background_color),
        "num_assets": result.num_assets,
        "assets": [asset.to_dict() for asset in result.assets],
    }


def save_manifest(
    result: ExtractionResult,
    output_path: str,
) -> str:
    """Save a JSON manifest file for the extraction result.

    Args:
        result: The extraction result.
        output_path: Path for the output JSON file.

    Returns:
        Absolute path to the saved manifest file.

    Example:
        >>> path = save_manifest(result, "output/manifest.json")
        >>> print(f"Manifest saved to: {path}")
    """
    manifest = build_manifest(result)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)

    return str(out.resolve())


def load_manifest(path: str) -> Dict[str, Any]:
    """Load a manifest JSON file.

    Args:
        path: Path to the manifest JSON file.

    Returns:
        Parsed manifest dictionary.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(p) as f:
        return json.load(f)
