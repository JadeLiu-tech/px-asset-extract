"""Command-line interface for px-asset-extract.

Usage::

    # Extract assets from a single image
    px-extract poster.png -o assets/

    # Custom thresholds
    px-extract infographic.png -o assets/ --min-area 500 --bg-threshold 30

    # Segment only (no extraction, just JSON output)
    px-extract poster.png --segments-only -o segments.json

    # Batch processing
    px-extract images/*.png -o output/ --batch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="px-extract",
        description="Extract individual assets from images as transparent PNGs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  px-extract poster.png -o assets/
  px-extract slide.png -o output/ --min-area 500 --bg-threshold 30
  px-extract diagram.png --segments-only
  px-extract images/*.png -o output/ --batch
  px-extract slide.png --types illustration,icon -o icons/
  px-extract slide.png --exclude-types text,line -o graphics/
  px-extract slide.png --regions grounded.json -o targeted/
""",
    )

    parser.add_argument(
        "images",
        nargs="+",
        help="Input image path(s)",
    )
    parser.add_argument(
        "-o", "--output",
        default="assets",
        help="Output directory (default: assets/)",
    )
    parser.add_argument(
        "--bg-threshold",
        type=float,
        default=22.0,
        help="Background color distance threshold (default: 22.0)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=60,
        help="Minimum segment area in pixels (default: 60)",
    )
    parser.add_argument(
        "--dilation",
        type=int,
        default=2,
        help="Character gap bridging passes (default: 2)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Extra pixels around each extracted asset (default: 10)",
    )
    parser.add_argument(
        "--max-coverage",
        type=float,
        default=0.5,
        help="Max fraction of image a segment can cover (default: 0.5)",
    )
    parser.add_argument(
        "--types",
        type=str,
        default=None,
        help="Only extract these segment types (comma-separated). "
        "Valid: text,illustration,icon,graphic,line,dot,diagram,"
        "diagram_network,shadow,element",
    )
    parser.add_argument(
        "--exclude-types",
        type=str,
        default=None,
        help="Skip these segment types (comma-separated). "
        "Cannot be combined with --types.",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default=None,
        help="JSON file with pre-computed bounding boxes to extract "
        "(e.g. from px-ground). Skips segmentation.",
    )
    parser.add_argument(
        "--segments-only",
        action="store_true",
        help="Output segment data as JSON without extracting PNGs",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip generating the visualization image",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple images, creating subdirectories per image",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    return parser.parse_args(argv)


def _parse_types(raw: str | None) -> list[str] | None:
    """Parse a comma-separated type list into a validated list."""
    if raw is None:
        return None
    from . import VALID_TYPES
    types = [t.strip() for t in raw.split(",") if t.strip()]
    bad = [t for t in types if t not in VALID_TYPES]
    if bad:
        raise SystemExit(
            f"Error: unknown type(s): {', '.join(bad)}. "
            f"Valid: {', '.join(sorted(VALID_TYPES))}"
        )
    return types


def _process_segments_only(image_path: str, args) -> dict:
    """Run segmentation + classification and return JSON-serializable result."""
    from . import classify, load_regions, segment

    if args.regions:
        segments = load_regions(args.regions)
    else:
        segments = segment(
            image_path,
            bg_threshold=args.bg_threshold,
            min_area=args.min_area,
            dilation=args.dilation,
        )
        segments = classify(image_path, segments)

    types = _parse_types(args.types)
    exclude_types = _parse_types(args.exclude_types)

    if types is not None:
        type_set = set(types)
        segments = [s for s in segments if s.label in type_set]
    elif exclude_types is not None:
        exclude_set = set(exclude_types)
        segments = [s for s in segments if s.label not in exclude_set]

    return {
        "source": Path(image_path).name,
        "num_segments": len(segments),
        "segments": [seg.to_dict() for seg in segments],
    }


def _process_image(image_path: str, output_dir: str, args) -> dict:
    """Run full extraction pipeline on a single image."""
    from . import extract_assets, load_regions

    types = _parse_types(args.types)
    exclude_types = _parse_types(args.exclude_types)
    regions = load_regions(args.regions) if args.regions else None

    result = extract_assets(
        image_path,
        output_dir=output_dir,
        bg_threshold=args.bg_threshold,
        min_area=args.min_area,
        dilation=args.dilation,
        padding=args.padding,
        max_coverage=args.max_coverage,
        visualization=not args.no_visualization,
        types=types,
        exclude_types=exclude_types,
        regions=regions,
    )

    return {
        "source": result.source_image,
        "num_assets": result.num_assets,
        "output_dir": output_dir,
        "manifest": result.manifest_path,
        "visualization": result.visualization_path,
        "assets": [
            {"id": a.id, "label": a.label, "file": a.file_path}
            for a in result.assets
        ],
    }


def main(argv=None):
    """CLI entry point."""
    args = _parse_args(argv)

    if args.types and args.exclude_types:
        print("Error: cannot use both --types and --exclude-types", file=sys.stderr)
        sys.exit(1)

    log = (lambda msg: None) if args.quiet else (lambda msg: print(msg, file=sys.stderr))

    results = []
    images = args.images

    for img_path in images:
        p = Path(img_path)
        if not p.exists():
            log(f"Warning: file not found: {img_path}")
            continue

        t0 = time.time()
        log(f"Processing: {p.name}")

        try:
            if args.segments_only:
                result = _process_segments_only(str(p), args)
            else:
                if args.batch and len(images) > 1:
                    out_dir = str(Path(args.output) / p.stem)
                else:
                    out_dir = args.output

                result = _process_image(str(p), out_dir, args)

            elapsed = round(time.time() - t0, 2)
            result["elapsed_s"] = elapsed

            if args.segments_only:
                log(f"  Found {result['num_segments']} segments ({elapsed}s)")
            else:
                log(f"  Extracted {result['num_assets']} assets -> {result.get('output_dir', args.output)} ({elapsed}s)")

            results.append(result)

        except Exception as e:
            error_result = {
                "source": p.name,
                "error": str(e),
            }
            log(f"  Error: {e}")
            results.append(error_result)

    # Output
    if args.json or args.segments_only:
        output = results[0] if len(results) == 1 else results
        print(json.dumps(output, indent=2))
    elif not args.quiet and results:
        total_assets = sum(r.get("num_assets", 0) for r in results)
        if len(results) == 1:
            r = results[0]
            if "error" not in r:
                print(f"Extracted {r.get('num_assets', 0)} assets -> {r.get('output_dir', args.output)}")
        else:
            print(f"Processed {len(results)} images, extracted {total_assets} total assets")


if __name__ == "__main__":
    main()
