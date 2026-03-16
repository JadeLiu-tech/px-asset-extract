"""Batch processing example for px-asset-extract.

Process multiple images and collect results.
"""

from pathlib import Path

from px_asset_extract import extract_assets

# Process all images in a directory
input_dir = Path("test_data")
output_base = Path("output/batch")

total_assets = 0

for image_path in sorted(input_dir.glob("*.png")):
    output_dir = output_base / image_path.stem

    print(f"Processing: {image_path.name}")
    result = extract_assets(str(image_path), output_dir=str(output_dir))

    total_assets += result.num_assets
    print(f"  -> {result.num_assets} assets extracted to {output_dir}")

    # Print type summary
    type_counts: dict[str, int] = {}
    for asset in result.assets:
        type_counts[asset.label] = type_counts.get(asset.label, 0) + 1
    for label, count in sorted(type_counts.items()):
        print(f"     {label}: {count}")
    print()

print(f"Total: {total_assets} assets from {len(list(input_dir.glob('*.png')))} images")
