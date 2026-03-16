"""Basic usage example for px-asset-extract.

Extract all assets from a single image.
"""

from px_asset_extract import extract_assets

# One-line extraction
result = extract_assets("test_data/slide.png", output_dir="output/basic/")

print(f"Source: {result.source_image} ({result.source_size[0]}x{result.source_size[1]})")
print(f"Background: RGB{result.background_color}")
print(f"Assets: {result.num_assets}")
print()

for asset in result.assets:
    print(f"  {asset.id}: {asset.label} ({asset.bbox.width}x{asset.bbox.height}) -> {asset.file_path}")

print()
print(f"Manifest: {result.manifest_path}")
print(f"Visualization: {result.visualization_path}")
