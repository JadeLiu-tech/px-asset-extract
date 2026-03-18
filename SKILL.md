---
name: px-asset-extract
description: >
  Extract individual assets from images (slides, posters, infographics, diagrams) as transparent
  PNGs with a JSON manifest. Zero ML models, pure classical CV (PIL+numpy). Automatically segments,
  classifies (text, illustration, icon, graphic, line, dot, diagram, shadow, element), and crops
  each element with anti-aliased alpha transparency. Supports type filtering (--types/--exclude-types)
  and pre-computed region extraction (--regions) for bridging with visual grounding models.
  Trigger on 'extract assets from image', 'decompose slide into elements', 'get all icons from
  poster', 'extract illustrations', 'segment and crop', 'pull out individual elements', or when
  the user has an image and wants individual transparent PNGs of each element.
---

# px-asset-extract: Image Asset Extraction

## What It Does

Decomposes images into individual transparent PNG assets with classification and a JSON manifest.
The full pipeline runs in 2-6 seconds on CPU with zero ML models:

1. **Background detection** — median color from image borders
2. **Foreground mask** — Euclidean color distance thresholding
3. **Character bridging** — dilation connects letters into words
4. **Connected components** — union-find with 8-connectivity
5. **Classification** — heuristic typing into 10 categories
6. **Text-line merging** — groups word fragments into text lines
7. **Alpha extraction** — anti-aliased transparent cropping
8. **Deduplication** — removes overlapping and oversized segments

## When to Use This

| Scenario | Use px-asset-extract? |
|----------|----------------------|
| Extract all elements from a slide/poster | Yes — this is the primary use case |
| Get only illustrations, skip text | Yes — use `--types illustration` or `--exclude-types text` |
| Extract specific objects by description | Use with `--regions` + a grounding model (e.g., Florence-2) |
| Remove background from a single photo | No — use a background removal model instead |
| Segment a photo scene | No — use SAM/FastSAM for photographic content |
| Image has textured/photographic background | Limited — works best on clean/solid backgrounds |

## Installation

```bash
git clone https://github.com/pxgenius/px-asset-extract.git
cd px-asset-extract
pip install .
```

## Usage

### CLI

```bash
# Basic extraction
px-extract <image> -o <output_dir>

# Only extract illustrations and icons
px-extract <image> -o <output_dir> --types illustration,icon

# Extract everything except text and dots
px-extract <image> -o <output_dir> --exclude-types text,dot,line

# Extract from pre-computed bounding boxes (e.g. from px-ground)
px-extract <image> -o <output_dir> --regions regions.json

# Segment only — output JSON, no PNGs
px-extract <image> --segments-only

# Batch processing
px-extract images/*.png -o output/ --batch

# JSON output to stdout
px-extract <image> -o <output_dir> --json --quiet
```

### Python API

```python
from px_asset_extract import extract_assets, load_regions

# Full extraction
result = extract_assets("slide.png", output_dir="assets/")
for asset in result.assets:
    print(f"{asset.id}: {asset.label} at ({asset.bbox.x}, {asset.bbox.y}) -> {asset.file_path}")

# Type filtering
result = extract_assets("slide.png", output_dir="icons/", types=["illustration", "icon"])
result = extract_assets("slide.png", output_dir="graphics/", exclude_types=["text", "line", "dot"])

# Pre-computed regions (from grounding model output)
regions = load_regions("grounded.json")
result = extract_assets("slide.png", output_dir="targeted/", regions=regions)

# Combine regions + type filter
result = extract_assets("slide.png", output_dir="charts/", regions=regions, types=["chart"])
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output` | `assets` | Output directory |
| `--bg-threshold` | `22.0` | Background color distance (lower = more sensitive) |
| `--min-area` | `60` | Minimum segment area in pixels |
| `--dilation` | `2` | Character gap bridging passes |
| `--padding` | `10` | Extra pixels around each asset |
| `--max-coverage` | `0.5` | Max fraction of image a segment can cover |
| `--types` | | Only extract these types (comma-separated) |
| `--exclude-types` | | Skip these types (comma-separated) |
| `--regions` | | JSON file with bounding boxes (skips segmentation) |
| `--segments-only` | | Output segment JSON without extracting PNGs |
| `--no-visualization` | | Skip visualization image |
| `--batch` | | Create subdirectories per image |
| `--json` | | Output results as JSON to stdout |
| `--quiet` | | Suppress progress messages |

## Output

Each run produces:
- `asset_NNN_<type>.png` — individual transparent PNGs
- `manifest.json` — positions, types, and metadata for all assets
- `visualization.png` — input image with color-coded bounding boxes

### Manifest format

```json
{
  "source_image": "slide.png",
  "source_size": {"width": 1920, "height": 1080},
  "background_color": [255, 255, 255],
  "num_assets": 44,
  "assets": [
    {
      "id": "asset_000_illustration",
      "label": "illustration",
      "file": "asset_000_illustration.png",
      "position": {"x": 100, "y": 50, "width": 400, "height": 300},
      "pixel_area": 120000
    }
  ]
}
```

### Regions JSON format (for --regions)

```json
[
  {"x": 100, "y": 50, "width": 400, "height": 300, "label": "chart"},
  {"x1": 600, "y1": 100, "x2": 800, "y2": 300, "label": "logo"}
]
```

Also supports `{"regions": [...]}` wrapper. Label defaults to `"region"` if omitted.

## Asset Types

| Type | Detection Logic |
|------|----------------|
| `text` | dark_ratio > 0.4, uniform ink color |
| `illustration` | Large (>1% image area), colorful |
| `icon` | Small (<3000px area, <60px max dimension) |
| `graphic` | Medium-sized, colored |
| `line` | Thin (min dimension <=5px, extreme aspect ratio) |
| `dot` | Very small (<150px area, <20px dimension) |
| `diagram` | Low fill ratio (<0.25) |
| `diagram_network` | Spans >80% of image, very low fill |
| `shadow` | Bright (>200), low contrast, low saturation |
| `element` | Catch-all for unclassified objects |

## Performance

| Image type | Assets | Time |
|-----------|--------|------|
| Presentation slide | 22-44 | 2-6s |
| Poster | 11 | 3.9s |
| Scientific diagram | 43 | 4.2s |
| Technical diagram | 42 | 4.5s |
| Data chart | 26 | 4.8s |

## Dependencies

Only `Pillow` and `numpy`. Optional `opencv-python` for better alpha edges.
