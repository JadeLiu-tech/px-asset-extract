"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Path to test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"


@pytest.fixture
def test_data_dir():
    """Return path to the test_data directory."""
    return TEST_DATA_DIR


@pytest.fixture
def slide_image(test_data_dir):
    """Path to the slide test image."""
    return str(test_data_dir / "slide.png")


@pytest.fixture
def poster_image(test_data_dir):
    """Path to the poster test image."""
    return str(test_data_dir / "poster.png")


@pytest.fixture
def diagram_image(test_data_dir):
    """Path to the diagram test image."""
    return str(test_data_dir / "diagram.png")


@pytest.fixture
def synthetic_image(tmp_path):
    """Create a synthetic test image with known objects on white background.

    Contains:
    - A red rectangle (illustration-like)
    - A small blue square (icon-like)
    - A thin black horizontal line
    - Dark text-like block
    """
    width, height = 400, 300
    img = np.ones((height, width, 4), dtype=np.uint8) * 255

    # Red rectangle (60x40) at position (50, 30)
    img[30:70, 50:110, 0] = 200
    img[30:70, 50:110, 1] = 30
    img[30:70, 50:110, 2] = 30

    # Small blue square (20x20) at position (200, 150)
    img[150:170, 200:220, 0] = 30
    img[150:170, 200:220, 1] = 30
    img[150:170, 200:220, 2] = 200

    # Thin black horizontal line at y=250, x=20..380
    img[248:252, 20:380, 0] = 10
    img[248:252, 20:380, 1] = 10
    img[248:252, 20:380, 2] = 10

    # Dark text-like block at (300, 50) — small, dark, uniform
    img[50:65, 300:360, 0] = 20
    img[50:65, 300:360, 1] = 20
    img[50:65, 300:360, 2] = 20

    path = str(tmp_path / "synthetic.png")
    Image.fromarray(img, "RGBA").save(path)
    return path


@pytest.fixture
def blank_image(tmp_path):
    """Create a blank white image (edge case)."""
    img = np.ones((100, 100, 4), dtype=np.uint8) * 255
    path = str(tmp_path / "blank.png")
    Image.fromarray(img, "RGBA").save(path)
    return path


@pytest.fixture
def single_color_image(tmp_path):
    """Create a solid red image (edge case)."""
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 1] = 30
    img[:, :, 2] = 30
    img[:, :, 3] = 255
    path = str(tmp_path / "solid_red.png")
    Image.fromarray(img, "RGBA").save(path)
    return path
