"""Shared pytest fixtures for training tests."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest
import yaml


# ---------------------------------------------------------------------------
# Minimal JPEG generator (no Pillow dependency)
# ---------------------------------------------------------------------------

# Smallest valid JPEG: 1x1 white pixel.
# This is a well-known minimal JFIF byte sequence.
_MINIMAL_JPEG_BYTES = bytes(
    [
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00,
        0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB,
        0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07,
        0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B,
        0x0B, 0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E,
        0x1D, 0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C,
        0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34,
        0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34,
        0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01,
        0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01,
        0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
        0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
        0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32,
        0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1,
        0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18,
        0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36,
        0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64,
        0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77,
        0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A,
        0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
        0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5,
        0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7,
        0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9,
        0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
        0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF,
        0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0x7B, 0x94,
        0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0xFF, 0xD9,
    ]
)


def _write_tiny_jpeg(path: Path) -> None:
    """Write a minimal valid JPEG file (1×1 white pixel)."""
    path.write_bytes(_MINIMAL_JPEG_BYTES)


# ---------------------------------------------------------------------------
# Fixtures from T01
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Path:
    """Create a minimal valid YAML config in a temp directory.

    The temp directory also acts as ``project_root``.  A ``source_data``
    sub-directory is created so that validation passes.
    """
    source_dir = tmp_path / "source_data"
    source_dir.mkdir()

    config: dict[str, Any] = {
        "project_root": ".",
        "source_data": str(source_dir),
        "output_dataset": "datasets/waste-detect",
        "models_dir": "models",
        "runs_dir": "runs/detect",
        "auto_annotate": {
            "method": "yolo-world",
            "model": "yolo-world-l",
            "confidence_threshold": 0.30,
            "batch_size": 8,
            "device": 0,
        },
        "training": {
            "model": "yolov8n.pt",
            "task": "detect",
            "epochs": 100,
            "batch": 16,
            "imgsz": 640,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "cos_lr": True,
            "patience": 20,
            "workers": 4,
            "cache": "ram",
            "seed": 42,
            "project": "${runs_dir}",
            "name": "yolov8n-waste",
            "exist_ok": True,
        },
        "augmentation": {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 10.0,
            "translate": 0.1,
            "scale": 0.5,
            "flipud": 0.5,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
        },
        "export": {
            "onnx": {"imgsz": 640, "half": True, "simplify": True},
            "tensorrt": {"imgsz": 640, "half": True},
        },
        "mlflow": {
            "tracking_uri": "sqlite:///mlflow.db",
            "experiment_name": "yolov8-waste-detection",
        },
        "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.dump(config, fh, default_flow_style=False)

    return config_path


@pytest.fixture()
def minimal_config(tmp_path: Path) -> dict[str, Any]:
    """Return a dict with all required keys and valid values."""
    source_dir = tmp_path / "source_data"
    source_dir.mkdir()

    return {
        "_config_dir": str(tmp_path),
        "project_root": ".",
        "source_data": str(source_dir),
        "output_dataset": "datasets/waste-detect",
        "models_dir": "models",
        "runs_dir": "runs/detect",
        "auto_annotate": {
            "method": "yolo-world",
            "model": "yolo-world-l",
            "confidence_threshold": 0.30,
            "batch_size": 8,
            "device": 0,
        },
        "training": {
            "model": "yolov8n.pt",
            "task": "detect",
            "epochs": 100,
            "batch": 16,
            "imgsz": 640,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "cos_lr": True,
            "patience": 20,
            "workers": 4,
            "cache": "ram",
            "seed": 42,
            "project": "${runs_dir}",
            "name": "yolov8n-waste",
            "exist_ok": True,
        },
        "augmentation": {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 10.0,
            "translate": 0.1,
            "scale": 0.5,
            "flipud": 0.5,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
        },
        "export": {
            "onnx": {"imgsz": 640, "half": True, "simplify": True},
            "tensorrt": {"imgsz": 640, "half": True},
        },
        "mlflow": {
            "tracking_uri": "sqlite:///mlflow.db",
            "experiment_name": "yolov8-waste-detection",
        },
        "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
    }


# ---------------------------------------------------------------------------
# T02 fixtures — sample split directory with CSV + images
# ---------------------------------------------------------------------------


_FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def fixtures_dir() -> Path:
    """Return the path to the static fixtures directory."""
    return _FIXTURES_DIR


@pytest.fixture()
def sample_classes_csv(fixtures_dir: Path) -> Path:
    """Return the path to the sample ``_classes.csv`` fixture."""
    return fixtures_dir / "sample_classes.csv"


@pytest.fixture()
def sample_split_dir(tmp_path: Path) -> Path:
    """Create a complete split directory with CSV and tiny JPEG images.

    Layout::

        tmp_path/train/
        ├── _classes.csv       (5 rows: 2 cardboard, 1 glass, 1 metal, 1 paper)
        ├── img001.jpg         (cardboard)
        ├── img002.jpg         (cardboard)
        ├── img003.jpg         (glass)
        ├── img004.jpg         (metal)
        └── img005.jpg         (paper)
    """
    split_dir = tmp_path / "train"
    split_dir.mkdir()

    csv_content = (
        "filename, cardboard, glass, metal, paper, plastic, trash\n"
        "img001.jpg, 1, 0, 0, 0, 0, 0\n"
        "img002.jpg, 1, 0, 0, 0, 0, 0\n"
        "img003.jpg, 0, 1, 0, 0, 0, 0\n"
        "img004.jpg, 0, 0, 1, 0, 0, 0\n"
        "img005.jpg, 0, 0, 0, 1, 0, 0\n"
    )
    (split_dir / "_classes.csv").write_text(csv_content, encoding="utf-8")

    for name in ("img001.jpg", "img002.jpg", "img003.jpg", "img004.jpg", "img005.jpg"):
        _write_tiny_jpeg(split_dir / name)

    return split_dir


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """Return a fresh output directory for conversion tests."""
    out = tmp_path / "output"
    out.mkdir()
    return out
