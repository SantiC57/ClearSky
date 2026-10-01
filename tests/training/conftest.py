"""Shared pytest fixtures for training tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


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
    """Return a dict with all required keys and valid values.

    Unlike ``tmp_config``, this returns an in-memory dict (not written to
    disk).  Useful for unit tests that call ``validate_config`` or
    ``resolve_paths`` directly.
    """
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
