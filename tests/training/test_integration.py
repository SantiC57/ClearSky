"""Integration smoke tests for the YOLOv8 waste detection training pipeline.

These tests exercise the full pipeline end-to-end:
  - convert_dataset + validate_dataset (no GPU needed)
  - train_detect (GPU required, marked slow)
  - export_model (GPU required, marked slow)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.training.convert_dataset import convert_dataset
from src.training.validate_dataset import validate_dataset

_CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
_NUM_CLASSES = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_synthetic_image(path: Path, class_id: int, size: int = 128) -> None:
    """Create a synthetic image with a distinct coloured rectangle.

    Each class gets a different colour so the contour annotator can find a
    clear object.  The rectangle occupies the centre ~60 % of the image.
    """
    # All colours are dark enough to be detected by the contour annotator
    # after grayscale conversion + adaptive thresholding.
    colours = [
        (180, 40, 40),    # cardboard — dark blue
        (40, 40, 180),    # glass — dark red
        (40, 180, 40),    # metal — dark green
        (150, 150, 40),   # paper — dark cyan
        (40, 180, 180),   # plastic — dark yellow
        (60, 60, 60),     # trash — dark gray
    ]
    bg = np.ones((size, size, 3), dtype=np.uint8) * 255
    colour = colours[class_id % len(colours)]
    margin = int(size * 0.2)
    cv2.rectangle(bg, (margin, margin), (size - margin, size - margin), colour, -1)
    cv2.imwrite(str(path), bg)


def _populate_source(source_dir: Path, images_per_split: int = 5) -> None:
    """Create train/valid/test splits with CSV + synthetic images."""
    for split in ("train", "valid", "test"):
        split_dir = source_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        csv_lines = ["filename, cardboard, glass, metal, paper, plastic, trash"]
        for i in range(images_per_split):
            class_id = i % _NUM_CLASSES
            fname = f"{split}_{i:03d}.jpg"
            one_hot = ["0"] * _NUM_CLASSES
            one_hot[class_id] = "1"
            csv_lines.append(f"{fname}, {', '.join(one_hot)}")
            _create_synthetic_image(split_dir / fname, class_id)

        (split_dir / "_classes.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")


def _build_convert_config(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Build a config dict for convert_dataset using the contour annotator."""
    return {
        "source_data": str(source_dir),
        "output_dataset": str(output_dir),
        "auto_annotate": {"method": "contour", "min_area_ratio": 0.005},
        "classes": list(_CLASS_NAMES),
    }


def _build_validation_config(output_dir: Path) -> dict[str, Any]:
    """Build a config dict for validate_dataset."""
    return {
        "output_dataset": Path(output_dir),
        "classes": list(_CLASS_NAMES),
    }


# ---------------------------------------------------------------------------
# test_e2e_convert_validate — no GPU, no model download
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_e2e_convert_validate(tmp_path: Path) -> None:
    """End-to-end: contour annotator → convert → validate → PASS.

    Creates a synthetic classification dataset, converts it to YOLO format
    using the ContourAnnotator (OpenCV only, no model download), then
    validates the output with validate_dataset.
    """
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "dataset"

    _populate_source(source_dir, images_per_split=5)

    # Convert
    convert_config = _build_convert_config(source_dir, output_dir)
    report = convert_dataset(convert_config)

    assert report.total_images == 15  # 5 per split × 3 splits
    assert report.successful == 15
    assert len(report.rejected) == 0

    # Validate
    val_config = _build_validation_config(output_dir)
    val_report = validate_dataset(val_config)

    assert val_report.is_valid, (
        f"Validation failed: {val_report.errors}"
    )
    assert val_report.total_images == 15
    assert val_report.valid_labels == 15
    assert len(val_report.missing_labels) == 0
    assert len(val_report.orphan_labels) == 0


# ---------------------------------------------------------------------------
# test_e2e_training_smoke — GPU required
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_e2e_training_smoke(tmp_path: Path) -> None:
    """End-to-end: convert → train 1 epoch → best.pt exists.

    Creates a tiny synthetic dataset, trains a YOLOv8n model for 1 epoch
    with minimal image size, and verifies the best checkpoint is produced.
    """
    from ultralytics import YOLO

    source_dir = tmp_path / "source"
    dataset_dir = tmp_path / "dataset"
    runs_dir = tmp_path / "runs"
    models_dir = tmp_path / "models"

    _populate_source(source_dir, images_per_split=5)

    # Convert
    convert_config = _build_convert_config(source_dir, dataset_dir)
    convert_dataset(convert_config)

    # Train — use ultralytics directly to avoid MLflow dependency in tests
    project_root = Path(__file__).resolve().parents[2]
    pretrained = project_root / "pretrained" / "yolov8n.pt"
    model = YOLO(str(pretrained))

    data_yaml = dataset_dir / "data.yaml"
    assert data_yaml.is_file(), f"data.yaml not found at {data_yaml}"

    results = model.train(
        data=str(data_yaml),
        task="detect",
        epochs=1,
        batch=2,
        imgsz=64,
        patience=1,
        workers=0,
        cache="ram",
        seed=42,
        project=str(runs_dir),
        name="smoke-test",
        exist_ok=True,
        verbose=False,
    )

    best_pt = runs_dir / "smoke-test" / "weights" / "best.pt"
    assert best_pt.is_file(), f"best.pt not found at {best_pt}"

    # Copy to models dir (mimics copy_best_model)
    models_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    dest = models_dir / "best.pt"
    shutil.copy2(best_pt, dest)
    assert dest.is_file()


# ---------------------------------------------------------------------------
# test_e2e_export_smoke — GPU required
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_e2e_export_smoke(tmp_path: Path) -> None:
    """End-to-end: train tiny model → export ONNX → file valid and < 20 MB.

    Depends on a trained checkpoint.  Creates a minimal dataset, trains for
    1 epoch, then exports to ONNX and validates the output.
    """
    from ultralytics import YOLO

    from src.training.export_model import export_onnx, validate_export

    source_dir = tmp_path / "source"
    dataset_dir = tmp_path / "dataset"
    runs_dir = tmp_path / "runs"
    models_dir = tmp_path / "models"
    export_dir = tmp_path / "export"

    _populate_source(source_dir, images_per_split=5)

    # Convert
    convert_config = _build_convert_config(source_dir, dataset_dir)
    convert_dataset(convert_config)

    # Train tiny model
    project_root = Path(__file__).resolve().parents[2]
    pretrained = project_root / "pretrained" / "yolov8n.pt"
    model = YOLO(str(pretrained))

    data_yaml = dataset_dir / "data.yaml"
    model.train(
        data=str(data_yaml),
        task="detect",
        epochs=1,
        batch=2,
        imgsz=64,
        patience=1,
        workers=0,
        cache="ram",
        seed=42,
        project=str(runs_dir),
        name="export-smoke",
        exist_ok=True,
        verbose=False,
    )

    best_pt = runs_dir / "export-smoke" / "weights" / "best.pt"
    assert best_pt.is_file(), f"best.pt not found at {best_pt}"

    # Export to ONNX
    export_config: dict[str, Any] = {
        "export": {
            "onnx": {"imgsz": 64, "half": False, "simplify": True},
        },
    }
    onnx_path = export_onnx(best_pt, export_dir, export_config)

    assert onnx_path.is_file(), f"ONNX file not found at {onnx_path}"
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    assert size_mb < 20, f"ONNX model too large: {size_mb:.1f} MB"

    # Validate the exported file
    assert validate_export(onnx_path, "onnx")
