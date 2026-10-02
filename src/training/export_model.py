"""Model export pipeline for YOLOv8 waste detection.

Exports a trained YOLOv8 model to ONNX and/or TensorRT formats for edge
deployment on Jetson Nano.

Usage::

    python src/training/export_model.py --config configs/yolov8n-waste.yaml --format onnx
    python src/training/export_model.py --config configs/yolov8n-waste.yaml --format engine
    python src/training/export_model.py --config configs/yolov8n-waste.yaml --format all
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import onnx
from ultralytics import YOLO

from src.training.config import load_config

logger = logging.getLogger(__name__)

# Maximum expected size for the ONNX model (FP16).  Spec NFR-02 says < 20 MB.
_MAX_ONNX_SIZE_MB = 20


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def export_onnx(model_path: Path, output_dir: Path, config: dict[str, Any]) -> Path:
    """Export a trained YOLO model to ONNX format.

    Parameters
    ----------
    model_path:
        Path to the trained ``.pt`` checkpoint.
    output_dir:
        Directory where the exported model will be placed.
    config:
        Fully resolved configuration dictionary.  Export parameters are read
        from ``config["export"]["onnx"]``.

    Returns
    -------
    Path
        Path to the exported ``.onnx`` file.
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_cfg = config.get("export", {}).get("onnx", {})
    imgsz = export_cfg.get("imgsz", 640)
    half = export_cfg.get("half", True)
    simplify = export_cfg.get("simplify", True)

    print(f"Exporting ONNX model from {model_path} ...")
    print(f"  imgsz={imgsz}, half={half}, simplify={simplify}")

    model = YOLO(str(model_path))
    result = model.export(format="onnx", imgsz=imgsz, half=half, simplify=simplify)

    exported = Path(result)
    dest = output_dir / "yolov8n-waste-best.onnx"

    # Ultralytics may save to different locations depending on version.
    # Check common fallback paths if the reported path doesn't exist.
    if not exported.exists():
        fallbacks = [
            model_path.parent / (model_path.stem + ".onnx"),
            Path("weights") / "best.onnx",
            model_path.with_suffix(".onnx"),
        ]
        for fb in fallbacks:
            if fb.exists():
                exported = fb
                break

    # Overwrite if destination already exists
    if dest.exists():
        dest.unlink()
    shutil.move(str(exported), str(dest))

    # File-size warning (NFR-02: < 20 MB)
    size_mb = dest.stat().st_size / (1024 * 1024)
    if size_mb > _MAX_ONNX_SIZE_MB:
        logger.warning(
            "ONNX model size %.1f MB exceeds %d MB limit", size_mb, _MAX_ONNX_SIZE_MB
        )
        print(
            f"  WARNING: ONNX model size ({size_mb:.1f} MB) exceeds "
            f"{_MAX_ONNX_SIZE_MB} MB target."
        )

    print(f"  ONNX model saved to {dest} ({size_mb:.1f} MB)")
    return dest


# ---------------------------------------------------------------------------
# TensorRT export
# ---------------------------------------------------------------------------


def export_tensorrt(
    model_path: Path, output_dir: Path, config: dict[str, Any]
) -> Path:
    """Export a trained YOLO model to TensorRT engine format.

    .. warning::
        TensorRT engines are hardware-specific.  This export **must** run on
        the target Jetson Nano — the resulting ``.engine`` file will NOT work
        on a different GPU.

    Parameters
    ----------
    model_path:
        Path to the trained ``.pt`` checkpoint.
    output_dir:
        Directory where the exported engine will be placed.
    config:
        Fully resolved configuration dictionary.  Export parameters are read
        from ``config["export"]["tensorrt"]``.

    Returns
    -------
    Path
        Path to the exported ``.engine`` file.
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_cfg = config.get("export", {}).get("tensorrt", {})
    imgsz = export_cfg.get("imgsz", 640)
    half = export_cfg.get("half", True)

    print("WARNING: TensorRT export MUST run on the target Jetson Nano.")
    print("  The resulting .engine file is hardware-specific.")
    print(f"Exporting TensorRT engine from {model_path} ...")
    print(f"  imgsz={imgsz}, half={half}")

    model = YOLO(str(model_path))
    result = model.export(format="engine", imgsz=imgsz, half=half)

    exported = Path(result)
    dest = output_dir / "yolov8n-waste-best.engine"

    # Ultralytics may save to different locations depending on version.
    # Check common fallback paths if the reported path doesn't exist.
    if not exported.exists():
        fallbacks = [
            model_path.parent / (model_path.stem + ".engine"),
            Path("weights") / "best.engine",
            model_path.with_suffix(".engine"),
        ]
        for fb in fallbacks:
            if fb.exists():
                exported = fb
                break

    if dest.exists():
        dest.unlink()
    shutil.move(str(exported), str(dest))

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  TensorRT engine saved to {dest} ({size_mb:.1f} MB)")
    return dest


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_export(export_path: Path, format: str) -> bool:
    """Validate that an exported model file is usable.

    Checks:

    * File exists and has size > 0.
    * For ONNX: attempts to load the graph with ``onnx.load()`` to verify the
      protobuf is well-formed.

    Parameters
    ----------
    export_path:
        Path to the exported file.
    format:
        ``"onnx"`` or ``"engine"``.

    Returns
    -------
    bool
        ``True`` when the file passes all checks.
    """
    export_path = Path(export_path)

    if not export_path.is_file():
        print(f"  Validation FAILED: file not found — {export_path}")
        return False

    if export_path.stat().st_size == 0:
        print(f"  Validation FAILED: file is empty — {export_path}")
        return False

    if format == "onnx":
        try:
            onnx.load(str(export_path))
        except Exception as exc:
            print(f"  Validation FAILED: ONNX load error — {exc}")
            return False

    print(f"  Validation passed for {export_path}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for model export."""
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 waste detection model to ONNX / TensorRT.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=("onnx", "engine", "all"),
        help="Export format: onnx, engine (TensorRT), or all.",
    )
    parser.add_argument(
        "--model",
        default="models/yolov8n-waste-best.pt",
        help="Path to the trained model checkpoint (default: models/yolov8n-waste-best.pt).",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)

    project_root = Path(config["project_root"])
    model_path = project_root / args.model if not Path(args.model).is_absolute() else Path(args.model)
    output_dir = Path(config["models_dir"])

    export_format: str = args.format

    if export_format in ("onnx", "all"):
        onnx_path = export_onnx(model_path, output_dir, config)
        validate_export(onnx_path, "onnx")

    if export_format in ("engine", "all"):
        engine_path = export_tensorrt(model_path, output_dir, config)
        validate_export(engine_path, "engine")

    print("Export complete.")


if __name__ == "__main__":
    main()
