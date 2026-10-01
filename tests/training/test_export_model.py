"""Tests for the model export pipeline (T07).

All tests use mocks for ultralytics.YOLO and onnx — no real model exports
or GPU access required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolved_config(tmp_path: Path) -> dict[str, Any]:
    """Return a config dict that mimics what ``load_config`` would produce."""
    source_dir = tmp_path / "source_data"
    source_dir.mkdir(exist_ok=True)

    return {
        "_config_dir": str(tmp_path),
        "project_root": tmp_path,
        "source_data": source_dir,
        "output_dataset": tmp_path / "datasets" / "waste-detect",
        "models_dir": tmp_path / "models",
        "runs_dir": tmp_path / "runs" / "detect",
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
            "project": str(tmp_path / "runs" / "detect"),
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


def _create_fake_export(tmp_path: Path, filename: str, content: bytes = b"fake") -> Path:
    """Simulate Ultralytics export output — create a fake exported file."""
    fake = tmp_path / filename
    fake.write_bytes(content)
    return fake


# ---------------------------------------------------------------------------
# export_onnx
# ---------------------------------------------------------------------------


class TestExportOnnx:
    """Tests for ``export_onnx`` — ONNX export via Ultralytics."""

    @patch("src.training.export_model.YOLO", create=True)
    def test_calls_ultralytics_with_correct_params(
        self, mock_yolo_cls: MagicMock, tmp_path: Path
    ) -> None:
        from src.training.export_model import export_onnx

        config = _resolved_config(tmp_path)
        model_path = tmp_path / "best.pt"
        model_path.write_bytes(b"weights")
        output_dir = tmp_path / "output"

        # Ultralytics export() returns the path to the exported file
        fake_exported = _create_fake_export(tmp_path, "best.onnx")
        mock_model = MagicMock()
        mock_model.export.return_value = str(fake_exported)
        mock_yolo_cls.return_value = mock_model

        export_onnx(model_path, output_dir, config)

        mock_yolo_cls.assert_called_once_with(str(model_path))
        mock_model.export.assert_called_once_with(
            format="onnx", imgsz=640, half=True, simplify=True
        )

    @patch("src.training.export_model.YOLO", create=True)
    def test_moves_file_to_correct_destination(
        self, mock_yolo_cls: MagicMock, tmp_path: Path
    ) -> None:
        from src.training.export_model import export_onnx

        config = _resolved_config(tmp_path)
        model_path = tmp_path / "best.pt"
        model_path.write_bytes(b"weights")
        output_dir = tmp_path / "output"

        fake_exported = _create_fake_export(tmp_path, "best.onnx", b"onnx-data")
        mock_model = MagicMock()
        mock_model.export.return_value = str(fake_exported)
        mock_yolo_cls.return_value = mock_model

        result = export_onnx(model_path, output_dir, config)

        expected_dest = output_dir / "yolov8n-waste-best.onnx"
        assert result == expected_dest
        assert expected_dest.is_file()
        assert expected_dest.read_bytes() == b"onnx-data"
        # Original should no longer exist (moved, not copied)
        assert not fake_exported.exists()

    @patch("src.training.export_model.YOLO", create=True)
    def test_overwrites_existing_destination(
        self, mock_yolo_cls: MagicMock, tmp_path: Path
    ) -> None:
        from src.training.export_model import export_onnx

        config = _resolved_config(tmp_path)
        model_path = tmp_path / "best.pt"
        model_path.write_bytes(b"weights")
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # Pre-existing file at destination
        dest = output_dir / "yolov8n-waste-best.onnx"
        dest.write_bytes(b"old-data")

        fake_exported = _create_fake_export(tmp_path, "best.onnx", b"new-data")
        mock_model = MagicMock()
        mock_model.export.return_value = str(fake_exported)
        mock_yolo_cls.return_value = mock_model

        result = export_onnx(model_path, output_dir, config)

        assert result.read_bytes() == b"new-data"

    @patch("src.training.export_model.YOLO", create=True)
    def test_uses_config_export_params(
        self, mock_yolo_cls: MagicMock, tmp_path: Path
    ) -> None:
        from src.training.export_model import export_onnx

        config = _resolved_config(tmp_path)
        config["export"]["onnx"] = {"imgsz": 416, "half": False, "simplify": False}

        model_path = tmp_path / "best.pt"
        model_path.write_bytes(b"weights")

        fake_exported = _create_fake_export(tmp_path, "best.onnx")
        mock_model = MagicMock()
        mock_model.export.return_value = str(fake_exported)
        mock_yolo_cls.return_value = mock_model

        export_onnx(model_path, tmp_path / "output", config)

        mock_model.export.assert_called_once_with(
            format="onnx", imgsz=416, half=False, simplify=False
        )


# ---------------------------------------------------------------------------
# export_tensorrt
# ---------------------------------------------------------------------------


class TestExportTensorRT:
    """Tests for ``export_tensorrt`` — TensorRT engine export."""

    @patch("src.training.export_model.YOLO", create=True)
    def test_calls_ultralytics_with_engine_format(
        self, mock_yolo_cls: MagicMock, tmp_path: Path
    ) -> None:
        from src.training.export_model import export_tensorrt

        config = _resolved_config(tmp_path)
        model_path = tmp_path / "best.pt"
        model_path.write_bytes(b"weights")

        fake_exported = _create_fake_export(tmp_path, "best.engine")
        mock_model = MagicMock()
        mock_model.export.return_value = str(fake_exported)
        mock_yolo_cls.return_value = mock_model

        export_tensorrt(model_path, tmp_path / "output", config)

        mock_yolo_cls.assert_called_once_with(str(model_path))
        mock_model.export.assert_called_once_with(
            format="engine", imgsz=640, half=True
        )

    @patch("src.training.export_model.YOLO", create=True)
    def test_moves_engine_to_correct_destination(
        self, mock_yolo_cls: MagicMock, tmp_path: Path
    ) -> None:
        from src.training.export_model import export_tensorrt

        config = _resolved_config(tmp_path)
        model_path = tmp_path / "best.pt"
        model_path.write_bytes(b"weights")
        output_dir = tmp_path / "output"

        fake_exported = _create_fake_export(tmp_path, "best.engine", b"engine-data")
        mock_model = MagicMock()
        mock_model.export.return_value = str(fake_exported)
        mock_yolo_cls.return_value = mock_model

        result = export_tensorrt(model_path, output_dir, config)

        expected_dest = output_dir / "yolov8n-waste-best.engine"
        assert result == expected_dest
        assert expected_dest.is_file()
        assert expected_dest.read_bytes() == b"engine-data"
        assert not fake_exported.exists()

    @patch("src.training.export_model.YOLO", create=True)
    def test_prints_jetson_warning(
        self, mock_yolo_cls: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        from src.training.export_model import export_tensorrt

        config = _resolved_config(tmp_path)
        model_path = tmp_path / "best.pt"
        model_path.write_bytes(b"weights")

        fake_exported = _create_fake_export(tmp_path, "best.engine")
        mock_model = MagicMock()
        mock_model.export.return_value = str(fake_exported)
        mock_yolo_cls.return_value = mock_model

        export_tensorrt(model_path, tmp_path / "output", config)

        captured = capsys.readouterr()
        assert "Jetson Nano" in captured.out


# ---------------------------------------------------------------------------
# validate_export
# ---------------------------------------------------------------------------


class TestValidateExport:
    """Tests for ``validate_export`` — checks exported file integrity."""

    @patch("src.training.export_model.onnx", create=True)
    def test_valid_onnx_file(self, mock_onnx: MagicMock, tmp_path: Path) -> None:
        from src.training.export_model import validate_export

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake-onnx-data")

        result = validate_export(onnx_file, "onnx")

        assert result is True
        mock_onnx.load.assert_called_once_with(str(onnx_file))

    def test_missing_file_returns_false(self, tmp_path: Path) -> None:
        from src.training.export_model import validate_export

        nonexistent = tmp_path / "does_not_exist.onnx"
        result = validate_export(nonexistent, "onnx")

        assert result is False

    def test_empty_file_returns_false(self, tmp_path: Path) -> None:
        from src.training.export_model import validate_export

        empty = tmp_path / "empty.onnx"
        empty.write_bytes(b"")
        result = validate_export(empty, "onnx")

        assert result is False

    @patch("src.training.export_model.onnx", create=True)
    def test_onnx_load_failure_returns_false(
        self, mock_onnx: MagicMock, tmp_path: Path
    ) -> None:
        from src.training.export_model import validate_export

        onnx_file = tmp_path / "corrupt.onnx"
        onnx_file.write_bytes(b"not-real-onnx")
        mock_onnx.load.side_effect = Exception("Parse error")

        result = validate_export(onnx_file, "onnx")

        assert result is False

    def test_engine_file_valid(self, tmp_path: Path) -> None:
        from src.training.export_model import validate_export

        engine_file = tmp_path / "model.engine"
        engine_file.write_bytes(b"fake-engine-data")

        # Engine format skips onnx.load — just checks existence + size
        result = validate_export(engine_file, "engine")

        assert result is True


# ---------------------------------------------------------------------------
# main — CLI entry point
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for ``main`` — CLI orchestration."""

    @patch("src.training.export_model.validate_export", return_value=True)
    @patch("src.training.export_model.export_onnx")
    @patch("src.training.export_model.load_config")
    def test_main_onnx_format(
        self,
        mock_load_config: MagicMock,
        mock_export_onnx: MagicMock,
        mock_validate: MagicMock,
        tmp_path: Path,
    ) -> None:
        from src.training.export_model import main

        config = _resolved_config(tmp_path)
        mock_load_config.return_value = config
        mock_export_onnx.return_value = tmp_path / "models" / "yolov8n-waste-best.onnx"

        main(["--config", "configs/yolov8n-waste.yaml", "--format", "onnx"])

        mock_export_onnx.assert_called_once()
        mock_validate.assert_called_once()
        # validate called with onnx format
        assert mock_validate.call_args[0][1] == "onnx"

    @patch("src.training.export_model.validate_export", return_value=True)
    @patch("src.training.export_model.export_tensorrt")
    @patch("src.training.export_model.export_onnx")
    @patch("src.training.export_model.load_config")
    def test_main_all_formats(
        self,
        mock_load_config: MagicMock,
        mock_export_onnx: MagicMock,
        mock_export_trt: MagicMock,
        mock_validate: MagicMock,
        tmp_path: Path,
    ) -> None:
        from src.training.export_model import main

        config = _resolved_config(tmp_path)
        mock_load_config.return_value = config
        mock_export_onnx.return_value = tmp_path / "model.onnx"
        mock_export_trt.return_value = tmp_path / "model.engine"

        main(["--config", "configs/yolov8n-waste.yaml", "--format", "all"])

        mock_export_onnx.assert_called_once()
        mock_export_trt.assert_called_once()
        assert mock_validate.call_count == 2

    @patch("src.training.export_model.validate_export", return_value=True)
    @patch("src.training.export_model.export_tensorrt")
    @patch("src.training.export_model.load_config")
    def test_main_engine_format(
        self,
        mock_load_config: MagicMock,
        mock_export_trt: MagicMock,
        mock_validate: MagicMock,
        tmp_path: Path,
    ) -> None:
        from src.training.export_model import main

        config = _resolved_config(tmp_path)
        mock_load_config.return_value = config
        mock_export_trt.return_value = tmp_path / "model.engine"

        main(["--config", "configs/yolov8n-waste.yaml", "--format", "engine"])

        mock_export_trt.assert_called_once()
        mock_validate.assert_called_once()
        assert mock_validate.call_args[0][1] == "engine"

    @patch("src.training.export_model.load_config")
    def test_main_requires_config(
        self, mock_load_config: MagicMock
    ) -> None:
        from src.training.export_model import main

        with pytest.raises(SystemExit):
            main(["--format", "onnx"])

    @patch("src.training.export_model.load_config")
    def test_main_requires_format(
        self, mock_load_config: MagicMock
    ) -> None:
        from src.training.export_model import main

        with pytest.raises(SystemExit):
            main(["--config", "configs/yolov8n-waste.yaml"])
