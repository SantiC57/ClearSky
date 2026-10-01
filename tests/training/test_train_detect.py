"""Tests for the training pipeline (T06).

All tests use mocks for ultralytics.YOLO and mlflow — no real training or
model downloads occur.
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


# ---------------------------------------------------------------------------
# build_train_args — pure function, no mocks needed
# ---------------------------------------------------------------------------


class TestBuildTrainArgs:
    """Tests for ``build_train_args`` — merges config into train kwargs."""

    def test_merges_training_keys(self, tmp_path: Path) -> None:
        from src.training.train_detect import build_train_args

        config = _resolved_config(tmp_path)
        kwargs = build_train_args(config)

        assert kwargs["epochs"] == 100
        assert kwargs["batch"] == 16
        assert kwargs["imgsz"] == 640
        assert kwargs["optimizer"] == "AdamW"
        assert kwargs["lr0"] == 0.001
        assert kwargs["cos_lr"] is True
        assert kwargs["patience"] == 20
        assert kwargs["seed"] == 42
        assert kwargs["task"] == "detect"
        assert kwargs["name"] == "yolov8n-waste"
        assert kwargs["exist_ok"] is True

    def test_resolves_data_yaml_path(self, tmp_path: Path) -> None:
        from src.training.train_detect import build_train_args

        config = _resolved_config(tmp_path)
        kwargs = build_train_args(config)

        expected_data = str(tmp_path / "datasets" / "waste-detect" / "data.yaml")
        assert kwargs["data"] == expected_data

    def test_resolves_project_path(self, tmp_path: Path) -> None:
        from src.training.train_detect import build_train_args

        config = _resolved_config(tmp_path)
        kwargs = build_train_args(config)

        expected_project = str(tmp_path / "runs" / "detect")
        assert kwargs["project"] == expected_project

    def test_includes_augmentation_params(self, tmp_path: Path) -> None:
        from src.training.train_detect import build_train_args

        config = _resolved_config(tmp_path)
        kwargs = build_train_args(config)

        assert kwargs["hsv_h"] == 0.015
        assert kwargs["hsv_s"] == 0.7
        assert kwargs["hsv_v"] == 0.4
        assert kwargs["degrees"] == 10.0
        assert kwargs["translate"] == 0.1
        assert kwargs["scale"] == 0.5
        assert kwargs["flipud"] == 0.5
        assert kwargs["fliplr"] == 0.5
        assert kwargs["mosaic"] == 1.0
        assert kwargs["mixup"] == 0.0

    def test_empty_augmentation(self, tmp_path: Path) -> None:
        from src.training.train_detect import build_train_args

        config = _resolved_config(tmp_path)
        config["augmentation"] = {}
        kwargs = build_train_args(config)

        assert "hsv_h" not in kwargs
        assert "mosaic" not in kwargs


# ---------------------------------------------------------------------------
# setup_mlflow
# ---------------------------------------------------------------------------


class TestSetupMlflow:
    """Tests for ``setup_mlflow`` — configures MLflow tracking."""

    @patch("src.training.train_detect.mlflow")
    def test_sets_tracking_uri_and_experiment(
        self, mock_mlflow: MagicMock, tmp_path: Path
    ) -> None:
        from src.training.train_detect import setup_mlflow

        config = _resolved_config(tmp_path)
        setup_mlflow(config)

        mock_mlflow.set_tracking_uri.assert_called_once_with(
            "sqlite:///mlflow.db"
        )
        mock_mlflow.set_experiment.assert_called_once_with(
            "yolov8-waste-detection"
        )


# ---------------------------------------------------------------------------
# log_hyperparams
# ---------------------------------------------------------------------------


class TestLogHyperparams:
    """Tests for ``log_hyperparams`` — logs params to MLflow."""

    @patch("src.training.train_detect.mlflow")
    def test_logs_training_params(
        self, mock_mlflow: MagicMock, tmp_path: Path
    ) -> None:
        from src.training.train_detect import log_hyperparams

        config = _resolved_config(tmp_path)
        log_hyperparams(config)

        mock_mlflow.log_params.assert_called_once()
        params = mock_mlflow.log_params.call_args[0][0]

        assert params["model"] == "yolov8n.pt"
        assert params["epochs"] == 100
        assert params["batch"] == 16
        assert params["imgsz"] == 640
        assert params["optimizer"] == "AdamW"
        assert params["lr0"] == 0.001
        assert params["cos_lr"] is True
        assert params["patience"] == 20
        assert params["seed"] == 42

    @patch("src.training.train_detect.mlflow")
    def test_logs_augmentation_params_with_prefix(
        self, mock_mlflow: MagicMock, tmp_path: Path
    ) -> None:
        from src.training.train_detect import log_hyperparams

        config = _resolved_config(tmp_path)
        log_hyperparams(config)

        params = mock_mlflow.log_params.call_args[0][0]

        assert params["aug_hsv_h"] == 0.015
        assert params["aug_hsv_s"] == 0.7
        assert params["aug_flipud"] == 0.5
        assert params["aug_fliplr"] == 0.5
        assert params["aug_mosaic"] == 1.0
        assert params["aug_mixup"] == 0.0


# ---------------------------------------------------------------------------
# copy_best_model
# ---------------------------------------------------------------------------


class TestCopyBestModel:
    """Tests for ``copy_best_model`` — copies best.pt to models dir."""

    def test_copies_to_correct_destination(self, tmp_path: Path) -> None:
        from src.training.train_detect import copy_best_model

        run_dir = tmp_path / "runs" / "detect"
        run_name = "yolov8n-waste"
        weights_dir = run_dir / run_name / "weights"
        weights_dir.mkdir(parents=True)
        best_pt = weights_dir / "best.pt"
        best_pt.write_bytes(b"fake model weights")

        models_dir = tmp_path / "models"
        dest = copy_best_model(run_dir, models_dir, run_name)

        assert dest == models_dir / "yolov8n-waste-best.pt"
        assert dest.is_file()
        assert dest.read_bytes() == b"fake model weights"

    def test_creates_models_dir_if_missing(self, tmp_path: Path) -> None:
        from src.training.train_detect import copy_best_model

        run_dir = tmp_path / "runs" / "detect"
        run_name = "yolov8n-waste"
        weights_dir = run_dir / run_name / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_bytes(b"data")

        models_dir = tmp_path / "new_models"
        assert not models_dir.exists()

        dest = copy_best_model(run_dir, models_dir, run_name)
        assert dest.is_file()

    def test_raises_when_source_missing(self, tmp_path: Path) -> None:
        from src.training.train_detect import copy_best_model

        run_dir = tmp_path / "runs" / "detect"
        models_dir = tmp_path / "models"

        with pytest.raises(
            FileNotFoundError, match="Training checkpoint not found"
        ):
            copy_best_model(run_dir, models_dir, "yolov8n-waste")


# ---------------------------------------------------------------------------
# train — full orchestration with mocks
# ---------------------------------------------------------------------------


def _setup_train_mocks(tmp_path: Path):
    """Create the directory structure expected by ``train()``."""
    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    (pretrained_dir / "yolov8n.pt").write_bytes(b"pretrained")

    run_dir = tmp_path / "runs" / "detect"
    weights_dir = run_dir / "yolov8n-waste" / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.pt").write_bytes(b"trained weights")


class TestTrain:
    """Tests for ``train`` — orchestrates the full training flow."""

    @patch("src.training.train_detect.mlflow")
    @patch("src.training.train_detect.YOLO", create=True)
    def test_orchestration(
        self,
        mock_yolo_cls: MagicMock,
        mock_mlflow: MagicMock,
        tmp_path: Path,
    ) -> None:
        from src.training.train_detect import train

        config = _resolved_config(tmp_path)
        _setup_train_mocks(tmp_path)

        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        # Mock MLflow context manager
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(
            return_value=False
        )

        result = train(config)

        # YOLO loaded with correct model path
        pretrained_dir = tmp_path / "pretrained"
        mock_yolo_cls.assert_called_once_with(
            str(pretrained_dir / "yolov8n.pt")
        )

        # model.train() called with correct args
        mock_model.train.assert_called_once()
        train_kwargs = mock_model.train.call_args[1]
        assert train_kwargs["epochs"] == 100
        assert train_kwargs["batch"] == 16
        assert train_kwargs["data"] == str(
            tmp_path / "datasets" / "waste-detect" / "data.yaml"
        )

        # Best model copied
        expected_dest = tmp_path / "models" / "yolov8n-waste-best.pt"
        assert result == expected_dest
        assert result.is_file()

        # MLflow calls
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once()
        mock_mlflow.start_run.assert_called_once_with(
            run_name="yolov8n-waste"
        )
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_artifact.assert_called_once()

    @patch("src.training.train_detect.mlflow")
    @patch("src.training.train_detect.YOLO", create=True)
    def test_passes_augmentation_to_train(
        self,
        mock_yolo_cls: MagicMock,
        mock_mlflow: MagicMock,
        tmp_path: Path,
    ) -> None:
        from src.training.train_detect import train

        config = _resolved_config(tmp_path)
        _setup_train_mocks(tmp_path)

        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(
            return_value=False
        )

        train(config)

        train_kwargs = mock_model.train.call_args[1]
        assert train_kwargs["hsv_h"] == 0.015
        assert train_kwargs["flipud"] == 0.5
        assert train_kwargs["mosaic"] == 1.0
        assert train_kwargs["mixup"] == 0.0


# ---------------------------------------------------------------------------
# main — CLI entry point
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for ``main`` — CLI entry point."""

    @patch("src.training.train_detect.train")
    @patch("src.training.train_detect.load_config")
    def test_main_calls_load_config_and_train(
        self,
        mock_load_config: MagicMock,
        mock_train: MagicMock,
        tmp_path: Path,
    ) -> None:
        from src.training.train_detect import main

        fake_config = {"key": "value"}
        mock_load_config.return_value = fake_config
        mock_train.return_value = Path("/fake/models/best.pt")

        main(["--config", "configs/yolov8n-waste.yaml"])

        mock_load_config.assert_called_once_with("configs/yolov8n-waste.yaml")
        mock_train.assert_called_once_with(fake_config)

    @patch("src.training.train_detect.train")
    @patch("src.training.train_detect.load_config")
    def test_main_requires_config_arg(
        self,
        mock_load_config: MagicMock,
        mock_train: MagicMock,
    ) -> None:
        from src.training.train_detect import main

        with pytest.raises(SystemExit):
            main([])
