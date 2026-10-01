"""Tests for src.training.config — YAML loading, path resolution, validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.training.config import (
    ConfigError,
    load_config,
    resolve_paths,
    validate_config,
)


# -----------------------------------------------------------------------
# load_config
# -----------------------------------------------------------------------


class TestLoadConfig:
    def test_load_config_valid_yaml(self, tmp_config: Path) -> None:
        """Loads a temp YAML and returns a dict with all expected top-level keys."""
        config = load_config(str(tmp_config))

        assert isinstance(config, dict)
        expected_keys = {
            "project_root",
            "source_data",
            "output_dataset",
            "models_dir",
            "runs_dir",
            "auto_annotate",
            "training",
            "augmentation",
            "export",
            "mlflow",
            "classes",
        }
        assert expected_keys.issubset(config.keys())
        assert len(config["classes"]) == 6

    def test_load_config_file_not_found(self) -> None:
        """Non-existent config path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


# -----------------------------------------------------------------------
# resolve_paths
# -----------------------------------------------------------------------


class TestResolvePaths:
    def test_resolve_paths_relative(self, minimal_config: dict[str, Any]) -> None:
        """Relative paths become absolute, resolved against project_root."""
        config = minimal_config
        resolve_paths(config)

        project_root: Path = config["project_root"]
        assert project_root.is_absolute()

        # output_dataset was "datasets/waste-detect" → should be under project_root
        output: Path = config["output_dataset"]
        assert output.is_absolute()
        assert str(output).startswith(str(project_root))

        # models_dir was "models" → should be under project_root
        models: Path = config["models_dir"]
        assert models.is_absolute()
        assert str(models).startswith(str(project_root))

    def test_resolve_paths_absolute(self, minimal_config: dict[str, Any], tmp_path: Path) -> None:
        """Absolute paths stay unchanged."""
        abs_source = tmp_path / "absolute_source"
        abs_source.mkdir()
        minimal_config["source_data"] = str(abs_source)

        resolve_paths(minimal_config)

        assert Path(minimal_config["source_data"]) == abs_source


# -----------------------------------------------------------------------
# interpolation
# -----------------------------------------------------------------------


class TestInterpolation:
    def test_interpolation(self, tmp_path: Path) -> None:
        """${var} interpolation in string values resolves correctly."""
        source_dir = tmp_path / "src"
        source_dir.mkdir()

        config_data = {
            "project_root": ".",
            "source_data": str(source_dir),
            "output_dataset": "datasets/out",
            "models_dir": "models",
            "runs_dir": "runs/detect",
            "auto_annotate": {
                "method": "contour",
                "model": "none",
                "confidence_threshold": 0.5,
                "batch_size": 4,
                "device": 0,
            },
            "training": {
                "model": "yolov8n.pt",
                "task": "detect",
                "epochs": 10,
                "batch": 8,
                "imgsz": 640,
                "seed": 42,
                "project": "${runs_dir}",
                "name": "test-run",
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
                "experiment_name": "test",
            },
            "classes": ["a", "b", "c", "d", "e", "f"],
        }

        config_path = tmp_path / "interp.yaml"
        with open(config_path, "w") as fh:
            yaml.dump(config_data, fh)

        config = load_config(str(config_path))

        # training.project was "${runs_dir}" which is "runs/detect"
        # After path resolution it should be project_root / "runs/detect"
        expected_runs = config["project_root"] / "runs" / "detect"
        assert config["training"]["project"] == str(expected_runs)


# -----------------------------------------------------------------------
# validate_config
# -----------------------------------------------------------------------


class TestValidateConfig:
    def test_validate_missing_keys(self) -> None:
        """Raises ConfigError listing all missing keys."""
        with pytest.raises(ConfigError) as exc_info:
            validate_config({})

        error_msg = str(exc_info.value)
        assert "project_root" in error_msg
        assert "source_data" in error_msg
        assert "classes" in error_msg

    def test_validate_invalid_batch(self, minimal_config: dict[str, Any]) -> None:
        """batch=0 raises ConfigError."""
        resolve_paths(minimal_config)
        minimal_config["training"]["batch"] = 0

        with pytest.raises(ConfigError) as exc_info:
            validate_config(minimal_config)

        assert "training.batch" in str(exc_info.value)

    def test_validate_invalid_confidence(self, minimal_config: dict[str, Any]) -> None:
        """confidence_threshold=1.5 raises ConfigError."""
        resolve_paths(minimal_config)
        minimal_config["auto_annotate"]["confidence_threshold"] = 1.5

        with pytest.raises(ConfigError) as exc_info:
            validate_config(minimal_config)

        assert "confidence_threshold" in str(exc_info.value)

    def test_validate_nonexistent_source(self, minimal_config: dict[str, Any]) -> None:
        """source_data pointing to a missing directory raises ConfigError."""
        minimal_config["source_data"] = "/nonexistent/path/that/does/not/exist"
        resolve_paths(minimal_config)

        with pytest.raises(ConfigError) as exc_info:
            validate_config(minimal_config)

        assert "source_data" in str(exc_info.value)

    def test_validate_accumulates_errors(self, minimal_config: dict[str, Any]) -> None:
        """Multiple failures are reported at once in a single ConfigError."""
        resolve_paths(minimal_config)
        # Introduce several errors simultaneously
        minimal_config["training"]["batch"] = 0
        minimal_config["auto_annotate"]["confidence_threshold"] = 2.0
        minimal_config["source_data"] = "/nonexistent/source"

        with pytest.raises(ConfigError) as exc_info:
            validate_config(minimal_config)

        error = exc_info.value
        assert len(error.errors) >= 3
        # All three problems should appear
        error_text = str(error)
        assert "training.batch" in error_text
        assert "confidence_threshold" in error_text
        assert "source_data" in error_text
