"""Training pipeline for YOLOv8 waste detection with MLflow integration.

Orchestrates model training using Ultralytics YOLO, logs hyperparameters and
artifacts to MLflow, and copies the best checkpoint to the models directory.

Usage::

    python src/training/train_detect.py --config configs/yolov8n-waste.yaml
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import mlflow
from ultralytics import YOLO

from src.training.config import load_config


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------


def setup_mlflow(config: dict[str, Any]) -> None:
    """Configure MLflow tracking URI and experiment.

    Parameters
    ----------
    config:
        Fully resolved configuration dictionary (must contain ``mlflow`` key).
    """
    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_hyperparams(config: dict[str, Any]) -> None:
    """Log all training hyperparameters to the active MLflow run.

    Logs training parameters (epochs, batch, imgsz, optimizer, lr0, etc.)
    and augmentation parameters (prefixed with ``aug_``).

    Parameters
    ----------
    config:
        Fully resolved configuration dictionary.
    """
    training = config["training"]
    params: dict[str, Any] = {
        "model": training["model"],
        "epochs": training["epochs"],
        "batch": training["batch"],
        "imgsz": training["imgsz"],
        "optimizer": training["optimizer"],
        "lr0": training["lr0"],
        "cos_lr": training["cos_lr"],
        "patience": training["patience"],
        "seed": training["seed"],
        "workers": training["workers"],
        "cache": training["cache"],
        "task": training["task"],
        "name": training["name"],
    }

    # Log augmentation params with prefix
    for key, value in config.get("augmentation", {}).items():
        params[f"aug_{key}"] = value

    mlflow.log_params(params)


# ---------------------------------------------------------------------------
# Training argument builder
# ---------------------------------------------------------------------------


def build_train_args(config: dict[str, Any]) -> dict[str, Any]:
    """Merge training + augmentation config into Ultralytics ``model.train()`` kwargs.

    Resolves ``data.yaml`` path and ``project``/``name`` paths from config.
    Returns a dict ready to pass as ``**kwargs`` to ``model.train()``.

    Parameters
    ----------
    config:
        Fully resolved configuration dictionary.
    """
    training = config["training"]
    project_root = config["project_root"]

    # Resolve data.yaml path
    output_dataset = config["output_dataset"]
    data_yaml = Path(output_dataset) / "data.yaml"

    # Resolve project directory for training runs
    runs_dir = config["runs_dir"]

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "task": training["task"],
        "epochs": training["epochs"],
        "batch": training["batch"],
        "imgsz": training["imgsz"],
        "optimizer": training["optimizer"],
        "lr0": training["lr0"],
        "cos_lr": training["cos_lr"],
        "patience": training["patience"],
        "workers": training["workers"],
        "cache": training["cache"],
        "seed": training["seed"],
        "project": str(runs_dir),
        "name": training["name"],
        "exist_ok": training["exist_ok"],
    }

    # Add augmentation params
    for key, value in config.get("augmentation", {}).items():
        train_kwargs[key] = value

    return train_kwargs


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


def copy_best_model(run_dir: Path, models_dir: Path, run_name: str) -> Path:
    """Copy ``best.pt`` from the training run to the models directory.

    Parameters
    ----------
    run_dir:
        The training project directory (e.g. ``runs/detect``).
    models_dir:
        Destination directory for the final model.
    run_name:
        Name of the training run (e.g. ``yolov8n-waste``).

    Returns
    -------
    Path
        Destination path of the copied model.

    Raises
    ------
    FileNotFoundError
        If ``best.pt`` does not exist at the expected location.
    """
    source = Path(run_dir) / run_name / "weights" / "best.pt"
    if not source.is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {source}")

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    dest = models_dir / "yolov8n-waste-best.pt"
    shutil.copy2(source, dest)
    return dest


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(config: dict[str, Any]) -> Path:
    """Run the full training pipeline.

    1. Setup MLflow tracking.
    2. Load pretrained YOLOv8n model.
    3. Build training arguments from config.
    4. Start MLflow run, log hyperparameters, train model.
    5. Copy best checkpoint to models directory.
    6. Log best model as MLflow artifact.

    Parameters
    ----------
    config:
        Fully resolved configuration dictionary.

    Returns
    -------
    Path
        Path to the best model checkpoint in the models directory.
    """
    # Setup MLflow
    setup_mlflow(config)

    # Load pretrained model
    project_root = Path(config["project_root"])
    model_path = project_root / "pretrained" / config["training"]["model"]
    model = YOLO(str(model_path))

    # Build training kwargs
    train_kwargs = build_train_args(config)

    # Train within an MLflow run
    run_name = config["training"]["name"]
    with mlflow.start_run(run_name=run_name):
        log_hyperparams(config)
        model.train(**train_kwargs)

        # Determine best.pt location
        run_dir = Path(train_kwargs["project"])
        best_pt = run_dir / train_kwargs["name"] / "weights" / "best.pt"

        # Copy best model to models directory
        models_dir = Path(config["models_dir"])
        dest = copy_best_model(run_dir, models_dir, run_name)

        # Log artifact
        mlflow.log_artifact(str(best_pt))

    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n waste detection model with MLflow tracking.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    best_model = train(config)
    print(f"Training complete. Best model: {best_model}")


if __name__ == "__main__":
    main()
