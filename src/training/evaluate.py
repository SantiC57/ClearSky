"""Evaluation pipeline for YOLOv8 waste detection.

Runs validation on the held-out test split, extracts per-class AP metrics,
generates a color-coded bar chart with a target line at AP=0.85, and logs
metrics to MLflow.

Usage::

    python src/training/evaluate.py --config configs/yolov8n-waste.yaml
    python src/training/evaluate.py --config configs/yolov8n-waste.yaml --model models/custom.pt
    python src/training/evaluate.py --config configs/yolov8n-waste.yaml --no-plot
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.training.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# Target AP@0.5 per the spec (NFR-02).
_TARGET_AP = 0.85


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model_path: Path, data_yaml: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Run YOLO validation on the test split and return parsed metrics.

    Parameters
    ----------
    model_path:
        Path to the trained ``.pt`` checkpoint.
    data_yaml:
        Path to the dataset ``data.yaml``.
    config:
        Fully resolved configuration dictionary.  Training parameters
        (``imgsz``, ``batch``, ``device``) are read from ``config["training"]``.

    Returns
    -------
    dict
        Metrics dictionary with keys ``mAP50``, ``mAP50_95``, ``precision``,
        ``recall``, and ``per_class`` (mapping class name -> per-class stats).

    Raises
    ------
    FileNotFoundError
        If *model_path* does not exist.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    training_cfg = config.get("training", {})
    imgsz = int(training_cfg.get("imgsz", 640))
    batch = int(training_cfg.get("batch", 16))
    device = training_cfg.get("device", 0)

    model = YOLO(str(model_path))
    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
    )

    return _parse_results(results, config)


def _parse_results(results: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Extract metrics from a YOLO validation results object."""
    class_names: dict[int, str] = dict(results.names)

    per_class: dict[str, dict[str, float]] = {}
    for class_id, name in class_names.items():
        per_class[name] = {
            "ap50": float(results.box.ap50[class_id]),
        }

    return {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------


def print_metrics_table(metrics: dict[str, Any], class_names: list[str]) -> None:
    """Print a formatted table of per-class metrics to stdout."""
    header = f"{'Class':<12} {'AP@0.5':<10} {'Precision':<10} {'Recall':<10} {'Support':<8}"
    print("")
    print(header)
    print("-" * len(header))

    for name in class_names:
        cls_stats = metrics.get("per_class", {}).get(name, {})
        ap = cls_stats.get("ap50", 0.0)
        precision = cls_stats.get("precision", 0.0)
        recall = cls_stats.get("recall", 0.0)
        support = cls_stats.get("support", "—")
        print(
            f"{name:<12} {ap:<10.4f} {precision:<10.4f} {recall:<10.4f} {support!s:<8}"
        )

    print("-" * len(header))
    print(
        f"{'mAP@0.5':<12} {metrics.get('mAP50', 0.0):<10.4f}"
    )
    print(
        f"{'mAP@0.5:0.95':<12} {metrics.get('mAP50_95', 0.0):<10.4f}"
    )
    print(
        f"{'Precision':<12} {metrics.get('precision', 0.0):<10.4f}"
    )
    print(
        f"{'Recall':<12} {metrics.get('recall', 0.0):<10.4f}"
    )


# ---------------------------------------------------------------------------
# Per-class AP chart
# ---------------------------------------------------------------------------


def _bar_color(score: float) -> str:
    """Return a color based on the AP score thresholds."""
    if score >= 0.85:
        return "#2ca02c"  # green
    if score >= 0.70:
        return "#ffbf00"  # yellow / amber
    return "#d62728"  # red


def plot_per_class_ap(
    metrics: dict[str, Any], class_names: list[str], output_path: Path
) -> Path:
    """Generate a color-coded horizontal bar chart of per-class AP@0.5.

    A vertical target line is drawn at AP=0.85 (per NFR-02).

    Parameters
    ----------
    metrics:
        Metrics dictionary as returned by :func:`evaluate_model`.
    class_names:
        Ordered list of class names (y-axis).
    output_path:
        Destination path for the PNG file.

    Returns
    -------
    Path
        The *output_path* where the chart was saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_class = metrics.get("per_class", {})
    classes = list(class_names)
    scores = [per_class.get(name, {}).get("ap50", 0.0) for name in classes]
    colors = [_bar_color(s) for s in scores]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(classes, scores, color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AP@0.5")
    ax.set_title("Per-Class AP@0.5 — YOLOv8n Waste Detection")

    # Target line at 0.85
    ax.axvline(x=_TARGET_AP, color="black", linestyle="--", linewidth=1.2,
               label=f"Target ({_TARGET_AP:.2f})")
    ax.legend(loc="lower right")

    # Value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def log_metrics_mlflow(metrics: dict[str, Any], config: dict[str, Any] | None = None) -> None:
    """Log evaluation metrics to the active MLflow run.

    Logs scalar metrics (mAP50, mAP50_95, precision, recall) and per-class
    AP@0.5 as individual metrics (``ap_<class_name>``).

    Parameters
    ----------
    metrics:
        Metrics dictionary as returned by :func:`evaluate_model`.
    config:
        Optional configuration dictionary.  When provided and MLflow is not
        already configured, ``setup_mlflow`` is called to initialize the
        tracking URI and experiment.  When omitted, metrics are logged to
        whatever MLflow run is currently active.
    """
    if config is not None:
        try:
            from src.training.train_detect import setup_mlflow
            setup_mlflow(config)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("MLflow setup failed; skipping logging: %s", exc)
            return

    mlflow.log_metric("mAP50", metrics.get("mAP50", 0.0))
    mlflow.log_metric("mAP50_95", metrics.get("mAP50_95", 0.0))
    mlflow.log_metric("precision", metrics.get("precision", 0.0))
    mlflow.log_metric("recall", metrics.get("recall", 0.0))

    for class_name, stats in metrics.get("per_class", {}).items():
        safe_name = class_name.replace(" ", "_")
        mlflow.log_metric(f"ap_{safe_name}", stats.get("ap50", 0.0))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLOv8 waste detection model.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--model",
        default="models/yolov8n-waste-best.pt",
        help="Path to the trained model checkpoint "
             "(default: models/yolov8n-waste-best.pt).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip per-class AP chart generation.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)

    project_root = Path(config["project_root"])
    model_path = (
        project_root / args.model
        if not Path(args.model).is_absolute()
        else Path(args.model)
    )

    data_yaml = Path(config["_config_dir"]) / "datasets" / "waste-detect" / "data.yaml"
    if not data_yaml.is_absolute():
        data_yaml = (project_root / data_yaml).resolve()

    class_names: list[str] = list(config.get("classes", []))

    print(f"Evaluating model: {model_path}")
    print(f"Dataset YAML:     {data_yaml}")

    metrics = evaluate_model(model_path, data_yaml, config)

    print_metrics_table(metrics, class_names)

    if not args.no_plot:
        output_path = Path(config["models_dir"]) / "per_class_ap.png"
        plot_per_class_ap(metrics, class_names, output_path)
        print(f"Per-class AP chart saved to {output_path}")

    # MLflow logging is best-effort; skip silently when no active run.
    try:
        log_metrics_mlflow(metrics, config)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("MLflow logging skipped: %s", exc)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
