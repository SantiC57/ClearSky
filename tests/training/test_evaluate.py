"""Tests for the evaluation pipeline (T08).

All tests use mocks for ultralytics.YOLO, mlflow, and matplotlib — no real
model validation, GPU access, or file I/O beyond tmp_path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


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
        "classes": list(CLASS_NAMES),
    }


def _make_mock_results(
    ap50_values: list[float] | None = None,
    map50: float = 0.88,
    map50_95: float = 0.62,
    mp: float = 0.90,
    mr: float = 0.85,
) -> MagicMock:
    """Build a mock that mimics the YOLO validation results object.

    ``results.box.ap50`` is a list indexed by class_id.
    ``results.box.map50``, ``map``, ``mp``, ``mr`` are scalar floats.
    ``results.names`` is a dict mapping class_id -> class_name.
    """
    if ap50_values is None:
        ap50_values = [0.92, 0.87, 0.90, 0.85, 0.88, 0.83]

    results = MagicMock()
    results.names = {i: name for i, name in enumerate(CLASS_NAMES)}
    results.box.ap50 = ap50_values
    results.box.map50 = map50
    results.box.map = map50_95
    results.box.mp = mp
    results.box.mr = mr
    return results


def _sample_metrics() -> dict[str, Any]:
    """Return a realistic metrics dict for testing display/logging functions."""
    return {
        "mAP50": 0.88,
        "mAP50_95": 0.62,
        "precision": 0.90,
        "recall": 0.85,
        "per_class": {
            "cardboard": {"ap50": 0.92},
            "glass": {"ap50": 0.87},
            "metal": {"ap50": 0.90},
            "paper": {"ap50": 0.85},
            "plastic": {"ap50": 0.88},
            "trash": {"ap50": 0.83},
        },
    }


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    """Tests for ``evaluate_model`` — loads YOLO model, runs val, parses."""

    @patch("src.training.evaluate.YOLO", create=True)
    def test_calls_ultralytics_val_with_test_split(
        self, mock_yolo_cls: MagicMock, tmp_path: Path
    ) -> None:
        from src.training.evaluate import evaluate_model

        config = _resolved_config(tmp_path)
        model_path = tmp_path / "best.pt"
        model_path.write_bytes(b"weights")
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("path: .", encoding="utf-8")

        mock_model = MagicMock()
        mock_model.val.return_value = _make_mock_results()
        mock_yolo_cls.return_value = mock_model

        evaluate_model(model_path, data_yaml, config)

        mock_yolo_cls.assert_called_once_with(str(model_path))
        mock_model.val.assert_called_once()
        val_kwargs = mock_model.val.call_args[1]
        assert val_kwargs["data"] == str(data_yaml)
        assert val_kwargs["split"] == "test"

    @patch("src.training.evaluate.YOLO", create=True)
    def test_passes_training_config_params(
        self, mock_yolo_cls: MagicMock, tmp_path: Path
    ) -> None:
        from src.training.evaluate import evaluate_model

        config = _resolved_config(tmp_path)
        config["training"]["imgsz"] = 416
        config["training"]["batch"] = 8
        config["training"]["device"] = "cpu"

        model_path = tmp_path / "best.pt"
        model_path.write_bytes(b"weights")

        mock_model = MagicMock()
        mock_model.val.return_value = _make_mock_results()
        mock_yolo_cls.return_value = mock_model

        evaluate_model(model_path, tmp_path / "data.yaml", config)

        val_kwargs = mock_model.val.call_args[1]
        assert val_kwargs["imgsz"] == 416
        assert val_kwargs["batch"] == 8
        assert val_kwargs["device"] == "cpu"

    @patch("src.training.evaluate.YOLO", create=True)
    def test_parses_results_into_metrics_dict(
        self, mock_yolo_cls: MagicMock, tmp_path: Path
    ) -> None:
        from src.training.evaluate import evaluate_model

        config = _resolved_config(tmp_path)
        model_path = tmp_path / "best.pt"
        model_path.write_bytes(b"weights")

        mock_model = MagicMock()
        mock_model.val.return_value = _make_mock_results(
            ap50_values=[0.92, 0.87, 0.90, 0.85, 0.88, 0.83],
            map50=0.88,
            map50_95=0.62,
            mp=0.90,
            mr=0.85,
        )
        mock_yolo_cls.return_value = mock_model

        metrics = evaluate_model(model_path, tmp_path / "data.yaml", config)

        assert metrics["mAP50"] == pytest.approx(0.88)
        assert metrics["mAP50_95"] == pytest.approx(0.62)
        assert metrics["precision"] == pytest.approx(0.90)
        assert metrics["recall"] == pytest.approx(0.85)

    def test_raises_file_not_found_for_missing_model(
        self, tmp_path: Path
    ) -> None:
        from src.training.evaluate import evaluate_model

        config = _resolved_config(tmp_path)
        missing = tmp_path / "nonexistent.pt"

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            evaluate_model(missing, tmp_path / "data.yaml", config)

    @patch("src.training.evaluate.YOLO", create=True)
    def test_per_class_ap_extracted_for_all_classes(
        self, mock_yolo_cls: MagicMock, tmp_path: Path
    ) -> None:
        from src.training.evaluate import evaluate_model

        config = _resolved_config(tmp_path)
        model_path = tmp_path / "best.pt"
        model_path.write_bytes(b"weights")

        ap_values = [0.92, 0.87, 0.90, 0.85, 0.88, 0.83]
        mock_model = MagicMock()
        mock_model.val.return_value = _make_mock_results(ap50_values=ap_values)
        mock_yolo_cls.return_value = mock_model

        metrics = evaluate_model(model_path, tmp_path / "data.yaml", config)

        per_class = metrics["per_class"]
        assert len(per_class) == 6
        assert per_class["cardboard"]["ap50"] == pytest.approx(0.92)
        assert per_class["glass"]["ap50"] == pytest.approx(0.87)
        assert per_class["metal"]["ap50"] == pytest.approx(0.90)
        assert per_class["paper"]["ap50"] == pytest.approx(0.85)
        assert per_class["plastic"]["ap50"] == pytest.approx(0.88)
        assert per_class["trash"]["ap50"] == pytest.approx(0.83)


# ---------------------------------------------------------------------------
# print_metrics_table
# ---------------------------------------------------------------------------


class TestPrintMetricsTable:
    """Tests for ``print_metrics_table`` — formatted stdout output."""

    def test_header_columns(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        from src.training.evaluate import print_metrics_table

        print_metrics_table(_sample_metrics(), CLASS_NAMES)
        output = capsys.readouterr().out

        assert "Class" in output
        assert "AP@0.5" in output
        assert "Precision" in output
        assert "Recall" in output

    def test_all_class_names_appear(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        from src.training.evaluate import print_metrics_table

        print_metrics_table(_sample_metrics(), CLASS_NAMES)
        output = capsys.readouterr().out

        for name in CLASS_NAMES:
            assert name in output, f"Class '{name}' missing from table output"

    def test_summary_rows(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        from src.training.evaluate import print_metrics_table

        print_metrics_table(_sample_metrics(), CLASS_NAMES)
        output = capsys.readouterr().out

        assert "mAP@0.5" in output
        assert "mAP@0.5:0.95" in output

    def test_ap_values_formatted(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        from src.training.evaluate import print_metrics_table

        print_metrics_table(_sample_metrics(), CLASS_NAMES)
        output = capsys.readouterr().out

        # cardboard AP@0.5 = 0.92 should appear as "0.9200"
        assert "0.9200" in output


# ---------------------------------------------------------------------------
# plot_per_class_ap
# ---------------------------------------------------------------------------


class TestPlotPerClassAp:
    """Tests for ``plot_per_class_ap`` — matplotlib bar chart."""

    def test_creates_png_file(self, tmp_path: Path) -> None:
        from src.training.evaluate import plot_per_class_ap

        output = tmp_path / "charts" / "per_class_ap.png"
        plot_per_class_ap(_sample_metrics(), CLASS_NAMES, output)

        assert output.is_file()
        assert output.suffix == ".png"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        from src.training.evaluate import plot_per_class_ap

        output = tmp_path / "a" / "b" / "c" / "chart.png"
        plot_per_class_ap(_sample_metrics(), CLASS_NAMES, output)

        assert output.is_file()

    def test_target_line_at_085(self, tmp_path: Path) -> None:
        """Verify the chart has a vertical line at x=0.85 (the target)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from src.training.evaluate import plot_per_class_ap

        output = tmp_path / "chart.png"

        # Spy on axvline via pyplot
        original_axvline = plt.Axes.axvline
        axvline_calls: list[Any] = []

        def _spy_axvline(self_ax, *args, **kwargs):
            axvline_calls.append((args, kwargs))
            return original_axvline(self_ax, *args, **kwargs)

        with patch.object(plt.Axes, "axvline", _spy_axvline):
            plot_per_class_ap(_sample_metrics(), CLASS_NAMES, output)

        # At least one axvline call should have x=0.85
        x_values = [c[1].get("x", c[0][0] if c[0] else None) for c in axvline_calls]
        assert any(abs(v - 0.85) < 1e-6 for v in x_values if v is not None), (
            f"Expected axvline at x=0.85, got calls with x={x_values}"
        )

    def test_returns_output_path(self, tmp_path: Path) -> None:
        from src.training.evaluate import plot_per_class_ap

        output = tmp_path / "chart.png"
        result = plot_per_class_ap(_sample_metrics(), CLASS_NAMES, output)

        assert result == output


# ---------------------------------------------------------------------------
# log_metrics_mlflow
# ---------------------------------------------------------------------------


class TestLogMetricsMlflow:
    """Tests for ``log_metrics_mlflow`` — logs metrics to MLflow."""

    @patch("src.training.evaluate.mlflow")
    def test_logs_scalar_metrics(self, mock_mlflow: MagicMock) -> None:
        from src.training.evaluate import log_metrics_mlflow

        metrics = _sample_metrics()
        log_metrics_mlflow(metrics)

        logged = {c.args[0]: c.args[1] for c in mock_mlflow.log_metric.call_args_list}
        assert logged["mAP50"] == pytest.approx(0.88)
        assert logged["mAP50_95"] == pytest.approx(0.62)
        assert logged["precision"] == pytest.approx(0.90)
        assert logged["recall"] == pytest.approx(0.85)

    @patch("src.training.evaluate.mlflow")
    def test_logs_per_class_ap(self, mock_mlflow: MagicMock) -> None:
        from src.training.evaluate import log_metrics_mlflow

        metrics = _sample_metrics()
        log_metrics_mlflow(metrics)

        logged = {c.args[0]: c.args[1] for c in mock_mlflow.log_metric.call_args_list}
        assert logged["ap_cardboard"] == pytest.approx(0.92)
        assert logged["ap_glass"] == pytest.approx(0.87)
        assert logged["ap_metal"] == pytest.approx(0.90)
        assert logged["ap_paper"] == pytest.approx(0.85)
        assert logged["ap_plastic"] == pytest.approx(0.88)
        assert logged["ap_trash"] == pytest.approx(0.83)

    @patch("src.training.evaluate.mlflow")
    def test_total_metric_count(self, mock_mlflow: MagicMock) -> None:
        from src.training.evaluate import log_metrics_mlflow

        log_metrics_mlflow(_sample_metrics())

        # 4 scalar + 6 per-class = 10
        assert mock_mlflow.log_metric.call_count == 10

    @patch("src.training.evaluate.mlflow")
    def test_sanitizes_class_name_spaces(self, mock_mlflow: MagicMock) -> None:
        from src.training.evaluate import log_metrics_mlflow

        metrics = {
            "mAP50": 0.5,
            "mAP50_95": 0.3,
            "precision": 0.5,
            "recall": 0.5,
            "per_class": {"my class": {"ap50": 0.7}},
        }
        log_metrics_mlflow(metrics)

        logged_keys = [c.args[0] for c in mock_mlflow.log_metric.call_args_list]
        assert "ap_my_class" in logged_keys


# ---------------------------------------------------------------------------
# main — CLI entry point
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for ``main`` — CLI orchestration."""

    @patch("src.training.evaluate.log_metrics_mlflow")
    @patch("src.training.evaluate.plot_per_class_ap")
    @patch("src.training.evaluate.print_metrics_table")
    @patch("src.training.evaluate.evaluate_model")
    @patch("src.training.evaluate.load_config")
    def test_main_calls_evaluate_and_plot(
        self,
        mock_load_config: MagicMock,
        mock_evaluate: MagicMock,
        mock_print: MagicMock,
        mock_plot: MagicMock,
        mock_log: MagicMock,
        tmp_path: Path,
    ) -> None:
        from src.training.evaluate import main

        config = _resolved_config(tmp_path)
        mock_load_config.return_value = config
        mock_evaluate.return_value = _sample_metrics()

        # Create the model file so it passes the existence check
        model_path = tmp_path / "models" / "yolov8n-waste-best.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"weights")

        main(["--config", str(tmp_path / "config.yaml")])

        mock_evaluate.assert_called_once()
        mock_print.assert_called_once()
        mock_plot.assert_called_once()
        mock_log.assert_called_once()

    @patch("src.training.evaluate.log_metrics_mlflow")
    @patch("src.training.evaluate.plot_per_class_ap")
    @patch("src.training.evaluate.print_metrics_table")
    @patch("src.training.evaluate.evaluate_model")
    @patch("src.training.evaluate.load_config")
    def test_main_no_plot_skips_chart(
        self,
        mock_load_config: MagicMock,
        mock_evaluate: MagicMock,
        mock_print: MagicMock,
        mock_plot: MagicMock,
        mock_log: MagicMock,
        tmp_path: Path,
    ) -> None:
        from src.training.evaluate import main

        config = _resolved_config(tmp_path)
        mock_load_config.return_value = config
        mock_evaluate.return_value = _sample_metrics()

        model_path = tmp_path / "models" / "yolov8n-waste-best.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"weights")

        main(["--config", str(tmp_path / "config.yaml"), "--no-plot"])

        mock_plot.assert_not_called()
        mock_evaluate.assert_called_once()

    @patch("src.training.evaluate.log_metrics_mlflow")
    @patch("src.training.evaluate.plot_per_class_ap")
    @patch("src.training.evaluate.print_metrics_table")
    @patch("src.training.evaluate.evaluate_model")
    @patch("src.training.evaluate.load_config")
    def test_main_default_model_path(
        self,
        mock_load_config: MagicMock,
        mock_evaluate: MagicMock,
        mock_print: MagicMock,
        mock_plot: MagicMock,
        mock_log: MagicMock,
        tmp_path: Path,
    ) -> None:
        from src.training.evaluate import main

        config = _resolved_config(tmp_path)
        mock_load_config.return_value = config
        mock_evaluate.return_value = _sample_metrics()

        # Default model path: models/yolov8n-waste-best.pt (relative to project_root)
        model_path = tmp_path / "models" / "yolov8n-waste-best.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"weights")

        main(["--config", str(tmp_path / "config.yaml")])

        # evaluate_model called with the default model path
        call_args = mock_evaluate.call_args[0]
        model_arg = call_args[0]
        assert "yolov8n-waste-best.pt" in str(model_arg)

    @patch("src.training.evaluate.load_config")
    def test_main_requires_config(self, mock_load_config: MagicMock) -> None:
        from src.training.evaluate import main

        with pytest.raises(SystemExit):
            main([])
