"""Tests for src.training.convert_dataset — CSV parsing, label writing, quality gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

import cv2
import numpy as np

from src.training.convert_dataset import (
    BoundingBox,
    CenterCropAnnotator,
    ContourAnnotator,
    ConversionReport,
    ImageLabel,
    QualityGateError,
    _convert_split,
    convert_dataset,
    parse_split_csv,
    quality_gate,
    write_data_yaml,
    write_yolo_label,
)

# Minimal valid JPEG bytes (1×1 white pixel) — duplicated from conftest to
# avoid import-path issues when ``tests`` is not a package.
_MINIMAL_JPEG = bytes(
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


def _write_jpeg(path: Path) -> None:
    """Write a minimal valid JPEG file."""
    path.write_bytes(_MINIMAL_JPEG)


# -----------------------------------------------------------------------
# parse_split_csv
# -----------------------------------------------------------------------


class TestParseSplitCsv:
    def test_parse_split_csv_valid(self, sample_classes_csv: Path) -> None:
        """Parses the 5-row fixture correctly."""
        labels = parse_split_csv(sample_classes_csv)

        assert len(labels) == 5
        assert labels[0] == ImageLabel(filename="img001.jpg", class_id=0, class_name="cardboard")
        assert labels[1] == ImageLabel(filename="img002.jpg", class_id=0, class_name="cardboard")
        assert labels[2] == ImageLabel(filename="img003.jpg", class_id=1, class_name="glass")
        assert labels[3] == ImageLabel(filename="img004.jpg", class_id=2, class_name="metal")
        assert labels[4] == ImageLabel(filename="img005.jpg", class_id=3, class_name="paper")

    def test_parse_split_csv_class_mapping(self, sample_classes_csv: Path) -> None:
        """Verifies the fixed class mapping: cardboard=0, glass=1, ..., trash=5."""
        labels = parse_split_csv(sample_classes_csv)
        class_ids = {lbl.class_name: lbl.class_id for lbl in labels}

        assert class_ids["cardboard"] == 0
        assert class_ids["glass"] == 1
        assert class_ids["metal"] == 2
        assert class_ids["paper"] == 3

    def test_parse_split_csv_strips_spaces(self, tmp_path: Path) -> None:
        """Handles header with spaces after commas."""
        csv_path = tmp_path / "_classes.csv"
        csv_path.write_text(
            "filename, cardboard, glass, metal, paper, plastic, trash\n"
            "test.jpg, 0, 0, 0, 0, 0, 1\n",
            encoding="utf-8",
        )

        labels = parse_split_csv(csv_path)
        assert len(labels) == 1
        assert labels[0].class_name == "trash"
        assert labels[0].class_id == 5

    def test_parse_split_csv_invalid_onehot_zero(self, tmp_path: Path) -> None:
        """Raises ValueError when no column has a 1."""
        csv_path = tmp_path / "_classes.csv"
        csv_path.write_text(
            "filename, cardboard, glass, metal, paper, plastic, trash\n"
            "bad.jpg, 0, 0, 0, 0, 0, 0\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="expected exactly one '1'"):
            parse_split_csv(csv_path)

    def test_parse_split_csv_invalid_onehot_multiple(self, tmp_path: Path) -> None:
        """Raises ValueError when more than one column has a 1."""
        csv_path = tmp_path / "_classes.csv"
        csv_path.write_text(
            "filename, cardboard, glass, metal, paper, plastic, trash\n"
            "bad.jpg, 1, 1, 0, 0, 0, 0\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="expected exactly one '1'"):
            parse_split_csv(csv_path)


# -----------------------------------------------------------------------
# write_yolo_label
# -----------------------------------------------------------------------


class TestWriteYoloLabel:
    def test_write_yolo_label_format(self, tmp_path: Path) -> None:
        """Writes correct normalized coordinates in YOLO format."""
        label_path = tmp_path / "labels" / "test.txt"
        bbox = BoundingBox(x_center=0.5, y_center=0.6, width=0.8, height=0.9)

        write_yolo_label(label_path, bbox, class_id=2)

        content = label_path.read_text(encoding="utf-8")
        assert content == "2 0.500000 0.600000 0.800000 0.900000\n"

    def test_write_yolo_label_normalized(self, tmp_path: Path) -> None:
        """Values are formatted as floats in [0, 1] range."""
        label_path = tmp_path / "labels" / "norm.txt"
        bbox = BoundingBox(x_center=0.123456, y_center=0.789012, width=0.5, height=0.3)

        write_yolo_label(label_path, bbox, class_id=0)

        content = label_path.read_text(encoding="utf-8").strip()
        parts = content.split()
        assert len(parts) == 5
        assert int(parts[0]) == 0
        for val_str in parts[1:]:
            val = float(val_str)
            assert 0.0 <= val <= 1.0

    def test_write_yolo_label_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Creates parent directories if they don't exist."""
        label_path = tmp_path / "deep" / "nested" / "labels" / "test.txt"
        bbox = BoundingBox(x_center=0.5, y_center=0.5, width=0.7, height=0.7)

        write_yolo_label(label_path, bbox, class_id=0)

        assert label_path.is_file()


# -----------------------------------------------------------------------
# write_data_yaml
# -----------------------------------------------------------------------


class TestWriteDataYaml:
    def test_write_data_yaml_structure(self, tmp_path: Path) -> None:
        """Has path, train, val, test, names keys with correct values."""
        output_dir = tmp_path / "dataset"
        output_dir.mkdir()
        config: dict[str, Any] = {
            "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
        }

        write_data_yaml(output_dir, config)

        yaml_path = output_dir / "data.yaml"
        assert yaml_path.is_file()

        with open(yaml_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        assert "path" in data
        assert data["train"] == "train/images"
        assert data["val"] == "valid/images"
        assert data["test"] == "test/images"
        assert "names" in data
        assert data["names"][0] == "cardboard"
        assert data["names"][5] == "trash"
        # path should be absolute
        assert Path(data["path"]).is_absolute()

    def test_write_data_yaml_class_count(self, tmp_path: Path) -> None:
        """names dict has exactly 6 entries."""
        output_dir = tmp_path / "dataset"
        output_dir.mkdir()
        config: dict[str, Any] = {
            "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
        }

        write_data_yaml(output_dir, config)

        with open(output_dir / "data.yaml", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        assert len(data["names"]) == 6


# -----------------------------------------------------------------------
# quality_gate
# -----------------------------------------------------------------------


class TestQualityGate:
    def test_quality_gate_passes_low_rejection(self) -> None:
        """3% rejection → no error."""
        report = ConversionReport(
            total_images=100,
            successful=97,
            rejected=["a.jpg", "b.jpg", "c.jpg"],
        )
        # Should not raise
        quality_gate(report, max_rejection_rate=0.05)

    def test_quality_gate_fails_high_rejection(self) -> None:
        """10% rejection → QualityGateError."""
        report = ConversionReport(
            total_images=100,
            successful=90,
            rejected=[f"img{i}.jpg" for i in range(10)],
        )

        with pytest.raises(QualityGateError) as exc_info:
            quality_gate(report, max_rejection_rate=0.05)

        assert exc_info.value.report is report
        assert exc_info.value.max_rate == 0.05

    def test_quality_gate_report_lists_rejected(self) -> None:
        """Rejected filenames appear in the error message."""
        report = ConversionReport(
            total_images=100,
            successful=85,
            rejected=["bad1.jpg", "bad2.jpg", "bad3.jpg"]
            + [f"img{i}.jpg" for i in range(12)],
        )

        with pytest.raises(QualityGateError) as exc_info:
            quality_gate(report, max_rejection_rate=0.05)

        msg = str(exc_info.value)
        assert "bad1.jpg" in msg
        assert "bad2.jpg" in msg

    def test_quality_gate_zero_images(self) -> None:
        """Zero images → 0% rejection rate → no error."""
        report = ConversionReport(total_images=0, successful=0)
        quality_gate(report, max_rejection_rate=0.05)

    def test_quality_gate_exact_threshold(self) -> None:
        """Exactly 5% rejection → passes (threshold is >, not >=)."""
        report = ConversionReport(
            total_images=100,
            successful=95,
            rejected=[f"img{i}.jpg" for i in range(5)],
        )
        quality_gate(report, max_rejection_rate=0.05)


# -----------------------------------------------------------------------
# _convert_split
# -----------------------------------------------------------------------


class TestConvertSplit:
    def test_convert_split_with_mock_annotator(
        self, sample_split_dir: Path, output_dir: Path
    ) -> None:
        """Uses mock annotator, verifies output structure."""
        mock_annotator = MagicMock()
        mock_annotator.annotate.return_value = BoundingBox(
            x_center=0.5, y_center=0.5, width=0.7, height=0.7
        )

        classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        report = ConversionReport(annotator_model="mock")

        _convert_split(sample_split_dir, output_dir, mock_annotator, classes, report)

        # Check directory structure
        assert (output_dir / "images").is_dir()
        assert (output_dir / "labels").is_dir()

        # Check all 5 images were processed
        assert report.total_images == 5
        assert report.successful == 5
        assert len(report.rejected) == 0

        # Check label files exist
        for name in ("img001", "img002", "img003", "img004", "img005"):
            assert (output_dir / "labels" / f"{name}.txt").is_file()

        # Check images were copied
        for name in ("img001.jpg", "img002.jpg", "img003.jpg", "img004.jpg", "img005.jpg"):
            assert (output_dir / "images" / name).is_file()

    def test_convert_split_rejection_tracking(
        self, sample_split_dir: Path, output_dir: Path
    ) -> None:
        """Annotator returns None for some images — they appear in report."""
        mock_annotator = MagicMock()
        # Return None for img002 and img004
        def side_effect(path: Path) -> BoundingBox | None:
            if path.name in ("img002.jpg", "img004.jpg"):
                return None
            return BoundingBox(x_center=0.5, y_center=0.5, width=0.7, height=0.7)

        mock_annotator.annotate.side_effect = side_effect

        classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        report = ConversionReport(annotator_model="mock")

        _convert_split(sample_split_dir, output_dir, mock_annotator, classes, report)

        assert report.total_images == 5
        assert report.successful == 3
        assert len(report.rejected) == 2
        assert "img002.jpg" in report.rejected
        assert "img004.jpg" in report.rejected

        # Rejected images should NOT have label files
        assert not (output_dir / "labels" / "img002.txt").exists()
        assert not (output_dir / "labels" / "img004.txt").exists()

        # Successful images should have label files
        assert (output_dir / "labels" / "img001.txt").is_file()
        assert (output_dir / "labels" / "img003.txt").is_file()
        assert (output_dir / "labels" / "img005.txt").is_file()

    def test_convert_split_missing_image(
        self, sample_split_dir: Path, output_dir: Path
    ) -> None:
        """CSV references nonexistent file → logged, skipped, counted."""
        # Remove one image to simulate missing file
        (sample_split_dir / "img003.jpg").unlink()

        mock_annotator = MagicMock()
        mock_annotator.annotate.return_value = BoundingBox(
            x_center=0.5, y_center=0.5, width=0.7, height=0.7
        )

        classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        report = ConversionReport(annotator_model="mock")

        _convert_split(sample_split_dir, output_dir, mock_annotator, classes, report)

        assert report.total_images == 5
        assert "img003.jpg" in report.missing_images
        assert "img003.jpg" in report.rejected
        assert report.successful == 4

    def test_convert_split_class_distribution(
        self, sample_split_dir: Path, output_dir: Path
    ) -> None:
        """Class distribution is tracked correctly."""
        mock_annotator = MagicMock()
        mock_annotator.annotate.return_value = BoundingBox(
            x_center=0.5, y_center=0.5, width=0.7, height=0.7
        )

        classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        report = ConversionReport(annotator_model="mock")

        _convert_split(sample_split_dir, output_dir, mock_annotator, classes, report)

        # Fixture has: 2 cardboard, 1 glass, 1 metal, 1 paper
        assert report.class_distribution["cardboard"] == 2
        assert report.class_distribution["glass"] == 1
        assert report.class_distribution["metal"] == 1
        assert report.class_distribution["paper"] == 1


# -----------------------------------------------------------------------
# convert_dataset (end-to-end)
# -----------------------------------------------------------------------


class TestConvertDataset:
    @staticmethod
    def _build_config(tmp_path: Path, source_dir: Path) -> dict[str, Any]:
        """Build a config dict pointing at *source_dir* (no side-effects)."""
        output_dir = tmp_path / "output_dataset"
        return {
            "_config_dir": str(tmp_path),
            "project_root": str(tmp_path),
            "source_data": str(source_dir),
            "output_dataset": str(output_dir),
            "models_dir": str(tmp_path / "models"),
            "runs_dir": str(tmp_path / "runs"),
            "auto_annotate": {
                "method": "center-crop",
                "model": "none",
                "confidence_threshold": 0.30,
                "batch_size": 8,
                "device": 0,
            },
            "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
        }

    @staticmethod
    def _populate_splits(source_dir: Path, splits: dict[str, list[tuple[str, int]]]) -> None:
        """Create split directories with CSV + tiny JPEGs.

        *splits* maps split name → list of ``(filename, class_id)`` pairs.
        ``class_id`` maps to the fixed class order.
        """
        classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        for split_name, entries in splits.items():
            split_dir = source_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            lines = ["filename, cardboard, glass, metal, paper, plastic, trash"]
            for fname, cid in entries:
                one_hot = ["0"] * 6
                one_hot[cid] = "1"
                lines.append(f"{fname}, {', '.join(one_hot)}")
                _write_jpeg(split_dir / fname)

            (split_dir / "_classes.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_convert_dataset_with_mock_annotator(self, tmp_path: Path) -> None:
        """Full pipeline with center-crop annotator, verifies directory structure."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        self._populate_splits(source_dir, {
            "train": [("train_001.jpg", 0), ("train_002.jpg", 1)],
            "valid": [("valid_001.jpg", 0), ("valid_002.jpg", 1)],
            "test":  [("test_001.jpg", 0), ("test_002.jpg", 1)],
        })
        config = self._build_config(tmp_path, source_dir)

        report = convert_dataset(config)

        output = Path(config["output_dataset"])

        # Check structure
        assert (output / "data.yaml").is_file()
        for split in ("train", "valid", "test"):
            assert (output / split / "images").is_dir()
            assert (output / split / "labels").is_dir()

        # Check counts
        assert report.total_images == 6  # 2 per split × 3 splits
        assert report.successful == 6
        assert len(report.rejected) == 0

        # Check data.yaml content
        with open(output / "data.yaml", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert data["train"] == "train/images"
        assert data["val"] == "valid/images"
        assert data["test"] == "test/images"
        assert len(data["names"]) == 6

    def test_convert_dataset_quality_gate_fail(self, tmp_path: Path) -> None:
        """Rejection rate > 5% raises QualityGateError."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create a train split with 20 images in CSV but only 15 on disk
        # → 5 missing = 25% rejection rate
        train_dir = source_dir / "train"
        train_dir.mkdir()

        csv_lines = ["filename, cardboard, glass, metal, paper, plastic, trash"]
        for i in range(20):
            csv_lines.append(f"img{i:03d}.jpg, 1, 0, 0, 0, 0, 0")
        (train_dir / "_classes.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

        # Only create 15 of 20 images
        for i in range(15):
            _write_jpeg(train_dir / f"img{i:03d}.jpg")

        config = self._build_config(tmp_path, source_dir)

        with pytest.raises(QualityGateError) as exc_info:
            convert_dataset(config)

        assert exc_info.value.report.total_images == 20
        assert len(exc_info.value.report.rejected) == 5

    def test_convert_dataset_missing_split_dir(self, tmp_path: Path) -> None:
        """Missing split directory is handled gracefully (logged, skipped)."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Only create train, no valid or test
        self._populate_splits(source_dir, {
            "train": [("img001.jpg", 0)],
        })

        config = self._build_config(tmp_path, source_dir)
        report = convert_dataset(config)

        # Only train split was processed
        assert report.total_images == 1
        assert report.successful == 1


# -----------------------------------------------------------------------
# CenterCropAnnotator
# -----------------------------------------------------------------------


class TestCenterCropAnnotator:
    def test_returns_fixed_bbox(self) -> None:
        """Returns center 70% bounding box regardless of image."""
        annotator = CenterCropAnnotator()
        bbox = annotator.annotate(Path("any_image.jpg"))

        assert bbox.x_center == 0.5
        assert bbox.y_center == 0.5
        assert bbox.width == 0.7
        assert bbox.height == 0.7

    def test_batch_returns_list(self) -> None:
        """Batch returns one bbox per input path."""
        annotator = CenterCropAnnotator()
        paths = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]
        results = annotator.annotate_batch(paths)

        assert len(results) == 3
        assert all(b.width == 0.7 for b in results)


# -----------------------------------------------------------------------
# ContourAnnotator
# -----------------------------------------------------------------------


def _save_img(array: np.ndarray, path: Path) -> Path:
    """Write a numpy BGR array to *path* as PNG and return the path."""
    cv2.imwrite(str(path), array)
    return path


class TestContourAnnotator:
    """Tests for the OpenCV contour-based fallback annotator."""

    def test_contour_annotator_simple_object(self, tmp_path: Path) -> None:
        """Black rectangle on white background → bbox matches the rect."""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        # Rect from (50, 30) to (250, 170) → w=200, h=140
        cv2.rectangle(img, (50, 30), (250, 170), (0, 0, 0), -1)
        img_path = _save_img(img, tmp_path / "rect.png")

        annotator = ContourAnnotator()
        bbox = annotator.annotate(img_path)

        assert bbox is not None
        # Expected center ≈ (150/300, 100/200) = (0.5, 0.5)
        # Expected size  ≈ (200/300, 140/200) ≈ (0.667, 0.7)
        assert abs(bbox.x_center - 0.5) < 0.05
        assert abs(bbox.y_center - 0.5) < 0.05
        assert abs(bbox.width - 200 / 300) < 0.05
        assert abs(bbox.height - 140 / 200) < 0.05

    def test_contour_annotator_no_object(self, tmp_path: Path) -> None:
        """All-black image → no contours → returns None."""
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img_path = _save_img(img, tmp_path / "black.png")

        annotator = ContourAnnotator()
        assert annotator.annotate(img_path) is None

    def test_contour_annotator_min_area_filter(self, tmp_path: Path) -> None:
        """Tiny contour below min_area_ratio threshold → None."""
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        # 3×3 white square → area ≈ 9 px, img_area = 60000 → ratio ≈ 0.00015
        cv2.rectangle(img, (100, 100), (103, 103), (255, 255, 255), -1)
        img_path = _save_img(img, tmp_path / "tiny.png")

        annotator = ContourAnnotator(min_area_ratio=0.01)
        assert annotator.annotate(img_path) is None

    def test_contour_annotator_returns_bounding_box(
        self, tmp_path: Path
    ) -> None:
        """Return type is BoundingBox with all coordinates in [0, 1]."""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (40, 20), (260, 180), (0, 0, 0), -1)
        img_path = _save_img(img, tmp_path / "obj.png")

        annotator = ContourAnnotator()
        bbox = annotator.annotate(img_path)

        assert isinstance(bbox, BoundingBox)
        assert 0.0 <= bbox.x_center <= 1.0
        assert 0.0 <= bbox.y_center <= 1.0
        assert 0.0 <= bbox.width <= 1.0
        assert 0.0 <= bbox.height <= 1.0

    def test_contour_annotator_batch(self, tmp_path: Path) -> None:
        """Batch processing returns a list of correct length."""
        # Image with object
        img_with = np.ones((200, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(img_with, (50, 30), (250, 170), (0, 0, 0), -1)
        p1 = _save_img(img_with, tmp_path / "with_obj.png")

        # Image without object (all black)
        img_without = np.zeros((200, 300, 3), dtype=np.uint8)
        p2 = _save_img(img_without, tmp_path / "no_obj.png")

        # Another image with object
        p3 = _save_img(img_with, tmp_path / "with_obj2.png")

        annotator = ContourAnnotator()
        results = annotator.annotate_batch([p1, p2, p3])

        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None

    def test_contour_annotator_missing_file(self, tmp_path: Path) -> None:
        """Nonexistent path → cv2.imread returns None → annotate returns None."""
        annotator = ContourAnnotator()
        result = annotator.annotate(tmp_path / "does_not_exist.jpg")
        assert result is None

    def test_contour_annotator_normalization(self, tmp_path: Path) -> None:
        """100×200 image with rect at known position → correct normalized values."""
        # Image: 200 tall × 300 wide (h×w)
        # Rect from (60, 40) to (240, 160) → x=60, y=40, w=180, h=120
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (60, 40), (240, 160), (0, 0, 0), -1)
        img_path = _save_img(img, tmp_path / "norm.png")

        annotator = ContourAnnotator()
        bbox = annotator.annotate(img_path)

        assert bbox is not None
        # Expected (approx): x_center ≈ (60+90)/300 = 0.5, y_center ≈ (40+60)/200 = 0.5
        # width ≈ 180/300 = 0.6, height ≈ 120/200 = 0.6
        assert abs(bbox.x_center - 0.5) < 0.05
        assert abs(bbox.y_center - 0.5) < 0.05
        assert abs(bbox.width - 0.6) < 0.05
        assert abs(bbox.height - 0.6) < 0.05

    def test_contour_annotator_largest_contour_selected(
        self, tmp_path: Path
    ) -> None:
        """Two rectangles → the larger one is selected."""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        # Large rect: (10,10)→(110,90) → 100×80 = 8000 px
        cv2.rectangle(img, (10, 10), (110, 90), (0, 0, 0), -1)
        # Small rect: (200,150)→(230,170) → 30×20 = 600 px
        cv2.rectangle(img, (200, 150), (230, 170), (0, 0, 0), -1)
        img_path = _save_img(img, tmp_path / "two_rects.png")

        annotator = ContourAnnotator()
        bbox = annotator.annotate(img_path)

        assert bbox is not None
        # Large rect center ≈ (60/300, 50/200) = (0.2, 0.25)
        assert abs(bbox.x_center - 60 / 300) < 0.05
        assert abs(bbox.y_center - 50 / 200) < 0.05
