"""Tests for src.training.validate_dataset — label format, split, full dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.training.validate_dataset import (
    ValidationReport,
    validate_dataset,
    validate_label_file,
    validate_split,
)

# Minimal valid JPEG bytes (1x1 white pixel) — same as conftest.
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

_NUM_CLASSES = 6
_CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def _write_jpeg(path: Path) -> None:
    path.write_bytes(_MINIMAL_JPEG)


def _write_label(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# -----------------------------------------------------------------------
# validate_label_file
# -----------------------------------------------------------------------


class TestValidateLabelFile:
    """Tests for single-file label validation."""

    def test_valid(self, tmp_path: Path) -> None:
        """Correct format produces no errors."""
        lbl = tmp_path / "img001.txt"
        _write_label(lbl, "0 0.487500 0.512500 0.750000 0.820000\n")
        errors = validate_label_file(lbl, _NUM_CLASSES)
        assert errors == []

    def test_valid_multiple_lines(self, tmp_path: Path) -> None:
        """Multiple valid lines in one file are accepted."""
        lbl = tmp_path / "img002.txt"
        _write_label(
            lbl,
            "0 0.5 0.5 0.4 0.6\n"
            "3 0.3 0.3 0.2 0.2\n",
        )
        errors = validate_label_file(lbl, _NUM_CLASSES)
        assert errors == []

    def test_wrong_value_count(self, tmp_path: Path) -> None:
        """4 values instead of 5 produces an error."""
        lbl = tmp_path / "img003.txt"
        _write_label(lbl, "0 0.5 0.5 0.4\n")
        errors = validate_label_file(lbl, _NUM_CLASSES)
        assert len(errors) == 1
        assert "expected 5 values, got 4" in errors[0]

    def test_invalid_class_id(self, tmp_path: Path) -> None:
        """class_id=99 with 6 classes produces an error."""
        lbl = tmp_path / "img004.txt"
        _write_label(lbl, "99 0.5 0.5 0.4 0.6\n")
        errors = validate_label_file(lbl, _NUM_CLASSES)
        assert any("class_id=99 out of range" in e for e in errors)

    def test_coords_out_of_range(self, tmp_path: Path) -> None:
        """x_center=1.5 produces an error."""
        lbl = tmp_path / "img005.txt"
        _write_label(lbl, "0 1.5 0.5 0.4 0.6\n")
        errors = validate_label_file(lbl, _NUM_CLASSES)
        assert any("x_center" in e and "out of range" in e for e in errors)

    def test_coords_negative(self, tmp_path: Path) -> None:
        """Negative coordinate produces an error."""
        lbl = tmp_path / "img005b.txt"
        _write_label(lbl, "0 -0.1 0.5 0.4 0.6\n")
        errors = validate_label_file(lbl, _NUM_CLASSES)
        assert any("x_center" in e and "out of range" in e for e in errors)

    def test_degenerate_box_width(self, tmp_path: Path) -> None:
        """width=0.0001 produces an error."""
        lbl = tmp_path / "img006.txt"
        _write_label(lbl, "0 0.5 0.5 0.0001 0.6\n")
        errors = validate_label_file(lbl, _NUM_CLASSES)
        assert any("width" in e and "degenerate" in e for e in errors)

    def test_degenerate_box_height(self, tmp_path: Path) -> None:
        """height=0.0005 produces an error."""
        lbl = tmp_path / "img007.txt"
        _write_label(lbl, "0 0.5 0.5 0.4 0.0005\n")
        errors = validate_label_file(lbl, _NUM_CLASSES)
        assert any("height" in e and "degenerate" in e for e in errors)

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty label file is valid (no objects)."""
        lbl = tmp_path / "img008.txt"
        _write_label(lbl, "")
        errors = validate_label_file(lbl, _NUM_CLASSES)
        assert errors == []

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        """Blank lines are ignored."""
        lbl = tmp_path / "img009.txt"
        _write_label(lbl, "\n0 0.5 0.5 0.4 0.6\n\n")
        errors = validate_label_file(lbl, _NUM_CLASSES)
        assert errors == []

    def test_non_integer_class_id(self, tmp_path: Path) -> None:
        """Non-integer class_id produces an error."""
        lbl = tmp_path / "img010.txt"
        _write_label(lbl, "abc 0.5 0.5 0.4 0.6\n")
        errors = validate_label_file(lbl, _NUM_CLASSES)
        assert any("not an integer" in e for e in errors)


# -----------------------------------------------------------------------
# validate_split
# -----------------------------------------------------------------------


class TestValidateSplit:
    """Tests for split-level validation."""

    def _make_split(
        self,
        tmp_path: Path,
        image_names: list[str] | None = None,
        label_contents: dict[str, str] | None = None,
    ) -> Path:
        """Helper: create a split directory with images/ and labels/."""
        split_dir = tmp_path / "train"
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "labels").mkdir(parents=True)

        for name in image_names or []:
            _write_jpeg(split_dir / "images" / name)

        for stem, content in (label_contents or {}).items():
            _write_label(split_dir / "labels" / f"{stem}.txt", content)

        return split_dir

    def test_valid_split(self, tmp_path: Path) -> None:
        """Valid split with images and labels passes."""
        split_dir = self._make_split(
            tmp_path,
            image_names=["img001.jpg", "img002.jpg"],
            label_contents={
                "img001": "0 0.5 0.5 0.4 0.6\n",
                "img002": "3 0.3 0.3 0.2 0.2\n",
            },
        )
        report = validate_split(split_dir, _NUM_CLASSES)
        assert report.is_valid
        assert report.total_images == 2
        assert report.total_labels == 2
        assert report.valid_labels == 2
        assert report.invalid_labels == 0

    def test_missing_labels(self, tmp_path: Path) -> None:
        """Image without label is reported in missing_labels."""
        split_dir = self._make_split(
            tmp_path,
            image_names=["img001.jpg", "img002.jpg"],
            label_contents={"img001": "0 0.5 0.5 0.4 0.6\n"},
        )
        report = validate_split(split_dir, _NUM_CLASSES)
        assert not report.is_valid
        assert "img002" in report.missing_labels

    def test_orphan_labels(self, tmp_path: Path) -> None:
        """Label without image is reported in orphan_labels."""
        split_dir = self._make_split(
            tmp_path,
            image_names=["img001.jpg"],
            label_contents={
                "img001": "0 0.5 0.5 0.4 0.6\n",
                "img099": "1 0.5 0.5 0.4 0.6\n",
            },
        )
        report = validate_split(split_dir, _NUM_CLASSES)
        assert not report.is_valid
        assert "img099" in report.orphan_labels

    def test_class_distribution(self, tmp_path: Path) -> None:
        """Class counts are aggregated correctly."""
        split_dir = self._make_split(
            tmp_path,
            image_names=["a.jpg", "b.jpg", "c.jpg"],
            label_contents={
                "a": "0 0.5 0.5 0.4 0.6\n",
                "b": "0 0.5 0.5 0.4 0.6\n",
                "c": "3 0.3 0.3 0.2 0.2\n",
            },
        )
        report = validate_split(split_dir, _NUM_CLASSES)
        assert report.class_distribution[0] == 2
        assert report.class_distribution[3] == 1

    def test_missing_images_dir(self, tmp_path: Path) -> None:
        """Missing images/ directory is reported as an error."""
        split_dir = tmp_path / "train"
        split_dir.mkdir()
        (split_dir / "labels").mkdir()
        report = validate_split(split_dir, _NUM_CLASSES)
        assert not report.is_valid
        assert any("Missing directory" in e for e in report.errors)

    def test_invalid_label_in_split(self, tmp_path: Path) -> None:
        """Invalid label file increments invalid_labels."""
        split_dir = self._make_split(
            tmp_path,
            image_names=["img001.jpg"],
            label_contents={"img001": "99 0.5 0.5 0.4 0.6\n"},
        )
        report = validate_split(split_dir, _NUM_CLASSES)
        assert report.invalid_labels == 1
        assert report.valid_labels == 0
        assert not report.is_valid


# -----------------------------------------------------------------------
# validate_dataset (end-to-end)
# -----------------------------------------------------------------------


class TestValidateDataset:
    """End-to-end tests with a full temp dataset."""

    def _build_dataset(
        self,
        tmp_path: Path,
        train_images: list[str] | None = None,
        train_labels: dict[str, str] | None = None,
        valid_images: list[str] | None = None,
        valid_labels: dict[str, str] | None = None,
    ) -> Path:
        """Build a minimal dataset directory structure."""
        ds = tmp_path / "datasets" / "waste-detect"
        for split_name in ("train", "valid", "test"):
            (ds / split_name / "images").mkdir(parents=True)
            (ds / split_name / "labels").mkdir(parents=True)

        for name in train_images or []:
            _write_jpeg(ds / "train" / "images" / name)
        for stem, content in (train_labels or {}).items():
            _write_label(ds / "train" / "labels" / f"{stem}.txt", content)

        for name in valid_images or []:
            _write_jpeg(ds / "valid" / "images" / name)
        for stem, content in (valid_labels or {}).items():
            _write_label(ds / "valid" / "labels" / f"{stem}.txt", content)

        return ds

    def _make_config(self, tmp_path: Path, ds_path: Path) -> dict[str, Any]:
        return {
            "_config_dir": str(tmp_path),
            "project_root": tmp_path,
            "output_dataset": ds_path,
            "classes": _CLASS_NAMES,
        }

    def test_full_valid_dataset(self, tmp_path: Path) -> None:
        """End-to-end with all valid data passes."""
        ds = self._build_dataset(
            tmp_path,
            train_images=["a.jpg", "b.jpg"],
            train_labels={
                "a": "0 0.5 0.5 0.4 0.6\n",
                "b": "3 0.3 0.3 0.2 0.2\n",
            },
            valid_images=["c.jpg"],
            valid_labels={"c": "5 0.5 0.5 0.4 0.6\n"},
        )
        config = self._make_config(tmp_path, ds)
        report = validate_dataset(config)
        assert report.is_valid
        assert report.total_images == 3
        assert report.valid_labels == 3

    def test_full_dataset_with_errors(self, tmp_path: Path) -> None:
        """End-to-end with invalid label fails."""
        ds = self._build_dataset(
            tmp_path,
            train_images=["a.jpg"],
            train_labels={"a": "99 0.5 0.5 0.4 0.6\n"},
        )
        config = self._make_config(tmp_path, ds)
        report = validate_dataset(config)
        assert not report.is_valid
        assert report.invalid_labels >= 1

    def test_missing_split_directory(self, tmp_path: Path) -> None:
        """Missing split directory is reported but doesn't crash."""
        ds = tmp_path / "datasets" / "waste-detect"
        # Only create train, leave valid and test missing
        (ds / "train" / "images").mkdir(parents=True)
        (ds / "train" / "labels").mkdir(parents=True)
        config = self._make_config(tmp_path, ds)
        report = validate_dataset(config)
        assert not report.is_valid
        assert any("Split directory missing" in e for e in report.errors)
