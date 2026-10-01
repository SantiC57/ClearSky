"""Post-conversion dataset validation for YOLOv8 waste detection.

Validates that every label file conforms to YOLO format, every image has a
corresponding label, and no orphan labels exist.  Produces a summary report
with class distribution.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.training.config import load_config

logger = logging.getLogger(__name__)

# Image extensions considered valid when scanning ``images/`` directories.
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

# Minimum allowed value for width/height to reject degenerate boxes.
_MIN_BOX_DIM = 0.001


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Aggregated validation results for one or more dataset splits."""

    total_images: int = 0
    total_labels: int = 0
    valid_labels: int = 0
    invalid_labels: int = 0
    orphan_labels: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    class_distribution: dict[int, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """True when there are no errors, no orphans, and no missing labels."""
        return (
            len(self.errors) == 0
            and len(self.orphan_labels) == 0
            and len(self.missing_labels) == 0
        )


# ---------------------------------------------------------------------------
# Label-file validation
# ---------------------------------------------------------------------------


def validate_label_file(label_path: Path, num_classes: int) -> list[str]:
    """Validate a single YOLO label file.

    Each line must have exactly 5 space-separated values:
    ``class_id x_center y_center width height``.

    Parameters
    ----------
    label_path:
        Path to the ``.txt`` label file.
    num_classes:
        Total number of classes (valid class_ids are ``[0, num_classes-1]``).

    Returns
    -------
    list[str]
        List of error messages.  Empty when the file is valid.
    """
    errors: list[str] = []
    stem = label_path.stem

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{stem}: cannot read file — {exc}")
        return errors

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue  # skip blank lines

        parts = line.split()
        if len(parts) != 5:
            errors.append(
                f"{stem}:{line_no}: expected 5 values, got {len(parts)}"
            )
            continue

        # --- class_id ---
        try:
            class_id = int(parts[0])
        except ValueError:
            errors.append(
                f"{stem}:{line_no}: class_id is not an integer: '{parts[0]}'"
            )
            continue

        if not (0 <= class_id <= num_classes - 1):
            errors.append(
                f"{stem}:{line_no}: class_id={class_id} out of range "
                f"[0, {num_classes - 1}]"
            )

        # --- coordinates ---
        coord_names = ("x_center", "y_center", "width", "height")
        coords: list[float] = []
        coord_ok = True
        for idx, name in enumerate(coord_names):
            try:
                val = float(parts[idx + 1])
            except ValueError:
                errors.append(
                    f"{stem}:{line_no}: {name} is not a float: '{parts[idx + 1]}'"
                )
                coord_ok = False
                break
            coords.append(val)

        if not coord_ok:
            continue

        x_center, y_center, width, height = coords

        for name, val in zip(coord_names, coords):
            if not (0.0 <= val <= 1.0):
                errors.append(
                    f"{stem}:{line_no}: {name}={val:.6f} out of range [0.0, 1.0]"
                )

        if width <= _MIN_BOX_DIM:
            errors.append(
                f"{stem}:{line_no}: width={width:.6f} is degenerate "
                f"(must be > {_MIN_BOX_DIM})"
            )
        if height <= _MIN_BOX_DIM:
            errors.append(
                f"{stem}:{line_no}: height={height:.6f} is degenerate "
                f"(must be > {_MIN_BOX_DIM})"
            )

    return errors


# ---------------------------------------------------------------------------
# Split validation
# ---------------------------------------------------------------------------


def validate_split(split_dir: Path, num_classes: int) -> ValidationReport:
    """Validate one dataset split (``train``, ``valid``, or ``test``).

    Parameters
    ----------
    split_dir:
        Path to the split directory containing ``images/`` and ``labels/``.
    num_classes:
        Total number of classes.

    Returns
    -------
    ValidationReport
        Per-split validation results.
    """
    report = ValidationReport()
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    # --- directory existence ---
    if not images_dir.is_dir():
        report.errors.append(f"Missing directory: {images_dir}")
        return report
    if not labels_dir.is_dir():
        report.errors.append(f"Missing directory: {labels_dir}")
        return report

    # --- collect stems ---
    image_stems: dict[str, Path] = {}
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() in _IMAGE_EXTENSIONS and img_path.is_file():
            image_stems[img_path.stem] = img_path

    label_stems: dict[str, Path] = {}
    for lbl_path in sorted(labels_dir.iterdir()):
        if lbl_path.suffix == ".txt" and lbl_path.is_file():
            label_stems[lbl_path.stem] = lbl_path

    report.total_images = len(image_stems)
    report.total_labels = len(label_stems)

    # --- images without labels ---
    for stem in sorted(image_stems):
        if stem not in label_stems:
            report.missing_labels.append(stem)

    # --- orphan labels (label without image) ---
    for stem in sorted(label_stems):
        if stem not in image_stems:
            report.orphan_labels.append(stem)

    # --- validate each label file ---
    for stem, lbl_path in sorted(label_stems.items()):
        file_errors = validate_label_file(lbl_path, num_classes)
        if file_errors:
            report.invalid_labels += 1
            report.errors.extend(file_errors)
            # Still count class_ids that are parseable for partial distribution
            _count_classes_partial(lbl_path, num_classes, report.class_distribution)
        else:
            report.valid_labels += 1
            _count_classes(lbl_path, report.class_distribution)

    return report


def _count_classes(label_path: Path, distribution: dict[int, int]) -> None:
    """Count class_ids from a validated label file into *distribution*."""
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 1:
            try:
                cid = int(parts[0])
                distribution[cid] = distribution.get(cid, 0) + 1
            except ValueError:
                pass


def _count_classes_partial(
    label_path: Path, num_classes: int, distribution: dict[int, int]
) -> None:
    """Best-effort class counting even for files with errors."""
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 1:
            try:
                cid = int(parts[0])
                if 0 <= cid <= num_classes - 1:
                    distribution[cid] = distribution.get(cid, 0) + 1
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Full dataset validation
# ---------------------------------------------------------------------------


def validate_dataset(config: dict[str, Any]) -> ValidationReport:
    """Validate the full converted dataset (train + valid + test splits).

    Parameters
    ----------
    config:
        Loaded configuration dict (from :func:`load_config`).

    Returns
    -------
    ValidationReport
        Combined report across all splits.
    """
    output_dataset: Path = config["output_dataset"]
    classes: list[str] = config["classes"]
    num_classes = len(classes)

    combined = ValidationReport()
    split_names = ("train", "valid", "test")

    for split_name in split_names:
        split_dir = output_dataset / split_name
        if not split_dir.is_dir():
            combined.errors.append(f"Split directory missing: {split_dir}")
            continue

        split_report = validate_split(split_dir, num_classes)

        combined.total_images += split_report.total_images
        combined.total_labels += split_report.total_labels
        combined.valid_labels += split_report.valid_labels
        combined.invalid_labels += split_report.invalid_labels
        combined.orphan_labels.extend(split_report.orphan_labels)
        combined.missing_labels.extend(split_report.missing_labels)
        combined.errors.extend(split_report.errors)

        for cid, count in split_report.class_distribution.items():
            combined.class_distribution[cid] = (
                combined.class_distribution.get(cid, 0) + count
            )

    print_validation_report(combined, classes)
    return combined


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_validation_report(
    report: ValidationReport, class_names: list[str]
) -> None:
    """Print a human-readable summary of the validation report."""
    status = "PASSED" if report.is_valid else "FAILED"
    print()
    print("=" * 60)
    print(f"  Dataset Validation Report — {status}")
    print("=" * 60)
    print(f"  Total images : {report.total_images}")
    print(f"  Total labels : {report.total_labels}")
    print(f"  Valid labels : {report.valid_labels}")
    print(f"  Invalid labels: {report.invalid_labels}")
    print(f"  Missing labels: {len(report.missing_labels)}")
    print(f"  Orphan labels : {len(report.orphan_labels)}")
    print()

    # --- class distribution ---
    if report.class_distribution:
        print("  Class distribution:")
        print(f"  {'ID':<5} {'Class':<15} {'Count':>8}")
        print(f"  {'-'*5} {'-'*15} {'-'*8}")
        for cid, name in enumerate(class_names):
            count = report.class_distribution.get(cid, 0)
            print(f"  {cid:<5} {name:<15} {count:>8}")
        print()

    # --- errors ---
    if report.errors:
        print(f"  Errors ({len(report.errors)}):")
        for err in report.errors[:20]:
            print(f"    - {err}")
        if len(report.errors) > 20:
            print(f"    ... and {len(report.errors) - 20} more")
        print()

    # --- missing / orphan ---
    if report.missing_labels:
        print(f"  Images without labels ({len(report.missing_labels)}):")
        for stem in report.missing_labels[:10]:
            print(f"    - {stem}")
        if len(report.missing_labels) > 10:
            print(f"    ... and {len(report.missing_labels) - 10} more")
        print()

    if report.orphan_labels:
        print(f"  Labels without images ({len(report.orphan_labels)}):")
        for stem in report.orphan_labels[:10]:
            print(f"    - {stem}")
        if len(report.orphan_labels) > 10:
            print(f"    ... and {len(report.orphan_labels) - 10} more")
        print()

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m src.training.validate_dataset --config ...``"""
    parser = argparse.ArgumentParser(
        description="Validate a converted YOLOv8 dataset."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file (e.g. configs/yolov8n-waste.yaml)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        config = load_config(args.config)
    except (FileNotFoundError, Exception) as exc:
        logger.error("Failed to load config: %s", exc)
        return 1

    report = validate_dataset(config)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
