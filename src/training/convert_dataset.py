"""Dataset conversion core — CSV parsing, YOLO label writing, quality gate.

Converts a classification-format dataset (one-hot CSV + images) into YOLO
detection format (normalized bounding-box labels + images).  Provides the
``Annotator`` protocol so T03 (YOLO-World) and T04 (contour) can plug in
their own bounding-box generators.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from src.training.config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundingBox:
    """Normalized bounding box in YOLO format (all values in [0, 1])."""

    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class ImageLabel:
    """A single row from ``_classes.csv`` after parsing."""

    filename: str
    class_id: int
    class_name: str


@dataclass
class ConversionReport:
    """Tracks conversion statistics across all splits."""

    total_images: int = 0
    successful: int = 0
    rejected: list[str] = field(default_factory=list)
    missing_images: list[str] = field(default_factory=list)
    class_distribution: dict[str, int] = field(default_factory=dict)
    annotator_model: str = "unknown"

    @property
    def rejection_rate(self) -> float:
        """Fraction of images that were rejected (no valid bbox)."""
        if self.total_images == 0:
            return 0.0
        return len(self.rejected) / self.total_images


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class QualityGateError(Exception):
    """Raised when the conversion rejection rate exceeds the threshold."""

    def __init__(self, report: ConversionReport, max_rate: float) -> None:
        self.report = report
        self.max_rate = max_rate
        super().__init__(self._format())

    def _format(self) -> str:
        r = self.report
        lines = [
            f"Quality gate FAILED: rejection rate {r.rejection_rate:.1%} "
            f"exceeds threshold {self.max_rate:.1%}",
            f"  Total images : {r.total_images}",
            f"  Successful   : {r.successful}",
            f"  Rejected     : {len(r.rejected)}",
            f"  Missing      : {len(r.missing_images)}",
            f"  Annotator    : {r.annotator_model}",
        ]
        if r.rejected:
            lines.append("  Rejected files:")
            for name in r.rejected[:20]:
                lines.append(f"    - {name}")
            if len(r.rejected) > 20:
                lines.append(f"    ... and {len(r.rejected) - 20} more")
        if r.class_distribution:
            lines.append("  Class distribution:")
            for cls, count in sorted(r.class_distribution.items()):
                lines.append(f"    {cls}: {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Annotator protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Annotator(Protocol):
    """Interface for bounding-box generators.

    Implementations must provide ``annotate``; ``annotate_batch`` has a
    default that loops over single-image calls.
    """

    def annotate(self, image_path: Path) -> BoundingBox | None:
        """Return a bounding box for the image, or ``None`` if no detection."""
        ...

    def annotate_batch(
        self, image_paths: list[Path]
    ) -> list[BoundingBox | None]:
        """Batch version — default loops over ``annotate``."""
        return [self.annotate(p) for p in image_paths]


# ---------------------------------------------------------------------------
# ContourAnnotator — OpenCV-based fallback (T04)
# ---------------------------------------------------------------------------


class ContourAnnotator:
    """Fallback annotator that finds the largest object via contour detection.

    Pipeline: load image → grayscale → Gaussian blur → adaptive threshold →
    morphological cleanup → find external contours → select largest by area →
    compute bounding rect → normalize to :class:`BoundingBox`.

    Parameters
    ----------
    min_area_ratio:
        Minimum contour area as a fraction of total image area.  Contours
        smaller than this threshold are ignored.  Defaults to ``0.01`` (1 %).
    """

    def __init__(self, min_area_ratio: float = 0.01) -> None:
        self.min_area_ratio = min_area_ratio

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _preprocess(img: "np.ndarray") -> "np.ndarray":
        """Grayscale → blur → adaptive threshold → morphological cleanup."""
        import cv2

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        return cleaned

    def _find_largest_contour(
        self, img: "np.ndarray"
    ) -> "tuple[int, int, int, int] | None":
        """Return ``(x, y, w, h)`` of the largest valid contour, or *None*."""
        import cv2

        cleaned = self._preprocess(img)
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        img_area = img.shape[0] * img.shape[1]
        valid = [
            c
            for c in contours
            if cv2.contourArea(c) / img_area >= self.min_area_ratio
        ]
        if not valid:
            return None

        largest = max(valid, key=cv2.contourArea)
        return cv2.boundingRect(largest)

    @staticmethod
    def _rect_to_bbox(
        rect: "tuple[int, int, int, int]", img_h: int, img_w: int
    ) -> BoundingBox:
        """Convert ``(x, y, w, h)`` to a normalized :class:`BoundingBox`."""
        x, y, w, h = rect
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        bbox_w = w / img_w
        bbox_h = h / img_h
        return BoundingBox(x_center, y_center, bbox_w, bbox_h)

    # -- public API ---------------------------------------------------------

    def annotate(self, image_path: Path) -> BoundingBox | None:
        """Return a bounding box for the main object, or ``None``."""
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            return None

        rect = self._find_largest_contour(img)
        if rect is None:
            return None

        img_h, img_w = img.shape[:2]
        return self._rect_to_bbox(rect, img_h, img_w)

    def annotate_batch(
        self, image_paths: list[Path]
    ) -> list[BoundingBox | None]:
        """Process images sequentially (OpenCV is already fast per-image)."""
        return [self.annotate(p) for p in image_paths]


# ---------------------------------------------------------------------------
# CenterCropAnnotator — placeholder for pipeline testing
# ---------------------------------------------------------------------------


class CenterCropAnnotator:
    """Placeholder annotator: assumes the object occupies the center 70%.

    Useful for testing the conversion pipeline without requiring YOLO-World
    or OpenCV.  Will be replaced by real annotators in T03/T04.
    """

    def annotate(self, image_path: Path) -> BoundingBox:
        return BoundingBox(x_center=0.5, y_center=0.5, width=0.7, height=0.7)

    def annotate_batch(
        self, image_paths: list[Path]
    ) -> list[BoundingBox]:
        return [self.annotate(p) for p in image_paths]


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def parse_split_csv(
    csv_path: Path, classes: list[str] | None = None
) -> list[ImageLabel]:
    """Parse a ``_classes.csv`` file into a list of ``ImageLabel`` records.

    Parameters
    ----------
    csv_path:
        Path to the ``_classes.csv`` file.
    classes:
        Ordered list of class names.  When *None* the class names are
        derived from the CSV header (columns after ``filename``).

    Returns
    -------
    list[ImageLabel]
        One entry per valid row.

    Raises
    ------
    ValueError
        If a row has an invalid one-hot encoding (zero or more than one ``1``).
    """
    if classes is None:
        classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    class_to_id = {name: idx for idx, name in enumerate(classes)}
    results: list[ImageLabel] = []

    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        raw_header = next(reader)
        # Strip whitespace from header columns
        header = [col.strip() for col in raw_header]

        # The first column is "filename"; the rest are class names
        class_columns = header[1:]

        for line_no, row in enumerate(reader, start=2):
            if not row or all(cell.strip() == "" for cell in row):
                continue

            filename = row[0].strip()
            # Parse one-hot values
            values = [int(cell.strip()) for cell in row[1:]]

            ones_count = sum(values)
            if ones_count != 1:
                raise ValueError(
                    f"Row {line_no} ({filename}): expected exactly one '1' "
                    f"in one-hot encoding, got {ones_count}"
                )

            one_idx = values.index(1)
            class_name = class_columns[one_idx]

            if class_name not in class_to_id:
                raise ValueError(
                    f"Row {line_no} ({filename}): unknown class '{class_name}'. "
                    f"Expected one of {classes}"
                )

            results.append(
                ImageLabel(
                    filename=filename,
                    class_id=class_to_id[class_name],
                    class_name=class_name,
                )
            )

    return results


# ---------------------------------------------------------------------------
# YOLO label writing
# ---------------------------------------------------------------------------


def write_yolo_label(
    label_path: Path, bbox: BoundingBox, class_id: int
) -> None:
    """Write a single-line YOLO label file.

    Format: ``"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\\n"``
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"{class_id} {bbox.x_center:.6f} {bbox.y_center:.6f} "
        f"{bbox.width:.6f} {bbox.height:.6f}\n"
    )
    label_path.write_text(line, encoding="utf-8")


# ---------------------------------------------------------------------------
# data.yaml generation
# ---------------------------------------------------------------------------


def write_data_yaml(output_dir: Path, config: dict[str, Any]) -> None:
    """Generate ``data.yaml`` at the dataset root.

    Writes absolute ``path`` and relative ``train``/``val``/``test`` image
    directories, plus the class ``names`` mapping.
    """
    output_dir = Path(output_dir).resolve()
    classes = config.get("classes", [])

    data: dict[str, Any] = {
        "path": str(output_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {idx: name for idx, name in enumerate(classes)},
    }

    yaml_path = output_dir / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


def quality_gate(
    report: ConversionReport, max_rejection_rate: float = 0.05
) -> None:
    """Check the rejection rate and halt if it exceeds the threshold.

    Raises ``QualityGateError`` with a full report when the gate fails.
    """
    if report.rejection_rate > max_rejection_rate:
        raise QualityGateError(report, max_rejection_rate)


# ---------------------------------------------------------------------------
# Split conversion
# ---------------------------------------------------------------------------


def _convert_split(
    split_dir: Path,
    output_dir: Path,
    annotator: Annotator,
    classes: list[str],
    report: ConversionReport,
) -> None:
    """Convert a single split (train / valid / test).

    Parses the CSV, calls the annotator for each image, writes labels, and
    copies images into the output directory.  Updates *report* in place.
    """
    csv_path = split_dir / "_classes.csv"
    if not csv_path.is_file():
        logger.warning("No _classes.csv found in %s — skipping split", split_dir)
        return

    labels = parse_split_csv(csv_path, classes)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for entry in labels:
        report.total_images += 1
        src_image = split_dir / entry.filename

        if not src_image.is_file():
            logger.warning("Missing image: %s", src_image)
            report.missing_images.append(entry.filename)
            report.rejected.append(entry.filename)
            continue

        bbox = annotator.annotate(src_image)

        if bbox is None:
            report.rejected.append(entry.filename)
            continue

        # Write label
        label_stem = Path(entry.filename).stem
        label_path = labels_dir / f"{label_stem}.txt"
        write_yolo_label(label_path, bbox, entry.class_id)

        # Copy image
        dst_image = images_dir / entry.filename
        shutil.copy2(src_image, dst_image)

        report.successful += 1
        report.class_distribution[entry.class_name] = (
            report.class_distribution.get(entry.class_name, 0) + 1
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def convert_dataset(config: dict[str, Any]) -> ConversionReport:
    """Main conversion pipeline.

    1. Create output directories.
    2. Build an annotator from config (``CenterCropAnnotator`` for now).
    3. Convert train / valid / test splits.
    4. Write ``data.yaml``.
    5. Run quality gate.

    Returns the ``ConversionReport``.
    """
    source_data = Path(config["source_data"])
    output_dataset = Path(config["output_dataset"])
    classes: list[str] = config.get("classes", [])

    # Build annotator — placeholder for now; T03/T04 will add real ones
    aa_cfg = config.get("auto_annotate", {})
    method = aa_cfg.get("method", "center-crop")

    if method == "center-crop":
        annotator: Annotator = CenterCropAnnotator()
        annotator_name = "center-crop"
    elif method == "contour":
        min_area = aa_cfg.get("min_area_ratio", 0.01)
        annotator = ContourAnnotator(min_area_ratio=min_area)
        annotator_name = "contour"
    else:
        # Real annotators (yolo-world) are implemented in T03.
        # For now, fall back to center-crop with a warning.
        logger.warning(
            "Annotator method '%s' not yet implemented — using CenterCropAnnotator",
            method,
        )
        annotator = CenterCropAnnotator()
        annotator_name = f"center-crop-fallback({method})"

    report = ConversionReport(annotator_model=annotator_name)

    # Create output structure
    output_dataset.mkdir(parents=True, exist_ok=True)

    # Process each split
    split_map = {"train": "train", "valid": "valid", "test": "test"}
    for split_name, output_name in split_map.items():
        split_dir = source_data / split_name
        split_output = output_dataset / output_name
        if split_dir.is_dir():
            _convert_split(split_dir, split_output, annotator, classes, report)
        else:
            logger.warning("Split directory not found: %s", split_dir)

    # Write data.yaml
    write_data_yaml(output_dataset, config)

    # Quality gate
    quality_gate(report)

    # Print summary
    _print_report(report)

    return report


def _print_report(report: ConversionReport) -> None:
    """Print a human-readable conversion summary."""
    print("\n" + "=" * 60)
    print("  Dataset Conversion Report")
    print("=" * 60)
    print(f"  Annotator      : {report.annotator_model}")
    print(f"  Total images   : {report.total_images}")
    print(f"  Successful     : {report.successful}")
    print(f"  Rejected       : {len(report.rejected)} ({report.rejection_rate:.1%})")
    print(f"  Missing images : {len(report.missing_images)}")
    if report.class_distribution:
        print("  Class distribution:")
        for cls, count in sorted(report.class_distribution.items()):
            print(f"    {cls:12s}: {count}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: ``python -m src.training.convert_dataset --config <path>``."""
    parser = argparse.ArgumentParser(
        description="Convert classification CSV dataset to YOLO detection format."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file (e.g. configs/yolov8n-waste.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    try:
        convert_dataset(config)
    except QualityGateError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
