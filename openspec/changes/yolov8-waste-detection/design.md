# Technical Design: YOLOv8 Waste Detection Training Pipeline

End-to-end architecture for converting a classification dataset into YOLOv8 detection format, training a YOLOv8n model, and exporting it for Jetson Nano deployment. This design translates the spec for the `yolov8-waste-detection` change into implementable modules, interfaces, and data flow.

## Quick path

1. **Convert**: `python src/training/convert_dataset.py --config configs/yolov8n-waste.yaml`
2. **Train**: `python src/training/train_detect.py --config configs/yolov8n-waste.yaml`
3. **Export**: `python src/training/export_model.py --config configs/yolov8n-waste.yaml --format onnx`
4. **Evaluate**: `python src/training/evaluate.py --config configs/yolov8n-waste.yaml`

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    configs/yolov8n-waste.yaml                │
│              (single source of truth for all modules)        │
└──────────────┬──────────────────────────────────┬────────────┘
               │                                  │
    ┌──────────▼──────────┐           ┌──────────▼──────────┐
    │  convert_dataset.py │           │  train_detect.py     │
    │                     │           │                      │
    │  CSV + Images       │           │  data.yaml + images  │
    │       │             │           │       │              │
    │  ┌────▼────┐        │           │  ┌────▼────┐        │
    │  │Annotator│(YOLO-W)│           │  │YOLOv8n  │        │
    │  │Fallback │(OpenCV)│           │  │train()  │        │
    │  └────┬────┘        │           │  └────┬────┘        │
    │       │             │           │       │              │
    │  YOLO labels/       │           │  best.pt + MLflow    │
    │  data.yaml          │           │                      │
    └──────────┬──────────┘           └──────────┬───────────┘
               │                                  │
    ┌──────────▼──────────┐           ┌──────────▼───────────┐
    │ validate_dataset.py │           │  export_model.py      │
    │ (quality gate)      │           │                      │
    └─────────────────────┘           │  best.pt              │
                                      │    │                  │
                                      │  ┌─▼──────────┐      │
                                      │  │ONNX / TRT  │      │
                                      │  └─┬──────────┘      │
                                      │    │                  │
                                      │  .onnx / .engine     │
                                      └──────────┬───────────┘
                                                 │
                                      ┌──────────▼───────────┐
                                      │  evaluate.py          │
                                      │                      │
                                      │  best.pt + test set   │
                                      │       │               │
                                      │  mAP, per-class AP    │
                                      │  bar chart PNG        │
                                      └──────────────────────┘
```

### Dependency graph (left → right means "depends on")

```
config_loader  ←  convert_dataset  ←  validate_dataset
config_loader  ←  train_detect
config_loader  ←  export_model
config_loader  ←  evaluate
```

Each script is independently runnable. No script imports another script. They share only the config loader and the output artifacts on disk.

---

## Module Architecture

### Module responsibilities

| Module | File | Responsibility | Depends on |
|--------|------|---------------|------------|
| Config loader | `src/training/config.py` | Load YAML, resolve paths, validate | `pyyaml` |
| Dataset converter | `src/training/convert_dataset.py` | CSV → YOLO labels with auto-annotation | `config`, `ultralytics`, `opencv` |
| Dataset validator | `src/training/validate_dataset.py` | Post-conversion quality gate | `config` |
| Training pipeline | `src/training/train_detect.py` | Train YOLOv8n with MLflow tracking | `config`, `ultralytics`, `mlflow` |
| Model exporter | `src/training/export_model.py` | Export to ONNX / TensorRT | `config`, `ultralytics` |
| Evaluator | `src/training/evaluate.py` | Compute metrics, per-class AP, charts | `config`, `ultralytics`, `matplotlib` |

### Design principles applied

- **Flat scripts, not packages**: Each module is a single file with a `main()` entry point. No `__init__.py` chains, no abstract base classes.
- **Config injection**: Every script takes `--config <path>` and loads the same YAML. No hardcoded paths.
- **Delegation over wrapping**: Training and export delegate to Ultralytics' `model.train()` and `model.export()`. We wrap only for config loading, MLflow setup, and post-processing.
- **Fail fast**: Each script validates its inputs before doing expensive work.

---

## Configuration System

### File: `src/training/config.py`

```python
from pathlib import Path
from typing import Any
import yaml

def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config and resolve all paths relative to project root.
    
    Resolution rules:
    - project_root is resolved relative to the config file's directory
    - All other paths (source_data, output_dataset, models_dir, runs_dir)
      are resolved relative to project_root
    - ${var} references in string values are interpolated from top-level keys
    """
    
def resolve_paths(config: dict) -> dict:
    """Resolve all path values to absolute Path objects.
    
    Mutates config in place. Raises ValueError if project_root doesn't exist.
    """

def validate_config(config: dict) -> None:
    """Validate required keys exist and values are in acceptable ranges.
    
    Raises ConfigError with a list of all validation failures (not just the first).
    Checks:
    - source_data directory exists
    - classes list has exactly 6 entries
    - training.batch > 0
    - auto_annotate.confidence_threshold in [0.0, 1.0]
    """
```

### Config schema (enforced by `validate_config`)

```
Required top-level keys:
  project_root: str
  source_data: str          # Must exist on disk
  output_dataset: str
  models_dir: str
  runs_dir: str
  auto_annotate: dict
    method: "yolo-world" | "contour"
    model: str
    confidence_threshold: float  [0.0, 1.0]
    batch_size: int              > 0
    device: int | str
  training: dict
    model: str
    epochs: int                  > 0
    batch: int                   > 0
    imgsz: int                   > 0
    seed: int
    ... (all Ultralytics train() params)
  augmentation: dict             (all float values)
  export: dict
    onnx: dict
    tensorrt: dict
  mlflow: dict
    tracking_uri: str
    experiment_name: str
  classes: list[str]             exactly 6 entries
```

### Path resolution example

Given `configs/yolov8n-waste.yaml` at `/home/santiago/Proyectos/python/ClearSkypy/configs/yolov8n-waste.yaml`:

```
project_root: "."                    → /home/santiago/Proyectos/python/ClearSkypy
source_data: "/home/santiago/..."    → absolute, used as-is
output_dataset: "datasets/waste-detect" → {project_root}/datasets/waste-detect
models_dir: "models"                 → {project_root}/models
runs_dir: "runs/detect"              → {project_root}/runs/detect
```

---

## Module 1: Dataset Conversion (`convert_dataset.py`)

This is the most complex module. It bridges the gap between classification labels (one-hot CSV) and detection labels (bounding boxes).

### Data flow

```
_classes.csv + JPEG images
        │
        ▼
┌───────────────────┐
│  Parse CSV        │  Read filename + one-hot class label
│  (pandas/csv)     │  Map one-hot → class_id (0-5)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐     ┌─────────────────────┐
│  Auto-annotate    │────▶│  YOLO-World model    │
│  (per image)      │     │  set_classes(6 names)│
│                   │     │  predict(img)        │
│                   │     │  filter conf ≥ 0.30  │
│                   │     │  take top-1 det      │
└────────┬──────────┘     └─────────────────────┘
         │
         │  (if rejection rate > 15%)
         ▼
┌───────────────────┐     ┌─────────────────────┐
│  Contour fallback │────▶│  OpenCV pipeline     │
│  (per image)      │     │  gray → threshold    │
│                   │     │  → morph cleanup     │
│                   │     │  → largest contour   │
│                   │     │  → bounding rect     │
└────────┬──────────┘     └─────────────────────┘
         │
         ▼
┌───────────────────┐
│  Write YOLO label │  {class_id} {xc} {yc} {w} {h}
│  + copy/symlink   │  normalized to [0, 1]
│    image          │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Quality gate     │  If rejection > 5%: halt + report
│  + write data.yaml│
└───────────────────┘
```

### Function signatures

```python
# --- Entry point ---
def convert_dataset(config: dict) -> ConversionReport:
    """Main conversion pipeline.
    
    1. Parse all split CSVs (train/valid/test)
    2. Run auto-annotation on each image
    3. Write YOLO labels + copy images
    4. Generate data.yaml
    5. Run quality gate
    Returns: ConversionReport with stats and rejected images list.
    """

# --- CSV parsing ---
def parse_split_csv(csv_path: Path) -> list[ImageLabel]:
    """Parse _classes.csv into list of ImageLabel records.
    
    ImageLabel = dataclass(filename: str, class_id: int, class_name: str)
    
    Handles: header with spaces after commas, one-hot encoding validation.
    Raises ValueError if a row doesn't have exactly one '1'.
    """

# --- Auto-annotation interface ---
class BoundingBox:
    """Simple dataclass: x_center, y_center, width, height (all normalized 0-1)."""

class Annotator:
    """Protocol/interface for bounding box generation."""
    def annotate(self, image_path: Path) -> BoundingBox | None:
        """Return bounding box or None if no detection found."""
    def annotate_batch(self, image_paths: list[Path]) -> list[BoundingBox | None]:
        """Batch version for efficiency. Default: loop over annotate()."""

# --- YOLO-World annotator ---
class YOLOWorldAnnotator(Annotator):
    """Zero-shot auto-annotation using YOLO-World.
    
    Initialization:
    - Load model: YOLO("yolo-world-l.pt") — downloads if not cached
    - Set text prompts: model.set_classes(class_names)
    - Model stays loaded for the full conversion run
    
    Per-image:
    - Run model.predict(img, conf=threshold, verbose=False)
    - Filter results[0].boxes by confidence >= threshold
    - If detections exist: take the one with highest confidence
    - Convert xyxy → xywh normalized → return BoundingBox
    - If no detections: return None
    """
    def __init__(self, model_name: str, class_names: list[str], 
                 confidence_threshold: float, device: int | str): ...

# --- Contour fallback annotator ---
class ContourAnnotator(Annotator):
    """Image-processing fallback for when YOLO-World fails.
    
    Pipeline per image:
    1. cv2.imread(path)
    2. cv2.cvtColor(img, BGR2GRAY)
    3. cv2.adaptiveThreshold(gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, 
       THRESH_BINARY_INV, blockSize=11, C=2)
    4. cv2.morphologyEx(thresh, MORPH_CLOSE, kernel=5x5)  # fill gaps
    5. cv2.findContours(contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    6. Take contour with largest area
    7. cv2.boundingRect(contour) → x, y, w, h
    8. Normalize to [0, 1] using image dimensions
    9. Return BoundingBox
    
    Returns None only if no contours found (shouldn't happen on waste images).
    """

# --- YOLO label writing ---
def write_yolo_label(label_path: Path, bbox: BoundingBox, class_id: int) -> None:
    """Write a single-line YOLO label file.
    
    Format: "{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
    """

def write_data_yaml(output_dir: Path, config: dict) -> None:
    """Generate data.yaml at the dataset root.
    
    Writes:
    - path: absolute path to output_dir
    - train: train/images
    - val: valid/images
    - test: test/images
    - names: {0: cardboard, 1: glass, ...}
    """

# --- Quality gate ---
class ConversionReport:
    """Tracks conversion statistics.
    
    Fields:
    - total_images: int
    - successful: int
    - rejected: list[str]  (filenames with no valid bbox)
    - rejection_rate: float
    - missing_images: list[str]  (CSV references file not on disk)
    - class_distribution: dict[str, int]
    - annotator_model: str  (e.g. "yolo-world-l" or "contour-opencv")
    """

def quality_gate(report: ConversionReport, max_rejection_rate: float = 0.05) -> None:
    """Check rejection rate. If > max_rejection_rate:
    - Print warning report with all rejected filenames
    - Print class distribution
    - Raise QualityGateError (halts the pipeline)
    """
```

### YOLO-World integration details

**Model loading**:

```python
from ultralytics import YOLO

# YOLO-World is supported natively in Ultralytics >= 8.1
# The model file is downloaded automatically on first use
model = YOLO("yolo-world-l.pt")  # or "yolo-world-s.pt" for lower VRAM

# Set open-vocabulary class prompts
model.set_classes(["cardboard", "glass", "metal", "paper", "plastic", "trash"])

# Run inference
results = model.predict(image_path, conf=0.30, imgsz=640, device=0, verbose=False)
```

**Fallback strategy if YOLO-World is unavailable**:

If the `yolo-world-l.pt` model fails to download or load (network issues, version incompatibility), the design supports two alternatives:

1. **Grounding DINO** (via `groundingdino` package) — another zero-shot detector. Same interface, different backend.
2. **Contour-only mode** — set `auto_annotate.method: "contour"` in config to skip YOLO-World entirely.

The `Annotator` interface makes this a config change, not a code change.

**Batch inference for speed**:

YOLO-World supports batched inference. Process images in batches of `auto_annotate.batch_size` (default 8) to reduce per-image overhead:

```python
def annotate_batch(self, image_paths: list[Path]) -> list[BoundingBox | None]:
    results = self.model.predict(
        [cv2.imread(str(p)) for p in image_paths],
        conf=self.confidence_threshold,
        imgsz=640,
        device=self.device,
        verbose=False,
    )
    return [self._extract_top_bbox(r) for r in results]
```

### Error handling strategy

| Error | Behavior | Rationale |
|-------|----------|-----------|
| Image file missing (CSV references nonexistent file) | Log warning, skip row, count in report | Don't halt for individual missing files |
| YOLO-World model download fails | Raise with clear message + suggest `method: "contour"` fallback | Fail fast, actionable message |
| YOLO-World returns no detections for an image | Return None, add to rejected list | Expected behavior for ambiguous images |
| Contour fallback finds no contours | Log warning, add to rejected list | Shouldn't happen on waste images |
| Rejection rate > 5% | Print full report, raise QualityGateError | Human must decide how to proceed |
| CSV has invalid one-hot encoding | Raise ValueError with row number | Data integrity — don't silently skip |

### Conversion output structure

```
datasets/waste-detect/
├── data.yaml
├── train/
│   ├── images/          # JPEG files (copied from source)
│   └── labels/          # One .txt per image, same stem
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Images: copy vs symlink** — Use copies (not symlinks) for portability. The dataset is ~85MB total, so disk space is not a concern. Symlinks break if the source directory moves.

---

## Module 2: Dataset Validation (`validate_dataset.py`)

### Purpose

Post-conversion quality check. Can be run standalone or as part of the conversion pipeline.

### Function signatures

```python
def validate_dataset(dataset_dir: Path) -> ValidationReport:
    """Validate a YOLO detection dataset.
    
    Checks:
    1. Every image in images/ has a corresponding .txt in labels/
    2. Every label file has valid format:
       - Exactly 5 space-separated values per line
       - class_id is int in [0, num_classes-1]
       - x_center, y_center, width, height are floats in [0.0, 1.0]
       - width > 0.001 and height > 0.001 (no degenerate boxes)
    3. data.yaml exists and has required keys
    4. Class distribution is logged
    
    Returns: ValidationReport with pass/fail status and details.
    """

def print_validation_report(report: ValidationReport) -> None:
    """Print human-readable summary table.
    
    Output:
    - Total images / labels
    - Class distribution (count per class)
    - Invalid label files (if any)
    - Missing label files (if any)
    - PASS / FAIL verdict
    """
```

---

## Module 3: Training Pipeline (`train_detect.py`)

### Data flow

```
config YAML
    │
    ▼
┌──────────────────┐
│ Load config      │
│ Setup MLflow     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────┐
│ Load YOLOv8n.pt  │────▶│ Ultralytics YOLO()  │
│ model.to("cuda") │     │ downloads if needed  │
└────────┬─────────┘     └─────────────────────┘
         │
         ▼
┌──────────────────┐
│ model.train(     │
│   data=data.yaml │
│   **training_cfg │
│   **aug_cfg      │
│ )                │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Copy best.pt     │
│ → models/        │
│ Log to MLflow    │
└──────────────────┘
```

### Function signatures

```python
def setup_mlflow(config: dict) -> None:
    """Configure MLflow tracking.
    
    - Set tracking URI from config (sqlite:///mlflow.db)
    - Set experiment name
    - Start a new run with hyperparameters as tags
    """

def build_train_args(config: dict) -> dict:
    """Merge training + augmentation config into Ultralytics train() kwargs.
    
    Combines:
    - config["training"] → model, epochs, batch, imgsz, optimizer, lr0, etc.
    - config["augmentation"] → hsv_h, hsv_s, degrees, flipud, etc.
    - Resolved data.yaml path
    - Resolved project/name paths
    
    Returns: dict ready to pass as **kwargs to model.train()
    """

def train(config: dict) -> Path:
    """Main training entry point.
    
    1. setup_mlflow(config)
    2. model = YOLO(config["training"]["model"])
    3. train_args = build_train_args(config)
    4. model.train(**train_args)
    5. Copy best.pt to models_dir
    6. Log best.pt as MLflow artifact
    7. Return path to best.pt
    """

def copy_best_model(run_dir: Path, models_dir: Path, run_name: str) -> Path:
    """Copy best.pt from training run to models/ directory.
    
    Source: {run_dir}/{run_name}/weights/best.pt
    Dest:   {models_dir}/yolov8n-waste-best.pt
    
    Returns: destination path
    """
```

### MLflow integration

MLflow logging happens at two levels:

1. **Ultralytics built-in**: Ultralytics has native MLflow support. When `settings.update({"mlflow": True})` is set, it automatically logs metrics per epoch. We enable this.

2. **Manual logging**: We additionally log:
   - All hyperparameters as MLflow params (via `mlflow.log_params()`)
   - Config file hash (for reproducibility)
   - Auto-annotation model version (if conversion was run)
   - Best model checkpoint as artifact

```python
def setup_mlflow(config: dict) -> str:
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    run = mlflow.start_run(run_name=config["training"]["name"])
    
    # Log all hyperparameters
    mlflow.log_params({
        "model": config["training"]["model"],
        "epochs": config["training"]["epochs"],
        "batch": config["training"]["batch"],
        "imgsz": config["training"]["imgsz"],
        "optimizer": config["training"]["optimizer"],
        "lr0": config["training"]["lr0"],
        "seed": config["training"]["seed"],
        # ... all training + augmentation params
    })
    
    return run.info.run_id
```

### Checkpoint management

After training completes (normally or via early stopping):

```
runs/detect/yolov8n-waste/
├── weights/
│   ├── best.pt          ← highest val mAP (copied to models/)
│   └── last.pt          ← final epoch
├── results.csv          ← per-epoch metrics
├── results.png          ← training curves
├── confusion_matrix.png
├── F1_curve.png
├── PR_curve.png
└── args.yaml            ← actual training args used
```

The `train()` function copies `best.pt` to `models/yolov8n-waste-best.pt` for easy access by export and evaluation scripts.

---

## Module 4: Model Export (`export_model.py`)

### Function signatures

```python
def export_onnx(model_path: Path, output_dir: Path, config: dict) -> Path:
    """Export PyTorch model to ONNX format.
    
    1. Load model: YOLO(model_path)
    2. Export: model.export(format="onnx", imgsz=640, half=True, simplify=True)
    3. Move output to: {output_dir}/yolov8n-waste-best.onnx
    4. Validate: run inference on 10 random test images
    5. Log file size (must be < 20 MB)
    6. Return output path
    """

def export_tensorrt(model_path: Path, output_dir: Path, config: dict) -> Path:
    """Export PyTorch model to TensorRT engine.
    
    CONSTRAINT: Must be run ON the Jetson Nano (TRT engines are hardware-specific).
    
    1. Load model: YOLO(model_path)
    2. Export: model.export(format="engine", imgsz=640, half=True, device=0)
    3. Move output to: {output_dir}/yolov8n-waste-best.engine
    4. Validate: run inference on 10 random test images
    5. Return output path
    """

def validate_export(model_path: Path, test_images_dir: Path, num_samples: int = 10) -> bool:
    """Run test inference on sample images to verify the exported model works.
    
    1. Load exported model (ONNX or engine)
    2. Pick `num_samples` random images from test set
    3. Run inference on each
    4. Verify: results have valid boxes (not empty, not errors)
    5. Return True if all pass, False otherwise
    """

def main():
    """CLI entry point.
    
    Usage:
        python export_model.py --config configs/yolov8n-waste.yaml --format onnx
        python export_model.py --config configs/yolov8n-waste.yaml --format tensorrt
        python export_model.py --config configs/yolov8n-waste.yaml --format all
    """
```

### Export flow detail

```
best.pt
    │
    ├── format=onnx ──────▶ model.export(format="onnx", imgsz=640, 
    │                       half=True, simplify=True)
    │                       → yolov8n-waste-best.onnx (< 20 MB)
    │                       → validate with 10 test images
    │
    └── format=engine ────▶ model.export(format="engine", imgsz=640,
                            half=True, device=0)
                            → yolov8n-waste-best.engine
                            → MUST run on Jetson Nano
                            → validate with 10 test images
```

---

## Module 5: Evaluation (`evaluate.py`)

### Function signatures

```python
def evaluate_model(model_path: Path, data_yaml: Path, config: dict) -> dict:
    """Run evaluation on the test split.
    
    1. Load model: YOLO(model_path)
    2. Run: results = model.val(data=data_yaml, split="test")
    3. Parse results into a metrics dict:
       {
         "mAP50": float,
         "mAP50_95": float,
         "precision": float,
         "recall": float,
         "per_class": {
           "cardboard": {"ap50": float, "precision": float, "recall": float, "support": int},
           "glass": {...},
           ...
         }
       }
    4. Return metrics dict
    """

def print_metrics_table(metrics: dict, class_names: list[str]) -> None:
    """Print formatted per-class AP table.
    
    Output:
    Class       AP@0.5    Precision   Recall    Support
    cardboard   0.XX      0.XX        0.XX      N
    glass       0.XX      0.XX        0.XX      N
    ...
    ------------------------------------------------
    mAP@0.5:    0.XX
    mAP@0.5:0.95: 0.XX
    """

def plot_per_class_ap(metrics: dict, class_names: list[str], output_path: Path) -> None:
    """Generate per-class AP bar chart.
    
    - matplotlib horizontal bar chart
    - One bar per class, colored by AP value (green > 0.85, yellow > 0.70, red < 0.70)
    - Target line at 0.85
    - Saved as PNG at output_path
    """

def log_metrics_mlflow(metrics: dict) -> None:
    """Log all evaluation metrics to the active MLflow run.
    
    - mAP50, mAP50_95, precision, recall as top-level metrics
    - Per-class AP as nested metrics: "ap50/cardboard", "ap50/glass", etc.
    """

def main():
    """CLI entry point.
    
    Usage:
        python evaluate.py --config configs/yolov8n-waste.yaml
        python evaluate.py --config configs/yolov8n-waste.yaml --model models/custom.pt
    """
```

### Visualization output

```
Per-Class AP@0.5 — YOLOv8n Waste Detection
┌────────────────────────────────────────────┐
│  cardboard  ████████████████████░░  0.92   │
│  glass      ███████████████░░░░░░░  0.78   │
│  metal      ██████████████████░░░░  0.88   │
│  paper      ███████████████████░░░  0.90   │
│  plastic    ████████████████░░░░░░  0.82   │
│  trash      ████████████░░░░░░░░░░  0.65   │
│                                    ── 0.85  │  ← target
└────────────────────────────────────────────┘
```

Saved as `runs/detect/yolov8n-waste/per_class_ap.png`.

---

## File Structure

### New files to create

```
ClearSkypy/
├── configs/
│   └── yolov8n-waste.yaml              # All configurable parameters
├── src/training/
│   ├── config.py                       # Config loader + validator (NEW)
│   ├── convert_dataset.py              # Classification CSV → YOLO detection (NEW)
│   ├── validate_dataset.py             # Post-conversion validation (NEW)
│   ├── train_detect.py                 # Training pipeline (NEW, replaces train_seg.py)
│   ├── export_model.py                 # ONNX + TensorRT export (NEW)
│   └── evaluate.py                     # Evaluation + per-class AP (NEW)
├── tests/
│   └── training/
│       ├── test_config.py              # Config loading/validation tests
│       ├── test_convert_dataset.py     # Conversion unit tests
│       ├── test_validate_dataset.py    # Validation unit tests
│       └── fixtures/
│           ├── sample_classes.csv      # 5-row test CSV
│           └── sample_images/          # 3 tiny JPEG files (64x64)
```

### Existing files — action plan

| File | Action | Rationale |
|------|--------|-----------|
| `src/training/train/train_seg.py` | **Deprecate** (add deprecation comment, keep file) | Replaced by `train_detect.py`. Keep for reference until new pipeline is proven. |
| `src/training/train/main.py` | **Deprecate** (add deprecation comment) | Old training script. Replaced by `train_detect.py`. |
| `src/training/train/test_model.py` | **Keep as-is** | Webcam/file inference testing — still useful, independent of training pipeline. |

### Generated artifacts (gitignored)

```
datasets/waste-detect/          # Converted dataset
models/                         # Trained + exported models
runs/detect/                    # Training run artifacts
mlflow.db                       # MLflow tracking database
```

---

## Testing Strategy

### Unit tests

| Module | What to test | Mock strategy |
|--------|-------------|---------------|
| `config.py` | YAML loading, path resolution, validation errors | No mocks — use temp YAML files |
| `convert_dataset.py` → `parse_split_csv` | CSV parsing, one-hot validation, class mapping | No mocks — use fixture CSV |
| `convert_dataset.py` → `YOLOWorldAnnotator` | BBox extraction from model results | Mock `YOLO.predict()` to return fixed detections |
| `convert_dataset.py` → `ContourAnnotator` | Contour detection on synthetic images | Create synthetic test images (white rect on black bg) |
| `convert_dataset.py` → `write_yolo_label` | Label file format, normalization | No mocks — write to temp dir, read back |
| `convert_dataset.py` → `quality_gate` | Rejection rate threshold, report generation | No mocks — construct ConversionReport directly |
| `validate_dataset.py` | Label format validation, missing file detection | Create temp dataset structure |
| `train_detect.py` → `build_train_args` | Config merging, path resolution | No mocks — pure function |
| `evaluate.py` → `print_metrics_table` | Table formatting | No mocks — construct metrics dict |
| `evaluate.py` → `plot_per_class_ap` | Chart generation | No mocks — verify PNG is created |

### Integration tests

| Test | What it verifies |
|------|-----------------|
| End-to-end conversion | CSV + real images → valid YOLO dataset (using contour annotator, no model download needed) |
| Validation on converted dataset | validate_dataset passes on output of convert_dataset |
| Training smoke test | 1 epoch on tiny dataset (10 images) completes without error |
| Export smoke test | ONNX export of a tiny trained model succeeds |

### Test fixtures

```
tests/training/fixtures/
├── sample_classes.csv          # 5 rows: 2 cardboard, 1 glass, 1 metal, 1 paper
├── sample_images/
│   ├── img001.jpg              # 64x64 solid color (cardboard)
│   ├── img002.jpg              # 64x64 solid color (glass)
│   └── img003.jpg              # 64x64 with white rectangle (for contour test)
└── conftest.py                 # Shared pytest fixtures (tmp directories, config dicts)
```

### Mock strategy for YOLO-World

The YOLO-World model is the hardest to test. Strategy:

```python
# In test_convert_dataset.py
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_yolo_world():
    """Mock YOLO-World model that returns predictable detections."""
    mock_model = MagicMock()
    
    # Create a fake result with one detection
    fake_box = MagicMock()
    fake_box.conf = [0.85]
    fake_box.xywhn = [[0.5, 0.5, 0.8, 0.8]]  # normalized xywh
    fake_box.cls = [0]  # cardboard
    
    fake_result = MagicMock()
    fake_result.boxes = fake_box
    
    mock_model.predict.return_value = [fake_result]
    return mock_model

def test_yolo_world_annotator_extracts_top_bbox(mock_yolo_world):
    annotator = YOLOWorldAnnotator.__new__(YOLOWorldAnnotator)
    annotator.model = mock_yolo_world
    annotator.confidence_threshold = 0.30
    
    bbox = annotator.annotate(Path("fake_image.jpg"))
    assert bbox is not None
    assert bbox.x_center == 0.5
    assert bbox.width == 0.8
```

---

## Dependencies

### Python packages (for `requirements.txt` or `pyproject.toml`)

```
ultralytics>=8.1,<8.3          # YOLOv8 + YOLO-World support
opencv-python>=4.8              # Image processing (contour fallback)
mlflow>=2.0                     # Experiment tracking
pyyaml>=6.0                     # Config loading
matplotlib>=3.7                 # Per-class AP charts
pandas>=2.0                     # CSV parsing (optional — csv module works too)
onnx>=1.14                      # ONNX validation (installed by ultralytics)
onnxruntime-gpu>=1.16           # ONNX inference validation (optional)

# Testing
pytest>=7.0
pytest-cov>=4.0
```

### Version constraints rationale

| Package | Constraint | Why |
|---------|-----------|-----|
| `ultralytics` | `>=8.1,<8.3` | 8.1+ has YOLO-World support. <8.3 for TensorRT 8.x compat on JetPack 4.6.1 |
| `opencv-python` | `>=4.8` | Training env. Jetson uses 4.5+ (separate install) |
| Python | `>=3.8,<3.11` | JetPack 4.6.1 limit |

---

## Risks and Mitigations

| Risk | Decision in this design | Open question |
|------|------------------------|---------------|
| YOLO-World model download fails or is unavailable | `Annotator` interface allows swapping to contour-only via config. No code change needed. | Should we pre-download the model and cache it in `pretrained/`? |
| "Trash" class is semantically vague for zero-shot detection | The prompt is the exact class name. If rejection rate is high for "trash", the prompt can be changed to "mixed waste object" in config without code changes. | What prompt variant works best for "trash"? Needs experimentation. |
| TensorRT export fails on Jetson | ONNX export is the primary path. TensorRT is a secondary step done on-device. The export script validates after export. | Do we need a Jetson-specific config variant? |
| RTX 3050 OOM at batch=16 | Config makes batch size trivially adjustable. Can also use `batch=-1` for Ultralytics auto-tuning. | Should we default to `batch=-1` for safety? |
| Auto-annotation noise caps model accuracy | Quality gate halts if rejection > 5%. Post-conversion validation checks label format. Visual spot-check is recommended but not automated. | Should we add a visual spot-check script that overlays bboxes on images? |
| Ultralytics version drift breaks pipeline | Pin to `<8.3` in requirements. Config isolates all hyperparameters. | What's the migration path when we need to upgrade? |

---

## Checklist

- [ ] `configs/yolov8n-waste.yaml` created with all parameters from spec
- [ ] `src/training/config.py` loads and validates config
- [ ] `src/training/convert_dataset.py` converts CSV → YOLO labels
- [ ] `src/training/validate_dataset.py` validates converted dataset
- [ ] `src/training/train_detect.py` trains YOLOv8n with MLflow
- [ ] `src/training/export_model.py` exports to ONNX and TensorRT
- [ ] `src/training/evaluate.py` computes metrics and generates charts
- [ ] `tests/training/` has unit tests for config, conversion, validation
- [ ] Deprecation comments added to `train_seg.py` and `main.py`
- [ ] All paths resolved relative to project root — no hardcoded absolutes
- [ ] `requirements.txt` pins all dependency versions
