# Spec: YOLOv8 Waste Detection Training Pipeline

End-to-end specification for converting a Roboflow image-classification dataset into YOLOv8 detection format, training a YOLOv8n model for urban solid waste detection, and exporting it for edge deployment on Jetson Nano. This spec covers the `yolov8-waste-detection` change in the ClearSky project.

## Quick path

1. Convert classification CSV → YOLO detection labels (auto-annotate bounding boxes)
2. Train YOLOv8n with MLflow experiment tracking
3. Export best model to ONNX and TensorRT
4. Evaluate mAP@0.5 and per-class AP on the held-out test set

---

## Overview

The ClearSky project needs to detect six categories of urban solid waste (cardboard, glass, metal, paper, plastic, trash) using a YOLOv8n model deployable on a Jetson Nano 4GB. The available dataset ("Waste Classification" from Roboflow) provides **image-level classification labels only** — one-hot vectors in `_classes.csv` — with **no bounding box annotations**. This spec defines how to bridge that gap: generate bounding boxes via zero-shot auto-annotation, build a reproducible training pipeline, and produce optimized inference artifacts.

---

## Requirements

### Functional Requirements

#### FR-01: Dataset Conversion

The system MUST convert the classification-format dataset into YOLO detection format.

**Input format** (per split directory):
- `_classes.csv` — header: `filename, cardboard, glass, metal, paper, plastic, trash`
- One row per image, one-hot encoded (exactly one `1`, rest `0`)
- Images are 640×640 JPEG files

**Output format** (YOLO detection):
- Directory structure: `datasets/waste-detect/{train,valid,test}/`
  - `images/` — symlinks or copies of original JPEGs
  - `labels/` — one `.txt` per image, same stem as the image
  - Each line: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0–1)
- `data.yaml` at dataset root with `train`, `val`, `test` paths and `names` list

**Class mapping** (fixed order, 0-indexed):

| ID | Class     |
|----|-----------|
| 0  | cardboard |
| 1  | glass     |
| 2  | metal     |
| 3  | paper     |
| 4  | plastic   |
| 5  | trash     |

**Bounding box generation strategy** (ordered by preference):

1. **Primary: Zero-shot auto-annotation with YOLO-World**
   - Load `yolo-world-l` (or `yolo-world-s` if VRAM-constrained) pretrained model
   - Set text prompts to the 6 class names: `["cardboard", "glass", "metal", "paper", "plastic", "trash"]`
   - Run inference on each image at 640×640
   - Filter detections by confidence ≥ 0.30
   - For each image, take the highest-confidence detection as the single bounding box (dataset is single-object per image)
   - Write YOLO-format `.txt` label file
   - Log rejection rate (images with zero detections above threshold)

2. **Fallback: Image-processing contour detection**
   - Convert to grayscale → adaptive threshold → morphological cleanup
   - Find largest contour by area → bounding rect
   - Assign class from CSV label
   - Use ONLY if zero-shot rejection rate exceeds 15%

3. **Quality gate**: If either method produces a rejection rate > 5% (images with no valid bbox), the conversion script MUST emit a warning report listing all rejected images and halt until the user decides how to proceed.

#### FR-02: Training Pipeline

The system MUST train a YOLOv8n detection model using the converted dataset.

**Configuration**:

| Parameter       | Value         | Rationale                                  |
|-----------------|---------------|--------------------------------------------|
| Model           | `yolov8n.pt`  | Nano — fits 4GB VRAM, targets edge deploy  |
| Task            | `detect`      | Bounding box detection (not segmentation)  |
| Epochs          | 100           | Sufficient for convergence on small data   |
| Batch size      | 16            | Tuned for RTX 3050 4GB (reduce to 8 if OOM)|
| Image size      | 640           | Matches source image dimensions            |
| Optimizer       | `AdamW`       | Stable for small datasets                  |
| Learning rate   | `0.001`       | Conservative for transfer learning         |
| LR schedule     | `cos_lr=True` | Cosine annealing                           |
| Patience        | 20            | Early stopping if val loss stagnates       |
| Workers         | 4             | Balanced for dataloader throughput         |
| Cache           | `ram`         | Dataset ~85MB fits in memory               |
| Pretrained      | `yolov8n.pt`  | COCO-pretrained weights from Ultralytics   |
| Project         | `runs/detect` | Local runs directory                       |
| Exist OK        | `True`        | Allow overwriting previous runs            |

**Data augmentation** (Ultralytics defaults + overrides):

| Augmentation | Value | Note                              |
|--------------|-------|-----------------------------------|
| `hsv_h`     | 0.015 | Low hue shift — waste colors vary |
| `hsv_s`     | 0.7   | Default saturation                |
| `hsv_v`     | 0.4   | Default value                     |
| `degrees`   | 10    | Slight rotation tolerance         |
| `translate` | 0.1   | Minor translation                 |
| `scale`     | 0.5   | Zoom variation                    |
| `flipud`    | 0.5   | Vertical flip — objects are upright-agnostic |
| `fliplr`    | 0.5   | Horizontal flip                   |
| `mosaic`    | 1.0   | Default mosaic                    |
| `mixup`     | 0.0   | Disabled — single-object images   |

#### FR-03: Model Export

The system MUST export the trained model to deployment formats.

**ONNX export**:
- Command: `yolo export model=best.pt format=onnx imgsz=640 half=True simplify=True`
- Target: Universal inference (CPU/GPU via ONNX Runtime)
- Output: `models/yolov8n-waste-best.onnx`

**TensorRT export** (for Jetson Nano):
- Must be performed ON the Jetson Nano (TensorRT engines are hardware-specific)
- Command: `yolo export model=best.pt format=engine imgsz=640 half=True device=0`
- JetPack 4.6.1 ships TensorRT 8.x — compatible with Ultralytics export
- Output: `models/yolov8n-waste-best.engine`
- Constraint: Jetson Nano has 4GB RAM — `half=True` (FP16) is mandatory

#### FR-04: Evaluation

The system MUST evaluate the trained model and produce metrics.

**Metrics to compute**:
- mAP@0.5 (target: ≥ 0.85)
- mAP@0.5:0.95 (target: ≥ 0.60)
- Per-class AP@0.5 for all 6 classes
- Precision and recall at IoU=0.5
- Inference latency (ms/image) on Jetson Nano

**Evaluation procedure**:
1. Run `yolo val model=best.pt data=data.yaml split=test` on the held-out test set
2. Parse results from `results.json` generated by Ultralytics
3. Generate per-class AP bar chart (saved as PNG)
4. Log all metrics to MLflow run

---

### Non-Functional Requirements

#### NFR-01: Reproducibility

- All paths MUST be relative to the project root or configurable via a YAML config file
- No hardcoded absolute paths in any script
- Random seed fixed to `42` for training (`seed=42` in `model.train()`)
- The conversion script MUST be deterministic given the same input and model version
- A `requirements.txt` or `pyproject.toml` MUST pin all dependency versions
- The dataset conversion script MUST log the auto-annotation model version used

#### NFR-02: Performance Constraints

| Constraint                | Target         | Environment   |
|---------------------------|----------------|---------------|
| Training time             | < 4 hours      | RTX 3050 4GB  |
| mAP@0.5                   | ≥ 0.85         | Validation    |
| Inference FPS             | ≥ 10           | Jetson Nano   |
| Inference latency         | < 100 ms/frame | Jetson Nano   |
| Model memory footprint    | < 3 GB         | Jetson Nano   |
| Exported model size       | < 20 MB        | ONNX (FP16)   |

#### NFR-03: Compatibility

| Component         | Version Constraint                                  |
|-------------------|-----------------------------------------------------|
| Python            | ≥ 3.8, < 3.11 (Jetson Nano JetPack 4.6.1 limit)    |
| PyTorch           | ≥ 2.0 (training); Jetson-compatible build (inference)|
| Ultralytics       | ≥ 8.0, < 8.3 (TensorRT 8.x compat on JetPack 4.6)  |
| CUDA              | 11.x (training); 10.2 (Jetson Nano)                 |
| TensorRT          | 8.x (JetPack 4.6.1)                                 |
| OpenCV            | ≥ 4.5                                               |
| MLflow            | ≥ 2.0                                               |

---

## Scenarios

### Dataset Conversion

#### SC-01: Convert classification CSV to YOLO detection format

**Given** a split directory containing `_classes.csv` and JPEG images
**When** the conversion script runs with the zero-shot auto-annotation strategy
**Then** for each row in `_classes.csv`:
- The script reads the image, runs the zero-shot model with the 6 class prompts
- The highest-confidence detection (≥ 0.30) is selected
- A `.txt` label file is written in YOLO format with normalized coordinates
- The image is copied/symlinked to the output `images/` directory
- The label file is written to `labels/` with the same stem as the image

#### SC-02: Validate generated bounding boxes

**Given** a completed conversion run
**When** the validation script runs on the output dataset
**Then** for every `.txt` label file:
- Each line has exactly 5 space-separated values
- `class_id` is an integer in [0, 5]
- `x_center`, `y_center`, `width`, `height` are floats in [0.0, 1.0]
- `width` and `height` are > 0.001 (no degenerate boxes)
- Every image in `images/` has a corresponding `.txt` in `labels/`
- A summary report is printed: total images, total labels, class distribution, rejected images

#### SC-03: Handle missing or invalid images

**Given** a row in `_classes.csv` referencing a filename that does not exist on disk
**When** the conversion script processes that row
**Then** the script logs a warning with the missing filename
**And** skips that row without halting
**And** includes the count of missing images in the final summary report

#### SC-04: Zero-shot model produces no detection above threshold

**Given** an image where the zero-shot model returns no detection with confidence ≥ 0.30
**When** the conversion script processes that image
**Then** the image is added to the rejected-images list
**And** no label file is written for it
**If** the total rejection rate exceeds 5%, the script halts with a report

### Training

#### SC-05: Configure and start training

**Given** a valid YOLO-format dataset with `data.yaml`
**When** the training script is invoked with the config parameters from FR-02
**Then** the model loads `yolov8n.pt` pretrained weights
**And** training begins with the specified hyperparameters
**And** an MLflow run is created with all hyperparameters logged as tags

#### SC-06: Track experiments with MLflow

**Given** an active training run
**When** each epoch completes
**Then** the following are logged to the MLflow run:
- Training loss (box, cls, dfl)
- Validation metrics (mAP@0.5, mAP@0.5:0.95, precision, recall)
- Learning rate
- Epoch duration
**And** the best model checkpoint path is logged as an MLflow artifact

#### SC-07: Early stopping and checkpoint saving

**Given** training is running with `patience=20`
**When** validation mAP@0.5 does not improve for 20 consecutive epochs
**Then** training stops early
**And** the best model checkpoint (`best.pt`) is preserved
**And** the last checkpoint (`last.pt`) is preserved
**And** the stopping epoch and best metric are logged to MLflow

### Export

#### SC-08: Export to ONNX

**Given** a trained `best.pt` model
**When** the export command runs with `format=onnx imgsz=640 half=True simplify=True`
**Then** an ONNX file is produced at `models/yolov8n-waste-best.onnx`
**And** the file size is < 20 MB
**And** a test inference on 10 random test images produces valid detections

#### SC-09: Export to TensorRT for Jetson

**Given** a trained `best.pt` model and a Jetson Nano with JetPack 4.6.1
**When** the export command runs on the Jetson with `format=engine imgsz=640 half=True`
**Then** a TensorRT engine file is produced at `models/yolov8n-waste-best.engine`
**And** the engine loads without errors on the Jetson
**And** a test inference produces detections consistent with the PyTorch model (±5% mAP)

### Evaluation

#### SC-10: Compute mAP metrics

**Given** a trained model and the test split
**When** `yolo val` runs on the test split
**Then** mAP@0.5, mAP@0.5:0.95, precision, and recall are computed
**And** results are saved to `results.json`
**And** all metrics are logged to the MLflow run

#### SC-11: Per-class AP analysis

**Given** evaluation results from SC-10
**When** the analysis script runs
**Then** a per-class AP@0.5 breakdown is printed as a table:

```
Class       AP@0.5    Precision   Recall    Support
cardboard   0.XX      0.XX        0.XX      N
glass       0.XX      0.XX        0.XX      N
metal       0.XX      0.XX        0.XX      N
paper       0.XX      0.XX        0.XX      N
plastic     0.XX      0.XX        0.XX      N
trash       0.XX      0.XX        0.XX      N
```

**And** a bar chart PNG is saved showing per-class AP

---

## Data Format Specification

### Input: Classification CSV

```
filename, cardboard, glass, metal, paper, plastic, trash
cardboard1_jpg.rf.abc123.jpg, 1, 0, 0, 0, 0, 0
glass42_jpg.rf.def456.jpg, 0, 1, 0, 0, 0, 0
```

- One-hot encoded: exactly one column is `1`, rest are `0`
- Filename matches the JPEG in the same directory
- Header has spaces after commas (must be stripped during parsing)

### Output: YOLO Detection Label

```
# File: cardboard1_jpg.rf.abc123.txt
0 0.487500 0.512500 0.750000 0.820000
```

- `<class_id>` — integer 0–5
- `<x_center>` — bounding box center X, normalized [0, 1]
- `<y_center>` — bounding box center Y, normalized [0, 1]
- `<width>` — bounding box width, normalized [0, 1]
- `<height>` — bounding box height, normalized [0, 1]
- Single line per image (one object per image in this dataset)

### Output: data.yaml

```yaml
path: /absolute/path/to/datasets/waste-detect
train: train/images
val: valid/images
test: test/images

names:
  0: cardboard
  1: glass
  2: metal
  3: paper
  4: plastic
  5: trash
```

---

## Configuration Specification

All configurable parameters MUST be exposed in a single YAML config file at `configs/yolov8n-waste.yaml`:

```yaml
# --- Paths ---
project_root: "."                          # Relative to this config file
source_data: "/home/santiago/Descargas/Waste Classification.v1i.multiclass"
output_dataset: "datasets/waste-detect"
models_dir: "models"
runs_dir: "runs/detect"

# --- Auto-annotation ---
auto_annotate:
  method: "yolo-world"                     # "yolo-world" | "contour"
  model: "yolo-world-l"                    # YOLO-World variant
  confidence_threshold: 0.30
  batch_size: 8                            # Inference batch for speed
  device: 0                                # GPU device

# --- Training ---
training:
  model: "yolov8n.pt"
  task: "detect"
  epochs: 100
  batch: 16
  imgsz: 640
  optimizer: "AdamW"
  lr0: 0.001
  cos_lr: true
  patience: 20
  workers: 4
  cache: "ram"
  seed: 42
  project: "${runs_dir}"
  name: "yolov8n-waste"
  exist_ok: true

# --- Augmentation ---
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 10.0
  translate: 0.1
  scale: 0.5
  flipud: 0.5
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.0

# --- Export ---
export:
  onnx:
    imgsz: 640
    half: true
    simplify: true
  tensorrt:
    imgsz: 640
    half: true

# --- MLflow ---
mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "yolov8-waste-detection"

# --- Classes ---
classes:
  - cardboard    # 0
  - glass        # 1
  - metal        # 2
  - paper        # 3
  - plastic      # 4
  - trash        # 5
```

---

## Dependencies and Constraints

### Hardware

| Role      | Device              | VRAM/RAM | Constraint                        |
|-----------|----------------------|----------|-----------------------------------|
| Training  | RTX 3050 (Laptop)   | 4 GB     | Batch ≤ 16 at 640px, may need 8   |
| Inference | Jetson Nano 4GB     | 4 GB     | JetPack 4.6.1, Python < 3.11      |

### Software Stack

| Package        | Training env | Jetson env       |
|----------------|--------------|------------------|
| Python         | 3.10         | 3.6–3.10         |
| PyTorch        | 2.0+ (CUDA 11.x) | 1.10 (Jetson build) |
| Ultralytics    | 8.0–8.2.x   | 8.0.x (pinned)   |
| CUDA           | 11.8         | 10.2             |
| TensorRT       | —            | 8.x (JetPack)    |
| OpenCV         | 4.8+         | 4.5+             |
| MLflow         | 2.0+         | — (training only)|
| YOLO-World     | 0.1+         | — (conversion only)|

### Key Constraints

1. **TensorRT engines are hardware-specific** — the `.engine` file MUST be generated on the target Jetson Nano, not on the training PC
2. **Python < 3.11 on Jetson** — JetPack 4.6.1 does not support Python 3.11+
3. **Ultralytics version pinning** — versions ≥ 8.3 may drop TensorRT 8.x support; pin to `< 8.3`
4. **Single-object images** — each image contains exactly one waste item; the auto-annotation pipeline should select only the top-1 detection per image
5. **Dataset is small** (~5300 train / ~500 val / ~250 test) — overfitting is a real risk; early stopping and augmentation are mandatory
6. **FP16 mandatory on Jetson** — FP32 will exceed 4GB memory budget during inference

---

## File Structure

```
ClearSkypy/
├── configs/
│   └── yolov8n-waste.yaml          # All configurable parameters
├── src/training/
│   ├── convert_dataset.py          # Classification CSV → YOLO detection
│   ├── validate_dataset.py         # Post-conversion validation
│   ├── train_detect.py             # Training pipeline (replaces train_seg.py)
│   ├── export_model.py             # ONNX + TensorRT export
│   └── evaluate.py                 # Evaluation + per-class AP
├── datasets/
│   └── waste-detect/               # Converted dataset (gitignored)
│       ├── data.yaml
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── valid/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
├── models/                          # Trained + exported models (gitignored)
│   ├── yolov8n-waste-best.pt
│   ├── yolov8n-waste-best.onnx
│   └── yolov8n-waste-best.engine
├── runs/
│   └── detect/
│       └── yolov8n-waste/          # Training artifacts
├── pretrained/
│   └── yolov8n.pt                  # COCO-pretrained base
└── mlflow.db                       # MLflow tracking database
```

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Zero-shot model produces poor bounding boxes for some classes (e.g., "trash" is ambiguous) | High | High | Validate bbox quality visually on a sample of 50 images per class before full conversion; lower threshold or switch to contour fallback if needed |
| "Trash" class is semantically vague — zero-shot model may not detect it consistently | High | Medium | Consider relabeling "trash" to "other-waste" in prompts; or use a more descriptive prompt like "mixed waste object" |
| TensorRT export fails on Jetson due to version mismatch | Medium | High | Pin Ultralytics to 8.0.x; test export early in the pipeline; have ONNX fallback |
| RTX 3050 OOM at batch=16 | Medium | Low | Reduce to batch=8 with gradient accumulation; or use `batch=-1` for Ultralytics auto-tuning |
| Auto-annotation introduces noisy labels that cap model accuracy | Medium | Medium | Spot-check 100 random images post-conversion; if >10% have visibly wrong boxes, re-annotate with tighter thresholds or manual correction |
| Dataset class imbalance skews training | Low | Medium | Check class distribution in CSV; if any class has <10% of samples, apply class-weighted loss or oversampling |
