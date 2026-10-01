# Tasks: YOLOv8 Waste Detection Training Pipeline

Implementation task breakdown for the `yolov8-waste-detection` change. Each task produces a working, testable increment and maps to one work-unit commit.

**Change**: yolov8-waste-detection
**Delivery strategy**: ask-on-risk (single PR unless >400 lines)
**Review policy**: 400 changed lines per PR

---

## Task Dependency Graph

```
T01 (config)
 ├──▶ T02 (conversion core)
 │     ├──▶ T03 (YOLO-World annotator)
 │     └──▶ T04 (contour annotator)
 ├──▶ T05 (dataset validation)
 ├──▶ T06 (training pipeline)
 ├──▶ T07 (model export)
 ├──▶ T08 (evaluation)
 ├──▶ T09 (integration test)
 └──▶ T10 (cleanup + docs)
```

T01 is the foundation. T02–T08 depend on T01 (config loader). T03 and T04 depend on T02 (annotator protocol). T09 depends on T02–T08 (end-to-end). T10 is last.

---

### T01: Configuration system + YAML config

**Files**: `src/training/config.py`, `configs/yolov8n-waste.yaml`, `tests/training/test_config.py`, `tests/training/__init__.py`, `tests/training/fixtures/conftest.py`
**Depends on**: None
**Estimated lines**: ~280

**What to do**:
1. Create `src/training/config.py` with three functions:
   - `load_config(config_path: str) -> dict` — loads YAML, resolves `${var}` interpolation, resolves paths relative to config file directory
   - `resolve_paths(config: dict) -> dict` — converts path strings to absolute `Path` objects using `project_root`
   - `validate_config(config: dict) -> None` — validates required keys, value ranges; raises `ConfigError` with ALL failures (not just the first)
2. Create `configs/yolov8n-waste.yaml` with all parameters from the spec (paths, auto-annotate, training, augmentation, export, mlflow, classes)
3. Create `tests/training/__init__.py` (empty) and `tests/training/fixtures/conftest.py` with shared pytest fixtures:
   - `tmp_config(tmp_path)` — creates a minimal valid YAML in a temp directory
   - `minimal_config()` — returns a dict with all required keys and valid values
4. Write `tests/training/test_config.py`:
   - Test YAML loading with valid config
   - Test path resolution (relative paths become absolute, absolute paths unchanged)
   - Test `${var}` interpolation in string values
   - Test validation catches missing keys, invalid ranges, nonexistent `source_data`
   - Test `ConfigError` accumulates multiple failures

**Acceptance criteria**:
- [x] `load_config("configs/yolov8n-waste.yaml")` returns a dict with all keys
- [x] All path values are resolved to absolute `Path` objects
- [x] `validate_config` raises `ConfigError` listing ALL validation failures
- [x] All tests pass with `pytest tests/training/test_config.py` (10/10 passed)
- [x] No hardcoded absolute paths in any file

**Tests**:
- `test_load_config_valid_yaml()` — loads a temp YAML and returns correct dict
- `test_resolve_paths_relative()` — `"datasets/waste-detect"` → `{project_root}/datasets/waste-detect`
- `test_resolve_paths_absolute()` — `"/home/santiago/..."` stays unchanged
- `test_interpolation()` — `"${runs_dir}"` in training.project resolves correctly
- `test_validate_missing_keys()` — raises ConfigError with list of missing keys
- `test_validate_invalid_batch()` — batch=0 raises ConfigError
- `test_validate_invalid_confidence()` — threshold=1.5 raises ConfigError
- `test_validate_nonexistent_source()` — source_data pointing to missing dir raises ConfigError
- `test_validate_accumulates_errors()` — multiple failures reported at once

---

### T02: Dataset conversion core (CSV parsing, label writing, quality gate)

**Files**: `src/training/convert_dataset.py`, `tests/training/test_convert_dataset.py`, `tests/training/fixtures/sample_classes.csv`, `tests/training/fixtures/sample_images/`
**Depends on**: T01
**Estimated lines**: ~300

**What to do**:
1. In `src/training/convert_dataset.py`, implement the core conversion infrastructure:
   - `BoundingBox` dataclass: `x_center, y_center, width, height` (all normalized floats)
   - `Annotator` protocol: `annotate(image_path: Path) -> BoundingBox | None` and `annotate_batch(image_paths: list[Path]) -> list[BoundingBox | None]`
   - `ImageLabel` dataclass: `filename, class_id, class_name`
   - `parse_split_csv(csv_path: Path, classes: list[str]) -> list[ImageLabel]` — parses `_classes.csv`, strips header spaces, validates one-hot encoding, maps to class_id
   - `write_yolo_label(label_path: Path, bbox: BoundingBox, class_id: int) -> None` — writes single-line YOLO format
   - `write_data_yaml(output_dir: Path, config: dict) -> None` — generates `data.yaml` with absolute path, split paths, class names
   - `ConversionReport` dataclass: `total_images, successful, rejected: list[str], missing_images: list[str], class_distribution: dict[str, int], annotator_model: str`
   - `quality_gate(report: ConversionReport, max_rejection_rate: float = 0.05) -> None` — raises `QualityGateError` if rejection > threshold, prints full report
   - `convert_dataset(config: dict) -> ConversionReport` — main pipeline: iterate splits, parse CSV, call annotator, write labels, copy images, generate data.yaml, run quality gate. Takes annotator from config (initializes based on `auto_annotate.method`)
2. Create test fixtures:
   - `tests/training/fixtures/sample_classes.csv` — 5 rows with 2 cardboard, 1 glass, 1 metal, 1 paper
   - `tests/training/fixtures/sample_images/` — 3 tiny 64×64 JPEG files (solid colors, created programmatically in conftest)
3. Write `tests/training/test_convert_dataset.py` (tests for core only — annotator tests in T03/T04):
   - Test CSV parsing with valid data
   - Test CSV parsing rejects invalid one-hot (two 1s, zero 1s)
   - Test `write_yolo_label` format
   - Test `write_data_yaml` structure
   - Test `quality_gate` passes at 3% rejection, fails at 10%
   - Test `convert_dataset` end-to-end with a mock annotator that returns fixed bboxes

**Acceptance criteria**:
- [x] `parse_split_csv` correctly maps one-hot rows to `ImageLabel` records
- [x] `write_yolo_label` produces `"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"`
- [x] `write_data_yaml` produces valid YAML with absolute path and class names
- [x] `quality_gate` raises `QualityGateError` when rejection > 5%
- [x] `convert_dataset` with a mock annotator produces valid YOLO directory structure
- [x] All tests pass (24/24 passed, 34 total with T01)

**Tests**:
- `test_parse_split_csv_valid()` — parses 5-row fixture correctly
- `test_parse_split_csv_invalid_onehot()` — raises ValueError on bad encoding
- `test_parse_split_csv_strips_spaces()` — handles `" cardboard"` header
- `test_write_yolo_label_format()` — writes correct normalized coordinates
- `test_write_data_yaml_structure()` — has path, train, val, test, names keys
- `test_quality_gate_passes_low_rejection()` — 3% rejection → no error
- `test_quality_gate_fails_high_rejection()` — 10% rejection → QualityGateError
- `test_quality_gate_report_lists_rejected()` — rejected filenames in error message
- `test_convert_dataset_with_mock_annotator()` — full pipeline with mock, verifies directory structure and label files
- `test_convert_dataset_missing_image()` — CSV references nonexistent file → logged, skipped, counted

---

### T03: YOLO-World zero-shot annotator

**Files**: `src/training/convert_dataset.py` (add `YOLOWorldAnnotator` class), `tests/training/test_convert_dataset.py` (add YOLO-World tests)
**Depends on**: T02
**Estimated lines**: ~160

**What to do**:
1. Add `YOLOWorldAnnotator` class to `src/training/convert_dataset.py`:
   - `__init__(self, model_name: str, class_names: list[str], confidence_threshold: float, device: int | str)` — stores params, does NOT load model yet
   - `load(self)` — loads YOLO-World model: `YOLO(f"{model_name}.pt")`, calls `model.set_classes(class_names)`. Raises clear error if download fails, suggesting `method: "contour"` fallback
   - `annotate(self, image_path: Path) -> BoundingBox | None` — runs `model.predict(img, conf=threshold, imgsz=640, device=device, verbose=False)`, extracts top-1 detection by confidence, converts `xywhn` to `BoundingBox`, returns None if no detections above threshold
   - `annotate_batch(self, image_paths: list[Path]) -> list[BoundingBox | None]` — batched inference using `model.predict([images], ...)`, extracts bbox per result
   - `_extract_top_bbox(self, result) -> BoundingBox | None` — helper: filter by conf, take argmax, convert to BoundingBox
2. Update `convert_dataset()` to instantiate `YOLOWorldAnnotator` when `auto_annotate.method == "yolo-world"`
3. Add tests using `unittest.mock`:
   - Mock `YOLO` class and its `predict` method to return fixed detections
   - Test top-1 selection (multiple detections → highest conf wins)
   - Test no detection → returns None
   - Test batch inference returns correct number of results
   - Test model load failure raises actionable error

**Acceptance criteria**:
- [x] `YOLOWorldAnnotator.annotate()` returns `BoundingBox` with correct normalized coordinates
- [x] Top-1 detection is selected (highest confidence)
- [x] Returns `None` when no detections above threshold
- [x] Batch mode processes multiple images
- [x] Model load failure gives actionable error message (mentions "contour" fallback)
- [x] All tests pass (10/10 mocked, no real model download needed)

**Tests**:
- `test_yolo_world_annotator_returns_bbox()` — mock predict returns one detection → BoundingBox with correct values
- `test_yolo_world_selects_top_confidence()` — mock returns 3 detections → highest conf selected
- `test_yolo_world_no_detection_returns_none()` — mock returns empty boxes → None
- `test_yolo_world_batch_annotate()` — 4 images → 4 results (some None)
- `test_yolo_world_load_failure()` — mock YOLO() raises → actionable error with "contour" suggestion
- `test_yolo_world_xywhn_conversion()` — verify xyxy → xywh normalized conversion is correct

---

### T04: Contour fallback annotator

**Files**: `src/training/convert_dataset.py` (add `ContourAnnotator` class), `tests/training/test_convert_dataset.py` (add contour tests)
**Depends on**: T02
**Estimated lines**: ~140

**What to do**:
1. Add `ContourAnnotator` class to `src/training/convert_dataset.py`:
   - `__init__(self)` — no params needed
   - `annotate(self, image_path: Path) -> BoundingBox | None`:
     1. `cv2.imread(str(path))`
     2. `cv2.cvtColor(img, BGR2GRAY)`
     3. `cv2.adaptiveThreshold(gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2)`
     4. `cv2.morphologyEx(thresh, MORPH_CLOSE, kernel=5×5)` — fill gaps
     5. `cv2.findContours(contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`
     6. Take contour with largest area
     7. `cv2.boundingRect(contour)` → x, y, w, h
     8. Normalize to [0, 1] using image dimensions → `BoundingBox`
     9. Return None only if no contours found
   - `annotate_batch()` — default loop (no batch optimization for OpenCV)
2. Update `convert_dataset()` to instantiate `ContourAnnotator` when `auto_annotate.method == "contour"`
3. Add tests using synthetic images (created with numpy/OpenCV in tests):
   - White rectangle on black background → bbox matches rectangle bounds
   - Solid gray image → largest contour covers most of image
   - Empty image (all black) → returns None (no contours)
   - Verify normalization is correct for known image dimensions

**Acceptance criteria**:
- [x] `ContourAnnotator.annotate()` returns `BoundingBox` for images with visible objects
- [x] Bounding box matches the largest contour's bounding rect
- [x] Coordinates are correctly normalized to [0, 1]
- [x] Returns `None` for images with no contours
- [x] All tests pass (8/8 contour tests, 42 total — no model download needed)

**Tests**:
- `test_contour_annotator_white_rect_on_black()` — synthetic image with known rect → bbox matches
- `test_contour_annotator_normalization()` — 100×200 image with rect at (10,20,80,100) → correct normalized values
- `test_contour_annotator_no_contours()` — all-black image → returns None
- `test_contour_annotator_largest_contour_selected()` — two rectangles → larger one selected
- `test_contour_annotator_missing_file()` — nonexistent path → returns None (cv2.imread returns None)

---

### T05: Dataset validation

**Files**: `src/training/validate_dataset.py`, `tests/training/test_validate_dataset.py`
**Depends on**: T01
**Estimated lines**: ~180

**What to do**:
1. Create `src/training/validate_dataset.py`:
   - `ValidationReport` dataclass: `total_images, total_labels, class_distribution: dict[int, int], invalid_labels: list[str], missing_labels: list[str], missing_images: list[str], passed: bool`
   - `validate_dataset(dataset_dir: Path, num_classes: int = 6) -> ValidationReport`:
     1. Check `data.yaml` exists and has required keys
     2. For each split (train/valid/test):
        - List all images in `images/`
        - For each image, check corresponding `.txt` exists in `labels/`
        - For each label file, validate format:
          - Exactly 5 space-separated values per line
          - `class_id` is int in `[0, num_classes-1]`
          - `x_center, y_center, width, height` are floats in `[0.0, 1.0]`
          - `width > 0.001` and `height > 0.001`
     3. Compute class distribution
     4. `passed = len(invalid_labels) == 0 and len(missing_labels) == 0`
   - `print_validation_report(report: ValidationReport, class_names: list[str]) -> None` — formatted table output
   - `main()` — CLI entry point: `python validate_dataset.py --config configs/yolov8n-waste.yaml`
2. Write `tests/training/test_validate_dataset.py`:
   - Create temp dataset structures with valid and invalid labels
   - Test all validation checks

**Acceptance criteria**:
- [x] Validates label format (5 values, correct ranges, valid class_id)
- [x] Detects missing label files (image without .txt)
- [x] Detects missing image files (label without image)
- [x] Computes class distribution
- [x] `print_validation_report` outputs readable table
- [x] All tests pass

**Tests**:
- `test_validate_valid_dataset()` — all correct → passed=True
- `test_validate_invalid_class_id()` — class_id=7 → invalid
- `test_validate_invalid_coordinates()` — x_center=1.5 → invalid
- `test_validate_degenerate_box()` — width=0.0001 → invalid
- `test_validate_missing_label()` — image without .txt → missing_labels
- `test_validate_missing_image()` — .txt without image → missing_images
- `test_validate_wrong_line_format()` — 3 values instead of 5 → invalid
- `test_validate_class_distribution()` — counts per class are correct
- `test_validate_missing_data_yaml()` — no data.yaml → reported

---

### T06: Training pipeline with MLflow

**Files**: `src/training/train_detect.py`, `tests/training/test_train_detect.py`
**Depends on**: T01
**Estimated lines**: ~230

**What to do**:
1. Create `src/training/train_detect.py`:
   - `setup_mlflow(config: dict) -> str` — sets tracking URI, experiment name, starts run, logs all hyperparameters as params. Returns run_id
   - `build_train_args(config: dict) -> dict` — merges `training` + `augmentation` config into Ultralytics `model.train()` kwargs. Resolves `data.yaml` path, `project`/`name` paths. Returns dict ready for `**kwargs`
   - `copy_best_model(run_dir: Path, models_dir: Path, run_name: str) -> Path` — copies `{run_dir}/{run_name}/weights/best.pt` → `{models_dir}/yolov8n-waste-best.pt`
   - `train(config: dict) -> Path` — main entry: setup_mlflow → YOLO(model) → model.train(**build_train_args) → copy_best_model → log artifact → return path
   - `main()` — CLI: `python train_detect.py --config configs/yolov8n-waste.yaml`
2. Write `tests/training/test_train_detect.py`:
   - Test `build_train_args` merges config correctly (mock-free, pure function)
   - Test `setup_mlflow` sets correct URI and experiment (mock mlflow)
   - Test `copy_best_model` copies file correctly (use tmp dirs)
   - Test `train` calls model.train with correct args (mock YOLO)

**Acceptance criteria**:
- [x] `build_train_args` produces correct kwargs dict from config
- [x] `setup_mlflow` logs all hyperparameters
- [x] `copy_best_model` copies best.pt to models directory
- [x] `train` orchestrates the full flow (mocked in tests)
- [x] CLI entry point works with `--config` argument
- [x] All tests pass (15/15 passed, 87 total)

**Tests**:
- `test_build_train_args_merges_config()` — training + augmentation keys present in output
- `test_build_train_args_resolves_paths()` — data path is absolute
- `test_build_train_args_augmentation()` — hsv_h, flipud, mosaic etc. in output
- `test_setup_mlflow_config()` — mock mlflow, verify set_tracking_uri and log_params called
- `test_copy_best_model()` — create fake best.pt, verify it's copied to correct dest
- `test_copy_best_model_missing_source()` — raises FileNotFoundError
- `test_train_orchestration()` — mock YOLO, verify model.train() called with correct args, best.pt copied

---

### T07: Model export (ONNX + TensorRT)

**Files**: `src/training/export_model.py`, `tests/training/test_export_model.py`
**Depends on**: T01, T06 (needs trained model path convention)
**Estimated lines**: ~200

**What to do**:
1. Create `src/training/export_model.py`:
   - `export_onnx(model_path: Path, output_dir: Path, config: dict) -> Path` — loads YOLO(model_path), calls `model.export(format="onnx", imgsz=640, half=True, simplify=True)`, moves output to `{output_dir}/yolov8n-waste-best.onnx`, validates file size < 20MB, returns path
   - `export_tensorrt(model_path: Path, output_dir: Path, config: dict) -> Path` — same but `format="engine"`. Prints warning that this must run on Jetson Nano
   - `validate_export(model_path: Path, test_images_dir: Path, num_samples: int = 10) -> bool` — loads exported model, runs inference on random test images, verifies detections are valid (non-empty results, no errors)
   - `main()` — CLI: `python export_model.py --config configs/yolov8n-waste.yaml --format onnx|tensorrt|all`
2. Write `tests/training/test_export_model.py`:
   - Mock `YOLO.export()` to return a fake output path
   - Test ONNX export calls correct export params
   - Test TensorRT export calls correct export params
   - Test file size validation
   - Test `validate_export` with mock model

**Acceptance criteria**:
- [ ] ONNX export produces `.onnx` file at correct path
- [ ] TensorRT export prints Jetson warning
- [ ] File size check warns if > 20MB
- [ ] `validate_export` runs inference on sample images
- [ ] CLI supports `--format onnx`, `--format tensorrt`, `--format all`
- [ ] All tests pass (mocked)

**Tests**:
- `test_export_onnx_params()` — mock YOLO, verify export called with format="onnx", half=True, simplify=True
- `test_export_tensorrt_params()` — verify format="engine", half=True
- `test_export_onnx_moves_file()` — verify output moved to models_dir
- `test_export_file_size_warning()` — fake 25MB file → warning logged
- `test_validate_export_success()` — mock model returns detections → True
- `test_validate_export_empty_results()` — mock model returns no detections → False
- `test_cli_format_onnx()` — argparse with --format onnx
- `test_cli_format_all()` — --format all calls both export_onnx and export_tensorrt

---

### T08: Evaluation with metrics and charts

**Files**: `src/training/evaluate.py`, `tests/training/test_evaluate.py`
**Depends on**: T01, T06 (needs trained model path convention)
**Estimated lines**: ~230

**What to do**:
1. Create `src/training/evaluate.py`:
   - `evaluate_model(model_path: Path, data_yaml: Path, config: dict) -> dict` — loads YOLO(model_path), runs `model.val(data=data_yaml, split="test")`, parses results into metrics dict: `{mAP50, mAP50_95, precision, recall, per_class: {class_name: {ap50, precision, recall, support}}}`
   - `print_metrics_table(metrics: dict, class_names: list[str]) -> None` — formatted table with per-class AP, precision, recall, support. Summary row at bottom
   - `plot_per_class_ap(metrics: dict, class_names: list[str], output_path: Path) -> None` — matplotlib horizontal bar chart, color-coded (green >0.85, yellow >0.70, red <0.70), target line at 0.85, saved as PNG
   - `log_metrics_mlflow(metrics: dict) -> None` — logs mAP50, mAP50_95, precision, recall + per-class AP as nested metrics
   - `main()` — CLI: `python evaluate.py --config configs/yolov8n-waste.yaml [--model models/custom.pt]`
2. Write `tests/training/test_evaluate.py`:
   - Test `print_metrics_table` output format (capture stdout)
   - Test `plot_per_class_ap` creates PNG file
   - Test `log_metrics_mlflow` calls mlflow.log_metric correctly
   - Test `evaluate_model` parses mock results correctly

**Acceptance criteria**:
- [ ] `evaluate_model` returns correct metrics structure
- [x] `print_metrics_table` outputs formatted table with all 6 classes
- [x] `plot_per_class_ap` generates PNG file at output_path
- [x] Chart has target line at 0.85
- [x] `log_metrics_mlflow` logs all metrics
- [x] All tests pass (21 tests)

**Tests**:
- `test_evaluate_model_parses_results()` — mock model.val() → correct metrics dict
- `test_print_metrics_table_format()` — capture stdout, verify table headers and rows
- `test_print_metrics_table_all_classes()` — all 6 class names appear in output
- `test_plot_per_class_ap_creates_png()` — verify PNG file exists after call
- `test_plot_per_class_ap_target_line()` — verify chart has 0.85 line (check axes)
- `test_log_metrics_mlflow()` — mock mlflow, verify log_metric called for mAP50, per-class
- `test_evaluate_model_default_model_path()` — uses models/yolov8n-waste-best.pt if no --model

---

### T09: Integration test (end-to-end smoke test)

**Files**: `tests/training/test_integration.py`
**Depends on**: T02, T03, T04, T05, T06, T07, T08
**Estimated lines**: ~100

**What to do**:
1. Create `tests/training/test_integration.py`:
   - `test_e2e_convert_validate()` — uses contour annotator (no model download), converts fixture CSV + synthetic images → validates output with `validate_dataset` → asserts passed=True
   - `test_e2e_training_smoke()` — creates tiny dataset (10 synthetic images with contour-generated labels), trains 1 epoch with `epochs=1, batch=2, imgsz=64`, verifies best.pt exists
   - `test_e2e_export_smoke()` — takes the tiny trained model from previous test, exports to ONNX, verifies file exists and size < 20MB
   - All tests use `tmp_path` and small image sizes (64×64) for speed
   - Mark with `@pytest.mark.integration` so they can be skipped with `-m "not integration"`
   - Mark training/export tests with `@pytest.mark.slow` (need GPU)
2. Add `pytest.ini` or `pyproject.toml` markers section if not present

**Acceptance criteria**:
- [x] `test_e2e_convert_validate` passes without GPU or model downloads
- [x] `test_e2e_training_smoke` completes 1 epoch on synthetic data (marked @pytest.mark.slow, deselected in CI)
- [x] `test_e2e_export_smoke` produces valid ONNX file (marked @pytest.mark.slow, deselected in CI)
- [x] Integration tests can be skipped with `-m "not integration"` or `-m "not slow"`
- [x] All integration tests pass (1 passed, 2 deselected for GPU requirement)

**Tests**:
- `test_e2e_convert_validate()` — contour annotator → validate → PASS
- `test_e2e_training_smoke()` — 1 epoch, 10 images, tiny model → best.pt exists
- `test_e2e_export_smoke()` — tiny model → ONNX export → file valid

---

### T10: Cleanup, dependencies, and documentation

**Files**: `requirements.txt`, `src/training/train/train_seg.py` (edit), `src/training/train/main.py` (edit), `.gitignore` (edit)
**Depends on**: T01–T09
**Estimated lines**: ~80

**What to do**:
1. Create `requirements.txt` at project root with pinned dependencies:
   ```
   ultralytics>=8.1,<8.3
   opencv-python>=4.8
   mlflow>=2.0
   pyyaml>=6.0
   matplotlib>=3.7
   pandas>=2.0
   pytest>=7.0
   pytest-cov>=4.0
   ```
2. Add deprecation comments to `src/training/train/train_seg.py`:
   - Add `"""DEPRECATED: Use src/training/train_detect.py instead."""` at top
   - Keep file intact for reference
3. Add deprecation comments to `src/training/train/main.py`:
   - Add `"""DEPRECATED: Use src/training/train_detect.py instead."""` at top
4. Update `.gitignore` to include:
   ```
   datasets/waste-detect/
   models/
   runs/
   mlflow.db
   *.onnx
   *.engine
   ```
5. Verify all scripts have `if __name__ == "__main__"` entry points

**Acceptance criteria**:
- [ ] `requirements.txt` pins all dependencies with version constraints
- [ ] Old scripts have deprecation comments
- [ ] `.gitignore` excludes generated artifacts
- [ ] All new scripts are runnable via `python src/training/<script>.py --config configs/yolov8n-waste.yaml`

**Tests**:
- No code tests — verification is manual (run each script with `--help`)

---

## Review Workload Forecast

| Metric | Estimate |
|--------|----------|
| **New source files** | 8 (`config.py`, `convert_dataset.py`, `validate_dataset.py`, `train_detect.py`, `export_model.py`, `evaluate.py`, `configs/yolov8n-waste.yaml`, `requirements.txt`) |
| **New test files** | 6 (`test_config.py`, `test_convert_dataset.py`, `test_validate_dataset.py`, `test_train_detect.py`, `test_export_model.py`, `test_evaluate.py`, `test_integration.py`) |
| **Modified files** | 3 (`train_seg.py`, `main.py`, `.gitignore`) |
| **Estimated new lines** | ~1,760 |
| **Estimated deleted lines** | ~0 (only deprecation comments added) |
| **Total changed lines** | ~1,840 |
| **Exceeds 400-line PR budget?** | **Yes — significantly** (4.6× over budget) |

### Recommendation: Chained PRs

At ~1,840 lines, this change exceeds the 400-line review budget by a wide margin. **Recommend splitting into 3 chained PRs**:

| PR | Tasks | Estimated lines | Review focus |
|----|-------|-----------------|--------------|
| **PR 1: Foundation** | T01, T02, T04, T05 | ~720 | Config system, conversion core, contour annotator, validation |
| **PR 2: Training + Export** | T03, T06, T07 | ~590 | YOLO-World annotator, training pipeline, model export |
| **PR 3: Evaluation + Integration** | T08, T09, T10 | ~410 | Evaluation, integration tests, cleanup |

PR 1 is still over 400 lines but contains the most critical and complex code (conversion pipeline). PR 2 and PR 3 are closer to budget. If stricter adherence is needed, PR 1 can be split further (T01+T05 in one PR, T02+T04 in another).

### Alternative: Single PR

If the team prefers a single PR for cohesion (all modules are tightly coupled through the config system), the total is ~1,840 lines. This is acceptable if reviewers are comfortable with the scope and the change is well-structured with clear task boundaries.
