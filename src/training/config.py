"""Configuration loader for the YOLOv8 waste detection training pipeline.

Loads YAML config, resolves ${var} interpolation, resolves paths relative
to the config file directory, and validates required keys and value ranges.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Accumulates multiple validation errors instead of failing on the first one."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(self._format())

    def _format(self) -> str:
        count = len(self.errors)
        header = f"Config validation failed with {count} error(s):"
        items = "\n".join(f"  - {e}" for e in self.errors)
        return f"{header}\n{items}"


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

_INTERPOLATION_RE = re.compile(r"\$\{(\w+)\}")


def _interpolate_value(value: Any, flat: dict[str, Any]) -> Any:
    """Replace ${var} references in a string value using top-level keys."""
    if not isinstance(value, str):
        return value

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key in flat:
            return str(flat[key])
        return match.group(0)  # leave unresolved references as-is

    return _INTERPOLATION_RE.sub(_replace, value)


def _interpolate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve ${var} references across the entire config dict.

    Only top-level keys are used as interpolation sources. Nested dicts are
    walked so that string values inside them also get interpolated.
    """
    flat = {k: v for k, v in config.items() if not isinstance(v, dict)}

    resolved: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            resolved[key] = {
                k: _interpolate_value(v, flat) for k, v in value.items()
            }
        else:
            resolved[key] = _interpolate_value(value, flat)
    return resolved


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_PATH_KEYS_TOP = ("project_root", "source_data", "output_dataset", "models_dir", "runs_dir")


def resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve path values to absolute ``Path`` objects.

    * ``project_root`` is resolved relative to the config file directory
      (stored in ``config["_config_dir"]`` by ``load_config``).
    * All other top-level path keys are resolved relative to ``project_root``.
    * ``source_data`` is left as-is when it is already absolute.

    Mutates *config* in place and also returns it for convenience.
    """
    config_dir = config.get("_config_dir")
    if config_dir is None:
        raise ValueError(
            "Cannot resolve paths: config was not loaded via load_config() "
            "(missing _config_dir). Set _config_dir manually or use load_config()."
        )

    config_dir = Path(config_dir)

    # project_root is relative to the config file's directory
    raw_root = config.get("project_root", ".")
    project_root = (config_dir / raw_root).resolve()
    config["project_root"] = project_root

    for key in _PATH_KEYS_TOP:
        if key == "project_root":
            continue
        if key not in config:
            continue
        raw = config[key]
        p = Path(raw)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        config[key] = p

    return config


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_TOP_KEYS = (
    "project_root",
    "source_data",
    "output_dataset",
    "models_dir",
    "runs_dir",
    "auto_annotate",
    "training",
    "augmentation",
    "export",
    "mlflow",
    "classes",
)


def validate_config(config: dict[str, Any]) -> None:
    """Validate required keys and value ranges.

    Raises ``ConfigError`` listing **all** validation failures at once.
    """
    errors: list[str] = []

    # --- Required top-level keys ---
    for key in _REQUIRED_TOP_KEYS:
        if key not in config:
            errors.append(f"Missing required key: '{key}'")

    # --- source_data must exist on disk ---
    source = config.get("source_data")
    if source is not None:
        source_path = Path(source) if not isinstance(source, Path) else source
        if not source_path.exists():
            errors.append(f"source_data path does not exist: {source_path}")

    # --- classes must have exactly 6 entries ---
    classes = config.get("classes")
    if classes is not None:
        if not isinstance(classes, list):
            errors.append("'classes' must be a list")
        elif len(classes) != 6:
            errors.append(
                f"'classes' must have exactly 6 entries, got {len(classes)}"
            )

    # --- auto_annotate checks ---
    aa = config.get("auto_annotate")
    if isinstance(aa, dict):
        conf = aa.get("confidence_threshold")
        if conf is not None:
            try:
                conf_f = float(conf)
                if not (0.0 <= conf_f <= 1.0):
                    errors.append(
                        f"auto_annotate.confidence_threshold must be in [0.0, 1.0], got {conf}"
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"auto_annotate.confidence_threshold must be numeric, got {conf!r}"
                )

        method = aa.get("method")
        if method is not None and method not in ("yolo-world", "contour"):
            errors.append(
                f"auto_annotate.method must be 'yolo-world' or 'contour', got '{method}'"
            )

        bs = aa.get("batch_size")
        if bs is not None:
            try:
                if int(bs) <= 0:
                    errors.append(
                        f"auto_annotate.batch_size must be > 0, got {bs}"
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"auto_annotate.batch_size must be an integer, got {bs!r}"
                )

    # --- training checks ---
    tr = config.get("training")
    if isinstance(tr, dict):
        batch = tr.get("batch")
        if batch is not None:
            try:
                if int(batch) <= 0:
                    errors.append(f"training.batch must be > 0, got {batch}")
            except (TypeError, ValueError):
                errors.append(
                    f"training.batch must be an integer, got {batch!r}"
                )

        epochs = tr.get("epochs")
        if epochs is not None:
            try:
                if int(epochs) <= 0:
                    errors.append(f"training.epochs must be > 0, got {epochs}")
            except (TypeError, ValueError):
                errors.append(
                    f"training.epochs must be an integer, got {epochs!r}"
                )

        imgsz = tr.get("imgsz")
        if imgsz is not None:
            try:
                if int(imgsz) <= 0:
                    errors.append(f"training.imgsz must be > 0, got {imgsz}")
            except (TypeError, ValueError):
                errors.append(
                    f"training.imgsz must be an integer, got {imgsz!r}"
                )

    if errors:
        raise ConfigError(errors)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML config file and return a fully resolved dict.

    Steps:
    1. Parse YAML.
    2. Resolve ``${var}`` interpolation using top-level keys.
    3. Resolve all path values to absolute ``Path`` objects.
    4. Validate required keys and value ranges.

    Parameters
    ----------
    config_path:
        Path to the YAML config file (relative or absolute).

    Returns
    -------
    dict
        The resolved configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    ConfigError
        If validation fails (lists ALL failures).
    """
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")

    # Stash the config file directory so resolve_paths can use it
    raw["_config_dir"] = str(path.parent)
    raw["_config_path"] = str(path)

    # Step 1: resolve top-level paths first (so interpolation picks up absolutes)
    resolve_paths(raw)

    # Step 2: interpolation — ${runs_dir} now resolves to the absolute path
    config = _interpolate_config(raw)

    # Step 3: validation
    validate_config(config)

    return config
