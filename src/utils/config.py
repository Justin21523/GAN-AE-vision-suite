"""
Configuration loading helpers.

This repository primarily uses YAML files under `configs/` to define datasets,
models, and training/evaluation settings.

Design goals:
- Keep configs human-editable (YAML).
- Support both `dict` access (`cfg["data"]["batch_size"]`) and attribute access
  (`cfg.data.batch_size`) when the optional `addict` dependency is available.
- Provide a small compatibility layer for historical key names used across the
  repo (e.g., `img_size` vs `image_size`, `train` vs `training`).
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional

import yaml

try:
    # `addict.Dict` behaves like a dict but also supports dotted access:
    # cfg.data.batch_size
    from addict import Dict as AddictDict  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AddictDict = None  # type: ignore


def _deep_update(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> None:
    """
    Recursively merge `src` into `dst`.

    Unlike `dict.update()`, this keeps existing nested keys unless explicitly
    overwritten by `src`.
    """
    for key, value in src.items():
        if (
            isinstance(value, Mapping)
            and isinstance(dst.get(key), Mapping)
            and dst.get(key) is not None
        ):
            _deep_update(dst[key], value)  # type: ignore[index]
        else:
            dst[key] = value


def _normalize_config(cfg: MutableMapping[str, Any]) -> None:
    """
    Normalize/alias commonly-used config keys across the repository.

    This repository has evolved over time; different modules/scripts may expect
    slightly different keys. The goal here is to make configs "just work"
    without forcing every caller to special-case legacy names.
    """
    data = cfg.get("data")
    if isinstance(data, MutableMapping):
        # Dataset name alias: `dataset` <-> `name`
        if "dataset" in data and "name" not in data:
            data["name"] = data["dataset"]
        if "name" in data and "dataset" not in data:
            data["dataset"] = data["name"]

        # Image size alias: `img_size` <-> `image_size`
        if "img_size" in data and "image_size" not in data:
            data["image_size"] = data["img_size"]
        if "image_size" in data and "img_size" not in data:
            data["img_size"] = data["image_size"]

    # Training section alias: `train` <-> `training`
    if "train" in cfg and "training" not in cfg:
        cfg["training"] = cfg["train"]
    if "training" in cfg and "train" not in cfg:
        cfg["train"] = cfg["training"]


def load_config(config_file: Optional[str] = None, default_file: str = "configs/default.yaml"):
    """
    Load a YAML config, optionally merging a user config on top of a default.

    Args:
        config_file: Path to a YAML file to merge on top of the default config.
        default_file: Base YAML file (defaults to `configs/default.yaml`).

    Returns:
        A mapping-like config object. If `addict` is installed, the return value
        supports dotted access (e.g., `cfg.data.batch_size`) in addition to
        regular dict indexing.
    """
    with open(default_file, "r", encoding="utf-8") as f:
        config: MutableMapping[str, Any] = yaml.safe_load(f) or {}

    if config_file is not None:
        with open(config_file, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        _deep_update(config, user_config)

    _normalize_config(config)

    # Convert to an "attribute dict" if available; otherwise return a plain dict.
    if AddictDict is not None:
        return AddictDict(config)
    return dict(config)


# ---------------------------------------------------------------------------
# Compatibility type aliases
# ---------------------------------------------------------------------------
# Some newer/experimental modules import `RunConfig` (and expect a structured,
# attribute-access config object). The repo currently uses YAML dicts, so we
# keep these names as aliases for readability without enforcing a schema here.
RunConfig = Any
