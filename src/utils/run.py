"""
Run/artifact utilities for training scripts.

Goals:
- Keep training runs reproducible and debuggable.
- Write small, append-only metrics (`metrics.jsonl`).
- Persist the effective config (`config_resolved.yaml`) alongside checkpoints.
- Provide lightweight "data fingerprints" (e.g., split list hashes).
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import yaml


def _now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def to_builtin(x: Any) -> Any:
    """
    Convert nested config objects (e.g., addict.Dict) into YAML-serializable types.
    """
    if isinstance(x, Mapping):
        return {str(k): to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    # Best-effort fallback (keeps `yaml.safe_dump` from crashing).
    return str(x)


def _to_builtin(x: Any) -> Any:
    """Backward-compatible alias for older imports."""
    return to_builtin(x)


def prepare_run_dir(base_dir: str | Path, run_name: Optional[str] = None, prefix: str = "run") -> Path:
    """
    Create and return a run directory.

    If `run_name` is not provided, uses `<prefix>_<YYYYmmdd-HHMMSS>`.
    """
    base = Path(base_dir)
    name = str(run_name).strip() if run_name else f"{prefix}_{_now_ts()}"
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_config_yaml(cfg: Mapping[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(to_builtin(cfg), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _sha256_text(path: Path) -> str:
    h = sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def data_fingerprint(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Best-effort data fingerprint.

    - If split list files are configured, include their sha256.
    - Otherwise, include dataset name + root only.
    """
    data = cfg.get("data", cfg)  # type: ignore[arg-type]
    out: Dict[str, Any] = {
        "dataset": str(data.get("dataset") or data.get("name") or "unknown"),
        "root": str(data.get("root", "")),
    }

    splits = data.get("splits") or {}
    train_list = data.get("train_list") or splits.get("train")
    val_list = data.get("val_list") or splits.get("val")

    def _add_list(key: str, v: Optional[str]) -> None:
        if not v:
            return
        p = Path(os.path.expanduser(str(v)))
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.exists():
            out[key] = {"path": str(p), "sha256": _sha256_text(p)}
        else:
            out[key] = {"path": str(p), "missing": True}

    _add_list("train_list", train_list)
    _add_list("val_list", val_list)
    return out


@dataclass
class JSONLMetricsWriter:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, obj: Mapping[str, Any]) -> None:
        line = json.dumps(dict(obj), ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def build_run_meta(cfg: Mapping[str, Any], extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "created_at": _now_ts(),
        "python": platform.python_version(),
        "torch": getattr(torch, "__version__", "unknown"),
        "device_available_cuda": bool(torch.cuda.is_available()),
        "data": data_fingerprint(cfg),
    }
    if extra:
        meta.update(dict(extra))
    return meta
