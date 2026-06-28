"""
Run registry helpers.

This scans local run directories for:
- meta.json
- metrics.jsonl
- config_resolved.yaml

It is intended for local UI browsing and comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RunInfo:
    id: str
    path: str
    meta_path: str
    metrics_path: Optional[str]
    config_path: Optional[str]
    created_at: Optional[str]
    script: Optional[str]


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tail_jsonl(path: Path, max_lines: int = 50) -> List[Dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for raw in lines[-int(max_lines) :]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except Exception:
            continue
    return out


def scan_runs(repo_root: Path, max_runs: int = 200) -> List[RunInfo]:
    """
    Scan `repo_root/logs/**/meta.json` for run directories.
    """
    repo_root = repo_root.resolve()
    logs_root = (repo_root / "logs").resolve()
    if not logs_root.exists():
        return []

    runs: List[RunInfo] = []
    for meta_path in logs_root.rglob("meta.json"):
        if not meta_path.is_file():
            continue
        run_dir = meta_path.parent
        rel = str(run_dir.relative_to(repo_root)).replace("\\", "/")
        meta = _safe_read_json(meta_path) or {}
        metrics = run_dir / "metrics.jsonl"
        config = run_dir / "config_resolved.yaml"
        runs.append(
            RunInfo(
                id=rel,
                path=str(run_dir),
                meta_path=str(meta_path),
                metrics_path=str(metrics) if metrics.exists() else None,
                config_path=str(config) if config.exists() else None,
                created_at=meta.get("created_at"),
                script=(meta.get("script") if isinstance(meta.get("script"), str) else None),
            )
        )
        if len(runs) >= int(max_runs):
            break

    # newest first (created_at is sortable yyyyMMdd-HHMMSS if present)
    runs.sort(key=lambda r: (r.created_at or "", r.id), reverse=True)
    return runs


def get_run(repo_root: Path, run_id: str) -> Optional[Path]:
    repo_root = repo_root.resolve()
    # run_id is a repo-relative path like "logs/gan_...".
    p = (repo_root / run_id).resolve()
    try:
        p.relative_to(repo_root)
    except Exception:
        return None
    if not p.exists() or not p.is_dir():
        return None
    return p


def load_run_detail(run_dir: Path, tail_metrics: int = 200) -> Dict[str, Any]:
    meta_path = run_dir / "meta.json"
    cfg_path = run_dir / "config_resolved.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    meta = _safe_read_json(meta_path) if meta_path.exists() else None
    metrics = _tail_jsonl(metrics_path, max_lines=int(tail_metrics)) if metrics_path.exists() else []
    return {
        "id": str(run_dir),
        "path": str(run_dir),
        "meta_path": str(meta_path) if meta_path.exists() else None,
        "config_path": str(cfg_path) if cfg_path.exists() else None,
        "metrics_path": str(metrics_path) if metrics_path.exists() else None,
        "meta": meta,
        "metrics_tail": metrics,
    }

