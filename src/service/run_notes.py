"""
Run notes/tags helpers.

Notes are stored per-run under:
  <run_dir>/notes.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def notes_path(run_dir: Path) -> Path:
    return run_dir / "notes.json"


def read_notes(run_dir: Path) -> Dict[str, Any]:
    p = notes_path(run_dir)
    if not p.exists():
        return {"tags": [], "note": "", "updated_at": None}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"tags": [], "note": "", "updated_at": None}
    if not isinstance(obj, dict):
        return {"tags": [], "note": "", "updated_at": None}
    tags = obj.get("tags") if isinstance(obj.get("tags"), list) else []
    note = obj.get("note") if isinstance(obj.get("note"), str) else ""
    updated_at = obj.get("updated_at")
    return {"tags": tags, "note": note, "updated_at": updated_at}


def write_notes(run_dir: Path, tags: Optional[List[str]] = None, note: Optional[str] = None) -> Dict[str, Any]:
    cur = read_notes(run_dir)
    if tags is not None:
        cur["tags"] = [str(t).strip() for t in tags if str(t).strip()]
    if note is not None:
        cur["note"] = str(note)
    cur["updated_at"] = time.time()
    p = notes_path(run_dir)
    p.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")
    return cur

