"""
Safe local filesystem browser helpers for the UI.

Design:
- Local-dev oriented (no auth).
- Restricts access to a small set of allowed roots (repo root + AI_CACHE_ROOT).
- Provides simple list/read/file helpers used by FastAPI endpoints.
"""

from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class FSEntry:
    name: str
    path: str
    type: str  # file|dir
    size: int
    mtime: float


class SafeFS:
    def __init__(self, allowed_roots: Sequence[Path]):
        roots = [Path(r).resolve() for r in allowed_roots]
        self.allowed_roots = roots

    def resolve(self, p: str) -> Path:
        p = os.path.expanduser(str(p))
        path = Path(p)
        if not path.is_absolute():
            # default to first allowed root
            base = self.allowed_roots[0]
            path = (base / path).resolve()
        else:
            path = path.resolve()

        if not any(_is_within(path, r) for r in self.allowed_roots):
            raise PermissionError("Path is outside allowed roots.")
        return path

    def list_dir(self, p: str) -> Dict[str, Any]:
        path = self.resolve(p)
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        entries: List[FSEntry] = []
        for child in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            try:
                st = child.stat()
            except Exception:
                continue
            entries.append(
                FSEntry(
                    name=child.name,
                    path=str(child),
                    type="dir" if child.is_dir() else "file",
                    size=int(st.st_size),
                    mtime=float(st.st_mtime),
                )
            )

        return {
            "path": str(path),
            "entries": [e.__dict__ for e in entries],
            "allowed_roots": [str(r) for r in self.allowed_roots],
        }

    def read_text(self, p: str, max_bytes: int = 200_000) -> Dict[str, Any]:
        path = self.resolve(p)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        raw = path.read_bytes()
        truncated = len(raw) > int(max_bytes)
        raw = raw[: int(max_bytes)]
        text = raw.decode("utf-8", errors="replace")
        return {"path": str(path), "text": text, "truncated": truncated, "size": int(path.stat().st_size)}

    def write_text(self, p: str, text: str, overwrite: bool = False) -> Dict[str, Any]:
        path = self.resolve(p)
        if path.exists() and not overwrite:
            raise FileExistsError("File exists (set overwrite=true to replace).")
        if path.exists() and not path.is_file():
            raise IsADirectoryError("Target exists and is not a file.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return {"path": str(path), "size": int(path.stat().st_size)}

    def mkdir(self, p: str, parents: bool = True, exist_ok: bool = True) -> Dict[str, Any]:
        path = self.resolve(p)
        path.mkdir(parents=bool(parents), exist_ok=bool(exist_ok))
        return {"path": str(path)}

    def guess_mime(self, p: str) -> str:
        path = self.resolve(p)
        mime, _ = mimetypes.guess_type(str(path))
        return mime or "application/octet-stream"
