"""
Checkpoint compatibility helpers.

PyTorch 2.6 changed `torch.load` to default to `weights_only=True`. Older
checkpoints in this project may contain config objects such as `addict.Dict`,
which are safe for local project checkpoints but rejected by the new default.
These helpers keep new checkpoints plain-dict friendly and load trusted local
checkpoints consistently across CLI/API entrypoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.utils.run import to_builtin


def checkpoint_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a torch-saveable payload with configs converted to builtin types."""
    out = dict(payload)
    if "cfg" in out:
        out["cfg"] = to_builtin(out["cfg"])
    return out


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """
    Load a trusted local project checkpoint.

    First tries PyTorch's default safer loader. If that fails because an older
    project checkpoint contains non-allowlisted config objects, fall back to
    `weights_only=False`. Do not use this helper for arbitrary untrusted files.
    """
    try:
        obj = torch.load(str(path), map_location=map_location)
    except Exception as first_error:
        try:
            obj = torch.load(str(path), map_location=map_location, weights_only=False)
        except TypeError:
            raise first_error
    if not isinstance(obj, dict):
        raise ValueError(f"Expected checkpoint dict, got {type(obj).__name__}.")
    return obj
