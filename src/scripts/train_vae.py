"""
Backward-compatible alias for AE/VAE training.

Prefer running:
  `python -m src.scripts.train_ae --config <yaml>`

This module exists so older notes/commands that reference `train_vae` still work,
but it delegates to the canonical implementation in `src/scripts/train_ae.py`.
"""

from __future__ import annotations

from src.scripts.train_ae import main


if __name__ == "__main__":
    main()

