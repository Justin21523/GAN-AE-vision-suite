"""
Placeholder inference entrypoint.

Most inference utilities in this repository live in:
- `src/service/gan_infer.py` (GAN checkpoint loading + sampling)
- `src/scripts/sample_gan.py` (CLI sampler for GAN checkpoints)
"""

import sys, os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main() -> None:
    """This module is currently a placeholder."""
    raise SystemExit(
        "src/infer.py is a placeholder. Try `python -m src.scripts.sample_gan ...` instead."
    )


if __name__ == "__main__":
    main()
