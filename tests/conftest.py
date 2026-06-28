"""
Pytest configuration.

Ensure the repository root is on `sys.path` so `import src.*` works when running
tests from the repo root without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

