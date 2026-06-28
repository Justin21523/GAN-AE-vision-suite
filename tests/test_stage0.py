"""
Stage-0 smoke tests.

These tests verify that basic utilities (config loader, logger, seeding) import and run.
They intentionally avoid GPU work.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


def test_imports():
    from src.utils import seed, logger, config

    assert callable(seed.set_seed)
    assert callable(logger.setup_logger)
    assert callable(config.load_config)


def test_config_loading():
    from src.utils.config import load_config

    config = load_config()
    assert "seed" in config


def test_logger(tmpdir):
    from src.utils.logger import setup_logger

    logger = setup_logger(tmpdir)
    logger.info("Test logging")
    log_file = os.listdir(tmpdir)[0]
    with open(os.path.join(tmpdir, log_file)) as f:
        assert "Test logging" in f.read()
