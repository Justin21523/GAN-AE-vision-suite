"""
Logging utilities.

This project typically writes logs to `./logs/` (configurable per YAML config).
`setup_logger()` configures the root logger with:
- a timestamped log file in the provided directory
- a console (stdout) handler for interactive runs
"""

import logging
import os
import sys
from datetime import datetime


def setup_logger(log_dir: str, level=logging.INFO):
    """
    Configure and return the root logger.

    Notes:
    - This function attaches handlers to the *root* logger.
    - If you call it multiple times in one process, you may end up with
      duplicate handlers (duplicated log lines). For scripts, call it once.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = datetime.now().strftime("%Y%m%d-%H%M%S.log")
    log_filepath = os.path.join(log_dir, log_file)

    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
