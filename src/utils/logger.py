import logging
import os
import sys
from datetime import datetime


def setup_logger(log_dir: str, level=logging.INFO):
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
