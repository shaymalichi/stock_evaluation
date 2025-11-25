import logging
import os
import sys
from datetime import datetime

_initialized = False


def setup_logging(log_dir: str = "data/logs", console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Saves logs to .data/logs
    """
    global _initialized
    if _initialized:
        return

    os.makedirs(log_dir, exist_ok=True)

    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file_path = os.path.join(log_dir, f"run_{current_date}.log")

    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(log_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)

    _initialized = True
    logging.info(f"📝 Logger initialized. Writing to: {log_file_path}")