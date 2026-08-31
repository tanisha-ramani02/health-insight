"""Loguru logging configuration with date-wise and timestamped files."""

import sys
from datetime import datetime
from pathlib import Path
from loguru import logger
from .settings import settings


def setup_logger():
    """Configure loguru handlers for console and daily log files."""
    logger.remove()

    # Console Handler (Colorized and concise)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )

    # Date-wise Log Directory: logs/YYYY-MM-DD/
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    log_dir = settings.LOGS_PATH / date_str
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = log_dir / f"app_{now.strftime('%H-%M-%S')}.log"

    # Persistent File Handler (Captures DEBUG and above with rotation and retention)
    logger.add(
        str(log_file_path),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} | {message}",
        level="DEBUG",
        rotation="50 MB",
        retention="30 days",
        encoding="utf-8"
    )

    logger.info(f"Logging initialized. Log file: {log_file_path}")
    return logger


# Initialize on import
app_logger = setup_logger()
