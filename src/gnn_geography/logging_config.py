"""Logging configuration for gnn_geography."""

import sys
from pathlib import Path

from loguru import logger


def configure_logging(
    log_level: str = "INFO",
    log_dir: Path | str | None = None,
    file_rotation: str = "500 MB",
) -> None:
    """Configure structured logging with file rotation.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files. If None, logs to stderr only.
        file_rotation: Rotation trigger (e.g., "500 MB", "00:00")
    """
    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    # File handler (optional)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "gnn_geography.log",
            level=log_level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{name}:{function}:{line} - {message}"
            ),
            rotation=file_rotation,
            retention="10 days",
        )


# Default configuration
configure_logging()
