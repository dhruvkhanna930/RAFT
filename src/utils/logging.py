"""Logging configuration.

Provides a pre-configured logger factory.  All experiment scripts should
obtain a logger via ``get_logger(__name__)`` rather than calling
``logging.getLogger`` directly.
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent format.

    Args:
        name: Logger name (typically ``__name__``).
        level: Logging level (default INFO).

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
