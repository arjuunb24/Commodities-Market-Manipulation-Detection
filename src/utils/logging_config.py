"""
src/utils/logging_config.py
============================
Structured logging configuration for the Veridex pipeline.

Sets up a consistent log format with timestamps, module names, and log levels.
LOG_LEVEL is read from the environment (defaults to INFO).

Usage:
    from src.utils.logging_config import setup_logging
    setup_logging()  # call once at application entry point

Or import this module — importing it calls setup_logging() automatically
with the environment-configured level.
"""

from __future__ import annotations

import logging
import os
import sys


def setup_logging(level: str | None = None) -> None:
    """
    Configure root logger with a structured format.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
               Defaults to the LOG_LEVEL environment variable, then "INFO".
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()

    numeric_level = getattr(logging, level, logging.INFO)

    # Format: timestamp | level | module:line | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(numeric_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging() is called multiple times
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    else:
        root_logger.handlers.clear()
        root_logger.addHandler(handler)

    # Quieten noisy third-party loggers
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)


# Auto-configure on import
setup_logging()
