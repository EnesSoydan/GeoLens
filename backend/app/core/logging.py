"""Minimal logging configuration for the backend."""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once (idempotent for repeated calls)."""
    logging.basicConfig(level=level, format=_LOG_FORMAT, stream=sys.stdout)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
