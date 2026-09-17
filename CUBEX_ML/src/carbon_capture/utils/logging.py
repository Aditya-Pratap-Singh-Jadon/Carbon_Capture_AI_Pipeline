"""Structured logging configuration for pipeline operations."""

import logging
import sys
from typing import Optional

_LOGGERS = {}

def setup_logging(level: str = "INFO", log_format: Optional[str] = None) -> None:
    """Configure root logger with unified formatting."""
    if log_format is None:
        log_format = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def get_logger(name: str) -> logging.Logger:
    """Retrieve or create a namespaced logger."""
    if name not in _LOGGERS:
        logger = logging.getLogger(name)
        _LOGGERS[name] = logger
    return _LOGGERS[name]
