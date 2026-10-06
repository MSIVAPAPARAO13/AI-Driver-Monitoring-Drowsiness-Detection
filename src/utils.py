"""
General Utilities Module.

Provides path resolution, logging helpers, and common tools.
"""

import logging
from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def resolve_path(path: Union[str, Path]) -> Path:
    """
    Resolve a path. If relative, resolve against the project root.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return get_project_root() / p


def setup_logger(name: str = "drowsiness_detection", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standardized console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
