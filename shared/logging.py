"""Shared logging utilities."""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and return a logger with a consistent format.

    Args:
        name: Logger name (e.g., __name__).
        level: Logging level (default: logging.INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Always apply level and propagate settings, even if the handler was
    # already added by a previous call.
    logger.setLevel(level)
    logger.propagate = False

    if not getattr(logger, "_configured", False):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger._configured = True

    return logger
