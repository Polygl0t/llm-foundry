"""
Shared utilities for CommonCrawl processing scripts.
"""
import glob
import json
import logging
import os
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a logger with a consistent format.

    Args:
        name: Logger name.
        level: Logging level (default: logging.INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid adding duplicate handlers if the logger was already configured.
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent log records from propagating to the root logger.
    logger.propagate = False

    return logger


def read_metadata(metadata_file: str) -> Optional[dict]:
    """
    Read metadata from a file in YAML-like key: value format.

    Args:
        metadata_file: Path to the metadata file.

    Returns:
        Dictionary with metadata values, or None if the file does not exist.
    """
    if not os.path.exists(metadata_file):
        return None

    metadata = {}
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    if "." in value:
                        metadata[key] = float(value)
                    else:
                        metadata[key] = int(value)
                except ValueError:
                    metadata[key] = value
    return metadata


def write_metadata(metadata_file: str, metadata: dict) -> None:
    """
    Write metadata to a file in YAML-like key: value format.

    Args:
        metadata_file: Path to the metadata file to write.
        metadata: Dictionary of metadata values to persist.
    """
    with open(metadata_file, "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")


def initialize_or_load_metadata(lang_output_path: str) -> dict:
    """
    Return metadata for a language output folder.

    Loads from an existing ``.metadata`` file if present. Otherwise scans all
    JSONL files in the folder to rebuild the statistics and writes the result
    to ``.metadata`` for future calls.

    Args:
        lang_output_path: Path to the language-specific output folder.

    Returns:
        Dictionary with at least ``lines`` and ``tokens`` keys.
    """
    metadata_file = os.path.join(lang_output_path, ".metadata")

    metadata = read_metadata(metadata_file)
    if metadata is not None:
        return metadata

    all_jsonl_files = glob.glob(os.path.join(lang_output_path, "*.jsonl"))

    if not all_jsonl_files:
        return {"lines": 0, "tokens": 0}

    total_lines = 0
    total_tokens = 0

    for jsonl_file in all_jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    total_lines += 1
                    total_tokens += data.get("token_count", 0)
                except json.JSONDecodeError:
                    continue

    metadata = {"lines": total_lines, "tokens": total_tokens}
    write_metadata(metadata_file, metadata)
    return metadata
