"""
Shared utilities for tokenization and packing scripts.
"""

import glob
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve()
while not (_REPO_ROOT / "pyproject.toml").exists() and _REPO_ROOT.parent != _REPO_ROOT:
    _REPO_ROOT = _REPO_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.dataset_loader import DatasetLoader as DatasetLoader  # noqa: E402
from shared.logging import get_logger as get_logger  # noqa: E402
from shared.save_dataset import save_dataset as save_dataset  # noqa: E402


def save_metadata(output_dir, **kwargs):
    """Write key-value metadata to `<output_dir>/.metadata`."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, ".metadata"), "w") as f:
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")


def list_matching_files(directory: str, *patterns: str) -> list[str]:
    """Return a sorted list of files in `directory` matching any glob pattern."""
    matches: set[str] = set()
    for pattern in patterns:
        matches.update(glob.glob(os.path.join(directory, pattern)))
    return sorted(matches)


def infer_file_features(file_path: str, output_type: str) -> list[str]:
    """Infer output feature names from a written parquet or jsonl shard."""
    if output_type == "parquet":
        import pyarrow.parquet as pq

        return list(pq.read_schema(file_path).names)

    with open(file_path, encoding="utf-8") as fh:
        line = fh.readline()
        if not line.strip():
            return []
        return list(json.loads(line).keys())
