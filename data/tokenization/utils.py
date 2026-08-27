"""
Shared utilities for tokenization and packing scripts.
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve()
while not (_REPO_ROOT / "pyproject.toml").exists() and _REPO_ROOT.parent != _REPO_ROOT:
    _REPO_ROOT = _REPO_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.dataset_loader import DatasetLoader as DatasetLoader  # noqa: E402
from shared.files import infer_file_features as infer_file_features  # noqa: E402
from shared.files import list_matching_files as list_matching_files  # noqa: E402
from shared.logging import get_logger as get_logger  # noqa: E402
from shared.save_dataset import save_dataset as save_dataset  # noqa: E402


def save_metadata(output_dir, **kwargs):
    """Write key-value metadata to `<output_dir>/.metadata`."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, ".metadata"), "w") as f:
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")
