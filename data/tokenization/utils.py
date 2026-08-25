"""
Shared utilities for tokenization and packing scripts.
"""

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
