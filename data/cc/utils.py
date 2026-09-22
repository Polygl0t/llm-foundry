"""
Shared utilities for CommonCrawl processing scripts.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve()
while not (_REPO_ROOT / "pyproject.toml").exists() and _REPO_ROOT.parent != _REPO_ROOT:
    _REPO_ROOT = _REPO_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.logging import get_logger as get_logger  # noqa: E402
from shared.metadata import initialize_or_load_metadata as initialize_or_load_metadata  # noqa: E402
from shared.metadata import read_metadata as read_metadata  # noqa: E402
from shared.metadata import write_metadata as write_metadata  # noqa: E402
