# Shared Utilities

This folder contains utility modules shared across the major pipeline folders.

## Modules

| Module                                   | Provides        | Notes                                                             |
|------------------------------------------|-----------------|-------------------------------------------------------------------|
| [`dataset_loader.py`](dataset_loader.py) | `DatasetLoader` | Loads datasets from local files, directories, or HuggingFace Hub. |
| [`logging.py`](logging.py)               | `get_logger`    | Creates a logger with a consistent format.                        |
| [`save_dataset.py`](save_dataset.py)     | `save_dataset`  | Saves a dataset to disk, splitting into token-counted chunks.     |

## Usage

Each pipeline folder re-exports these utilities from its own `utils.py`, so the existing `from utils import ...` imports keep working.

```python
from shared.dataset_loader import DatasetLoader
from shared.logging import get_logger
from shared.save_dataset import save_dataset
```

Local `utils.py` files add the repository root to `sys.path` (by walking up to `pyproject.toml`) before importing from `shared`, since the project is not installed as a Python package (for now).

## Adding new shared utilities

1. Put the function/class in a new or existing module in this folder.
2. Re-export it from the relevant local `utils.py` files.
3. Update this README's module table.
