"""
Shared utilities for filtering and annotation scripts.
"""
import glob
import os
import sys
import numpy as np
import datasets
import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a logger with a consistent format.

    Args:
        name: Logger name (typically __name__ of the calling module).
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


class DatasetLoader:
    """Loads datasets from a local file, local directory, or HuggingFace Hub.

    Source type is detected automatically:
    - Directory  -> all .jsonl or .parquet files inside are loaded.
    - Local file -> .jsonl or .parquet are supported.
    - Anything else is treated as a HuggingFace Hub dataset identifier.
    """

    _FILE_FORMATS = {".jsonl": "json", ".json": "json", ".parquet": "parquet"}

    def __init__(
        self,
        path: str,
        cache_dir: str | None = None,
        seed: int | None = None,
        split: str = "train",
        subset: str | None = None,
    ) -> None:
        self.path = path
        self.cache_dir = cache_dir
        self.seed = seed
        self.split = split
        self.subset = subset

    def load(self):
        if os.path.isdir(self.path):
            dataset = self._from_directory()
        elif os.path.isfile(self.path):
            dataset = self._from_file()
        else:
            dataset = self._from_hf()
        return dataset.shuffle(seed=self.seed) if self.seed is not None else dataset

    def _from_file(self):
        ext = os.path.splitext(self.path)[1].lower()
        fmt = self._FILE_FORMATS.get(ext)
        if fmt is None:
            raise ValueError(f"Unsupported file format '{ext}'. Expected .jsonl or .parquet.")
        return datasets.load_dataset(fmt, data_files=self.path, split="train", cache_dir=self.cache_dir)

    def _from_directory(self):
        for ext, fmt in (("*.jsonl", "json"), ("*.parquet", "parquet")):
            files = sorted(glob.glob(os.path.join(self.path, ext)))
            if files:
                return datasets.load_dataset(
                    fmt,
                    data_files=files,
                    split="train",
                    num_proc=len(files),
                    cache_dir=self.cache_dir,
                )
        raise ValueError(f"No .jsonl or .parquet files found in '{self.path}'.")

    def _from_hf(self):
        load_args = {"path": self.path, "split": self.split, "cache_dir": self.cache_dir}
        if self.subset is not None:
            load_args["name"] = self.subset
        return datasets.load_dataset(**load_args)


def save_dataset(dataset, output_dir, output_type, tokens_per_chunk, *, token_count, n_chunks=None):
    """Save a dataset to disk, splitting into chunks of at most `tokens_per_chunk` tokens.

    Args:
        dataset:          HuggingFace Dataset to save.
        output_dir:       Directory to write output files into.
        output_type:      `'parquet'` or `'jsonl'`.
        tokens_per_chunk: Maximum number of tokens per output file.
        token_count:      Total token count (used to compute the number of chunks).
        n_chunks:         If provided, use this directly instead of computing from
                          `tokens_per_chunk` and `token_count`.

    Returns:
        Number of chunks written (0 if the dataset is empty).
    """
    sample_count = len(dataset)
    if sample_count == 0:
        return 0

    if n_chunks is None:
        n_chunks = max(1, (token_count + tokens_per_chunk - 1) // tokens_per_chunk)
    indices = np.array_split(np.arange(sample_count), n_chunks)

    os.makedirs(output_dir, exist_ok=True)
    extension = "parquet" if output_type == "parquet" else "jsonl"

    for i, idx in enumerate(indices):
        chunk = dataset.select(idx)
        filename = os.path.join(output_dir, f"train-{i:05d}-of-{n_chunks:05d}.{extension}")
        if output_type == "parquet":
            chunk.to_parquet(filename)
        else:
            chunk.to_json(filename)

    return n_chunks


def is_messages_column(dataset, column_name):
    """Heuristically determine if a column contains a list of message dicts with 'content' fields."""
    if column_name not in dataset.column_names or len(dataset) == 0:
        return False
    value = dataset[0].get(column_name)
    if isinstance(value, list) and len(value) > 0:
        return isinstance(value[0], dict) and 'content' in value[0]
    return False


def flatten_messages(messages):
    """Convert a list of message dicts to a single string by concatenating the 'content' fields."""
    if not messages:
        return ""
    contents = []
    for msg in messages:
        if isinstance(msg, dict) and 'content' in msg:
            content = msg['content']
            if content:
                contents.append(str(content))
    return '\n'.join(contents)
