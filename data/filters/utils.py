"""
Shared utilities for filtering and annotation scripts.
"""

import glob
import json
import logging
import os
import sys

import datasets
import numpy as np


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a logger with a consistent format.

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


class DatasetLoader:
    """Loads datasets from a local file, local directory, or HuggingFace Hub.

    Source type is detected automatically:
    - Directory  -> all .jsonl or .parquet files inside are loaded.
    - Local file -> .jsonl or .parquet are supported.
    - Anything else is treated as a HuggingFace Hub dataset identifier.

    When `path` is a list, each entry is loaded independently and the
    resulting datasets are concatenated.  Entries may mix directories, files,
    and HuggingFace Hub identifiers in a single list.
    """

    _FILE_FORMATS = {".jsonl": "json", ".json": "json", ".parquet": "parquet"}

    def __init__(
        self,
        path: str | list[str],
        cache_dir: str | None = None,
        seed: int | None = None,
        split: str = "train",
        subset: str | None = None,
        num_proc: int | None = None,
    ) -> None:
        self.path = path
        self.cache_dir = cache_dir
        self.seed = seed
        self.split = split
        self.subset = subset
        self.num_proc = num_proc

    def load(self):
        paths = [self.path] if isinstance(self.path, str) else self.path
        if not paths:
            raise ValueError("At least one path must be provided.")

        datasets_list = [self._load_single(p) for p in paths]

        if len(datasets_list) == 1:
            dataset = datasets_list[0]
        else:
            dataset = datasets.concatenate_datasets(datasets_list)

        return dataset.shuffle(seed=self.seed) if self.seed is not None else dataset

    def _load_single(self, path: str):
        """Load a single path, dispatching by type (directory, file, or HF)."""
        if os.path.isdir(path):
            return self._from_directory(path)
        elif os.path.isfile(path):
            return self._from_file(path)
        else:
            return self._from_hf(path)

    def _from_file(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        fmt = self._FILE_FORMATS.get(ext)
        if fmt is None:
            raise ValueError(f"Unsupported file format '{ext}'. Expected .jsonl or .parquet.")
        return datasets.load_dataset(fmt, data_files=path, split="train", cache_dir=self.cache_dir)

    def _from_directory(self, path: str):
        for ext, fmt in (("*.jsonl", "json"), ("*.parquet", "parquet")):
            files = sorted(glob.glob(os.path.join(path, ext)))
            if files:
                return datasets.load_dataset(
                    fmt,
                    data_files=files,
                    split="train",
                    num_proc=self.num_proc if self.num_proc is not None else len(files),
                    cache_dir=self.cache_dir,
                )
        raise ValueError(f"No .jsonl or .parquet files found in '{path}'.")

    def _from_hf(self, path: str):
        load_args = {"path": path, "split": self.split, "cache_dir": self.cache_dir}
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


def read_metadata(metadata_file: str) -> dict | None:
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
    with open(metadata_file, encoding="utf-8") as f:
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

    Loads from an existing `.metadata` file if present. Otherwise scans all
    JSONL files in the folder to rebuild the statistics and writes the result
    to `.metadata` for future calls.

    Args:
        lang_output_path: Path to the language-specific output folder.

    Returns:
        Dictionary with at least `lines` and `tokens` keys.
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
        with open(jsonl_file, encoding="utf-8") as f:
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


def is_messages_column(dataset, column_name):
    """Heuristically determine if a column contains a list of message dicts with 'content' fields."""
    if column_name not in dataset.column_names or len(dataset) == 0:
        return False
    value = dataset[0].get(column_name)
    if isinstance(value, list) and len(value) > 0:
        return isinstance(value[0], dict) and "content" in value[0]
    return False


def flatten_messages(messages):
    """Convert a list of message dicts to a single string by concatenating the 'content' fields."""
    if not messages:
        return ""
    contents = []
    for msg in messages:
        if isinstance(msg, dict) and "content" in msg:
            content = msg["content"]
            if content:
                contents.append(str(content))
    return "\n".join(contents)


def apply_chat_template_to_dataset(dataset, tokenizer, text_column, num_proc):
    """
    Apply chat template formatting to the text column of the dataset.

    Args:
        dataset: The input dataset
        tokenizer: The tokenizer with chat template
        text_column: Name of the column containing text
        num_proc: Number of processes for parallel processing

    Returns:
        tuple: (formatted_dataset, new_text_column_name)
    """
    if tokenizer.chat_template is None:
        raise ValueError(
            "The tokenizer does not have a chat template. "
            "Please use a tokenizer that supports chat templates."
        )

    def format_messages(example):
        formatted_text = tokenizer.apply_chat_template(
            example[text_column],
            tokenize=False,  # Returns string instead of token IDs
        )
        example["formatted_text"] = formatted_text
        return example

    formatted_dataset = dataset.map(
        format_messages, num_proc=num_proc, desc="Formatting messages with chat template"
    )

    return formatted_dataset, "formatted_text"
