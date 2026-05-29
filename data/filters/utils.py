"""
Shared utilities for language filter scripts.
"""
import glob
import os
import numpy as np
import datasets


class DatasetLoader:
    """Loads datasets from a local file, local directory, or HuggingFace Hub.
    Source type is detected automatically:
    - Directory  -> all .jsonl or .parquet files inside are loaded.
    - Local file -> .jsonl or .parquet are supported.
    - Anything else is treated as a HuggingFace Hub dataset identifier.
    """
    _FILE_FORMATS = {".jsonl": "json", ".json": "json", ".parquet": "parquet"}

    def __init__(self, path, cache_dir=None, seed=None, split="train", subset=None):
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


def save_dataset(dataset, output_dir, output_type, tokens_per_chunk, token_count, *, n_chunks=None):
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
    if column_name not in dataset.column_names:
        return False
    for example in dataset:
        value = example.get(column_name)
        if value is None:
            continue
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], dict) and 'content' in value[0]:
                return True
        break
    return False


def flatten_messages(messages):
    if not messages:
        return ""
    contents = []
    for msg in messages:
        if isinstance(msg, dict) and 'content' in msg:
            content = msg['content']
            if content:
                contents.append(str(content))
    return '\n'.join(contents)
