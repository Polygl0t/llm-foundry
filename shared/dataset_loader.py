"""Shared dataset loading utilities.

Provides:
    - DatasetLoader: A class for loading datasets from local files, directories, or HuggingFace Hub.
"""

import glob
import os

import datasets


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
                num_proc = (
                    min(len(files), self.num_proc) if self.num_proc is not None else len(files)
                )
                return datasets.load_dataset(
                    fmt,
                    data_files=files,
                    split="train",
                    num_proc=num_proc,
                    cache_dir=self.cache_dir,
                )
        raise ValueError(f"No .jsonl or .parquet files found in '{path}'.")

    def _from_hf(self, path: str):
        load_args = {"path": path, "split": self.split, "cache_dir": self.cache_dir}
        if self.subset is not None:
            load_args["name"] = self.subset
        return datasets.load_dataset(**load_args)
