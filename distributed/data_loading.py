"""
Dataset loading and DataLoader creation for the distributed trainers.

Provides:
    - create_collate_fn:          factory for the default collate function with token masking
    - prepare_dataloaders:        main entry point; returns fully configured dataloaders
    - DataLoaderBundle:           return type bundling dataloaders and metadata

Dataset loading prefers a "prebuilt" dataset materialised once with
`Dataset.save_to_disk` by `distributed/prebuilt_dataset.py`. Building straight from
raw shards forks a `datasets` process pool on *every* rank, which on a shared cluster
filesystem can start-up into an I/O storm; memory-mapping a prebuilt dataset avoids it
entirely.

The helpers for reading raw shards and for that layout live in this module too, since
the builder needs the same contract:

    - limit_num_proc / effective_num_proc:            bound the `datasets` worker count.
    - discover_dataset_files:                         glob dataset shards from paths.
    - load_raw_dataset:                               build a `Dataset` from raw shards.
    - save_dataset_to_disk / write_metadata:          materialise one (builder side only).
    - load_dataset_from_disk:                         read a materialised `Dataset` back.
    - is_prebuilt_dataset / resolve_prebuilt_dataset: validate a prebuilt dataset path.

On-disk layout of a prebuilt dataset:

    <root>/
        train/          # output of datasets.save_to_disk()
        validation/     # output of datasets.save_to_disk()
        .metadata       # build provenance, written by the builder
"""

import glob
import os
import shutil
from dataclasses import dataclass

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import default_data_collator

# Map user-facing format names to HuggingFace `datasets` format names.
_FORMAT_MAP = {
    "parquet": "parquet",
    "jsonl": "json",
}

SUPPORTED_FORMATS = frozenset(_FORMAT_MAP)

# Layout written by `prebuilt_dataset.py` and read by `resolve_prebuilt_dataset`.
TRAIN_SUBDIR = "train"
VAL_SUBDIR = "validation"

# Files `Dataset.save_to_disk()` leaves behind, used by `is_prebuilt_dataset` for a
# cheap check before attempting to load.
_REQUIRED_FILES = ("dataset_info.json", "state.json")

# Conservative default worker count
DEFAULT_MAX_NUM_PROC = 16


@dataclass
class DataLoaderBundle:
    """Everything the training loop needs from the data pipeline."""

    train_dataloader: DataLoader
    val_dataloader: DataLoader
    train_sampler: DistributedSampler
    num_train_samples: int
    num_val_samples: int
    mask_token_ids: set


def create_collate_fn(mask_token_ids):
    """
    Create a collate function that generates labels from input_ids and masks
    the specified token IDs by setting them to -100.

    `mask_token_ids`: Collection of token IDs to mask in the labels.
    Typically includes pad, eos, bos, and any user-specified IDs.
    If empty, no masking is applied.

    This is the default collate function for the trainer. To swap in a different
    batching strategy (e.g., sequence packing), replace this factory with one that
    returns a function matching the signature `collate_fn(examples) -> batch`.
    """
    # Pre-compute a tensor of IDs to mask for efficient vectorized lookup.
    _mask_ids_tensor = (
        torch.tensor(sorted(mask_token_ids), dtype=torch.long) if mask_token_ids else None
    )

    def collate_fn(examples):
        batch = default_data_collator(examples)

        # If labels are already provided, trust them.
        if "labels" in batch:
            return batch

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        # Mask all specified token IDs in a single vectorized operation.
        if _mask_ids_tensor is not None:
            labels[torch.isin(labels, _mask_ids_tensor)] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def _validate_dataset_type(dataset_type):
    """Raise a helpful error when `dataset_type` is not supported."""
    if dataset_type not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported dataset_type '{dataset_type}'. "
            f"Expected one of: {sorted(SUPPORTED_FORMATS)}."
        )


def _as_list(paths):
    """Normalise a path or list of paths to a plain list."""
    if paths is None:
        return []
    if isinstance(paths, str | os.PathLike):
        return [paths]
    return list(paths)


def limit_num_proc(num_proc, max_num_proc=DEFAULT_MAX_NUM_PROC):
    """Clamp a requested `datasets` worker count to a safe maximum.

    Args:
        num_proc: Requested number of worker processes.
        max_num_proc: Hard upper bound. Pass `None` to disable clamping and keep the
            historical behaviour.

    Returns:
        `int` >= 1 and never larger than `max_num_proc` (when a maximum is given).

    Raises:
        ValueError: If `num_proc` is not a positive integer, or `max_num_proc` is
            given but is smaller than 1.
    """
    num_proc = int(num_proc)
    if num_proc < 1:
        raise ValueError(f"`num_proc` must be >= 1 (got {num_proc}).")

    if max_num_proc is None:
        return num_proc

    max_num_proc = int(max_num_proc)
    if max_num_proc < 1:
        raise ValueError(
            f"`max_num_proc` must be >= 1 (got {max_num_proc}). Use None to disable clamping."
        )

    return min(num_proc, max_num_proc)


def effective_num_proc(num_proc, num_files, max_num_proc):
    """Resolve the worker count to use for a build of `num_files` shards.

    Defaults to one worker per shard (the `datasets` intent), then applies both the
    configured maximum and the natural limit of one job per shard.
    """
    num_files = max(1, int(num_files))
    requested = num_files if num_proc is None else num_proc
    return min(limit_num_proc(requested, max_num_proc), num_files)


def discover_dataset_files(paths, dataset_type):
    """Return a sorted list of dataset shards from files and/or directories.

    Args:
        paths: A single path or a list of paths. A file must end in
            `.<dataset_type>`; a directory is globbed one level deep for
            `*.<dataset_type>`, matching the flat shard directories written by the
            tokenization/packing scripts.
        dataset_type: `'parquet'` or `'jsonl'`.

    Returns:
        Sorted list of file paths.

    Raises:
        ValueError: If no path is given, the format is unsupported, or a path is a
            file whose extension does not match `dataset_type`.
        FileNotFoundError: If a path does not exist.
    """
    _validate_dataset_type(dataset_type)

    paths = _as_list(paths)
    if not paths:
        raise ValueError("At least one dataset path must be provided.")

    files = []
    for path in paths:
        path = os.fspath(path)
        if os.path.isfile(path):
            if not path.endswith(f".{dataset_type}"):
                raise ValueError(f"File '{path}' does not match dataset_type '{dataset_type}'.")
            files.append(path)
        elif os.path.isdir(path):
            files += glob.glob(os.path.join(path, f"*.{dataset_type}"))
        else:
            raise FileNotFoundError(f"Dataset path does not exist: '{path}'")

    return sorted(files)


def load_raw_dataset(
    data_files,
    dataset_type,
    *,
    num_proc=None,
    max_num_proc=DEFAULT_MAX_NUM_PROC,
    cache_dir=None,
    split="train",
):
    """Build a `datasets.Dataset` from raw shards, with a bounded worker count.

    Args:
        data_files: List of shard paths (see `discover_dataset_files`).
        dataset_type: `'parquet'` or `'jsonl'`.
        num_proc: Requested worker count. Defaults to one worker per shard, then
            clamped by `max_num_proc`.
        max_num_proc: Hard upper bound on workers. Pass `None` to disable clamping.
        cache_dir: HuggingFace cache directory for the generated Arrow files.
        split: Split name to read.

    Returns:
        The loaded `datasets.Dataset`.

    Raises:
        ValueError: If `data_files` is empty or the format is unsupported.
    """
    _validate_dataset_type(dataset_type)

    data_files = _as_list(data_files)
    if not data_files:
        raise ValueError("`data_files` must contain at least one file.")

    num_proc = effective_num_proc(num_proc, len(data_files), max_num_proc)

    return datasets.load_dataset(
        _FORMAT_MAP[dataset_type],
        data_files=data_files,
        split=split,
        num_proc=num_proc,
        cache_dir=cache_dir,
    )


def write_metadata(output_dir, **kwargs):
    """Write `key: value` lines to `<output_dir>/.metadata`."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, ".metadata")
    with open(path, "w") as f:
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")
    return path


def save_dataset_to_disk(dataset, output_dir, *, metadata=None, overwrite=False):
    """Materialise a dataset into `output_dir` via `Dataset.save_to_disk`.

    Only used by the builder (`prebuilt_dataset.py`); the trainer reads the result
    back with `load_dataset_from_disk`.

    Args:
        dataset: HuggingFace `Dataset` (or `DatasetDict`) to materialise.
        output_dir: Destination directory.
        metadata: Optional flat mapping of key/value pairs written to
            `<output_dir>/.metadata` for provenance.
        overwrite: Remove an existing `output_dir` first. Without this, an existing
            directory is an error rather than being silently replaced.

    Returns:
        The absolute path that was written.

    Raises:
        FileExistsError: If `output_dir` exists and `overwrite` is False.
    """
    output_dir = os.path.abspath(os.fspath(output_dir))

    if os.path.exists(output_dir):
        if not overwrite:
            raise FileExistsError(
                f"'{output_dir}' already exists. Pass overwrite=True to replace it."
            )
        shutil.rmtree(output_dir)

    dataset.save_to_disk(output_dir)

    if metadata:
        write_metadata(output_dir, **metadata)

    return output_dir


def load_dataset_from_disk(path, *, validate_column=None):
    """Read a dataset materialised with `save_dataset_to_disk`.

    Args:
        path: Directory holding the output of `Dataset.save_to_disk()`.
        validate_column: Optional column that must be present (e.g. `input_ids`).

    Returns:
        The loaded `datasets.Dataset`.

    Raises:
        FileNotFoundError: If `path` is not an existing directory.
        ValueError: If `path` holds multiple splits (the trainer expects one split per
            path) or is missing `validate_column`.
    """
    path = os.path.abspath(os.fspath(path))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Prebuilt dataset directory not found: '{path}'")

    dataset = datasets.load_from_disk(path)

    if isinstance(dataset, datasets.DatasetDict):
        raise ValueError(
            f"'{path}' contains multiple splits ({list(dataset)}); expected a single "
            "dataset. Point at a split directory instead (e.g. '<root>/train')."
        )

    if validate_column is not None and validate_column not in dataset.column_names:
        raise ValueError(
            f"'{path}' is missing the required column '{validate_column}'. "
            f"Found: {dataset.column_names}."
        )

    return dataset


def is_prebuilt_dataset(path):
    """Whether `path` looks like a dataset materialised with `Dataset.save_to_disk`."""
    if not path:
        return False

    path = os.path.abspath(os.fspath(path))
    if not os.path.isdir(path):
        return False

    return all(os.path.isfile(os.path.join(path, name)) for name in _REQUIRED_FILES)


def resolve_prebuilt_dataset(root, *, train_subdir=TRAIN_SUBDIR, val_subdir=VAL_SUBDIR):
    """Return the `(train, validation)` directories inside `root`.

    Args:
        root: The `prebuilt_dataset_dir` spec, typically `<packed>/prebuilt`. `None` (or
            an empty string) means "no prebuilt dataset": the caller then builds from
            the raw shards instead.
        train_subdir: Name of the training subdirectory.
        val_subdir: Name of the validation subdirectory.

    Returns:
        `(train_path, val_path)`, or `None` when `root` is not set.

    Raises:
        ValueError: If `root` is set but does not hold a materialised dataset pair.
            Failing loudly keeps a typo from silently triggering an expensive rebuild
            from raw shards.
    """
    if not root:
        return None

    root = os.path.abspath(os.fspath(root))
    train_path = os.path.join(root, train_subdir)
    val_path = os.path.join(root, val_subdir)

    if not is_prebuilt_dataset(train_path) or not is_prebuilt_dataset(val_path):
        raise ValueError(
            f"`prebuilt_dataset_dir` is set to '{root}', but it does not hold a prebuilt "
            f"dataset. Expected both '{train_path}' and '{val_path}' to exist. Build it "
            "with distributed/prebuilt_dataset.py, or unset `prebuilt_dataset_dir` to "
            "build from the raw shards."
        )

    return train_path, val_path


class RandomTokenDataset(torch.utils.data.Dataset):
    """
    Lazy synthetic dataset for sanity-checking the training pipeline.

    Generates sequences on-the-fly so arbitrarily large sample counts
    don't blow up memory.  Each sequence is deterministic per (seed, idx),
    making it safe with shuffling, multiple workers, and multi-epoch runs.

    To let the model "learn" something (and thus show decreasing loss),
    a fraction of each sequence is filled with simple repeating patterns
    instead of pure noise.  The patterns are:

    * Copy-next: token at position *i* equals (token at *i-1* + 1) mod vocab_size.
    * Fixed bigram: a randomly chosen (A, B) pair that always appear together.

    The `pattern_ratio` controls what share of the sequence carries the
    learnable signal (default 30 %).
    """

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        seed: int = 0,
        pattern_ratio: float = 0.3,
        dtype: torch.dtype = torch.long,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed
        self.pattern_ratio = pattern_ratio
        self.dtype = dtype

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # Deterministic per (seed, idx), independent of iteration order.
        g = torch.Generator()
        g.manual_seed(self.seed + int(idx))

        input_ids = torch.randint(
            0,
            self.vocab_size,
            (self.seq_len,),
            generator=g,
            dtype=self.dtype,
        )

        # Overwrite a contiguous slice with a learnable pattern.
        pattern_len = max(2, int(self.seq_len * self.pattern_ratio))
        start = torch.randint(
            0,
            self.seq_len - pattern_len + 1,
            (1,),
            generator=g,
        ).item()

        # Alternate between two simple patterns based on idx parity.
        if idx % 2 == 0:
            # Copy-next: each token is (previous + 1) mod vocab_size.
            anchor = torch.randint(0, self.vocab_size, (1,), generator=g, dtype=self.dtype)
            input_ids[start : start + pattern_len] = (
                anchor + torch.arange(pattern_len, dtype=self.dtype)
            ) % self.vocab_size
        else:
            # Fixed bigram: repeating (A, B, A, B, ...) pair.
            a = torch.randint(0, self.vocab_size, (1,), generator=g, dtype=self.dtype).item()
            b = torch.randint(0, self.vocab_size, (1,), generator=g, dtype=self.dtype).item()
            for j in range(pattern_len):
                input_ids[start + j] = a if j % 2 == 0 else b

        return {"input_ids": input_ids}


def _load_sanity_check_datasets(args):
    """
    Create lazy synthetic datasets for sanity-checking the training pipeline.
    Returns a tuple of (train_dataset, val_dataset).
    """
    num_val = max(1, int(args.sanity_check_num_samples * 0.1))

    train_dataset = RandomTokenDataset(
        num_samples=args.sanity_check_num_samples,
        seq_len=args.max_position_embeddings,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    # Offset the seed so val samples are distinct from train samples.
    val_dataset = RandomTokenDataset(
        num_samples=num_val,
        seq_len=args.max_position_embeddings,
        vocab_size=args.vocab_size,
        seed=args.seed + args.sanity_check_num_samples,
    )

    return train_dataset, val_dataset


def _load_disk_datasets(args, logger=None, file_logger=None):
    """Load train and validation datasets from disk.

    Either `prebuilt_dataset_dir` points at a dataset materialised with
    `Dataset.save_to_disk` — whose Arrow files are then memory-mapped, so no `datasets`
    build happens on the compute nodes and there is no per-rank process pool — or the
    raw shards are read with the worker count capped by
    `max_num_proc_for_dataset_loading`. There is no in-between: nothing is guessed.

    A prebuilt dataset defines both splits, so `train_dataset_dir`/`val_dataset_dir`
    are bypassed when one is used.
    """
    dataset_type = args.dataset_type

    assert dataset_type in SUPPORTED_FORMATS, (
        f"Dataset type must be one of {SUPPORTED_FORMATS}, got '{dataset_type}'."
    )

    # A configured-but-invalid path raises instead of silently rebuilding from shards.
    prebuilt = resolve_prebuilt_dataset(args.prebuilt_dataset_dir)

    if prebuilt is not None:
        train_path, val_path = prebuilt

        if logger:
            logger.info(f"Loading prebuilt dataset (memory-mapped): {train_path}")
        if file_logger:
            file_logger.log_metadata(f"Using prebuilt dataset: {train_path}")

        train_dataset = load_dataset_from_disk(train_path)
        val_dataset = load_dataset_from_disk(val_path)
    else:
        # Collect training files.
        train_files = discover_dataset_files(args.train_dataset_dir, dataset_type)
        assert len(train_files) > 0, (
            f"No {dataset_type} files found in train_dataset_dir: {args.train_dataset_dir}"
        )

        if args.shuffle_dataset:
            if logger:
                logger.info(f"Shuffling enabled. Shuffling {len(train_files)} dataset files.")
            if file_logger:
                file_logger.log_metadata(
                    f"Shuffling enabled. Shuffling {len(train_files)} dataset files."
                )
            np.random.seed(args.seed)
            np.random.shuffle(train_files)

        # Validation files.
        val_files = discover_dataset_files(args.val_dataset_dir, dataset_type)
        assert len(val_files) > 0, (
            f"No {dataset_type} files found in val_dataset_dir: {args.val_dataset_dir}"
        )

        # The worker count is bounded inside `load_raw_dataset`: `datasets` defaults to
        # one process per shard, and every rank builds its own dataset.
        max_num_proc = args.max_num_proc_for_dataset_loading

        if logger:
            logger.info(
                f"Building dataset from raw shards (num_proc capped at {max_num_proc}: "
                f"{len(train_files)} training file(s), {len(val_files)} validation file(s))."
            )
        if file_logger:
            file_logger.log_metadata(
                f"Building dataset from raw shards with num_proc capped at {max_num_proc}."
            )

        train_dataset = load_raw_dataset(
            train_files,
            dataset_type,
            max_num_proc=max_num_proc,
            cache_dir=args.cache_dir,
        )

        val_dataset = load_raw_dataset(
            val_files,
            dataset_type,
            max_num_proc=max_num_proc,
            cache_dir=args.cache_dir,
        )

    if args.shuffle_dataset:
        train_dataset = train_dataset.shuffle(seed=args.seed)
        if logger:
            logger.info("Shuffling enabled. Shuffling indices.")

    # Validate that datasets contain the expected column.
    assert "input_ids" in train_dataset.column_names, (
        f"Training dataset must contain an 'input_ids' column. Found: {train_dataset.column_names}"
    )
    assert "input_ids" in val_dataset.column_names, (
        f"Validation dataset must contain an 'input_ids' column. Found: {val_dataset.column_names}"
    )

    train_dataset = train_dataset.with_format("torch")
    val_dataset = val_dataset.with_format("torch")

    return train_dataset, val_dataset


def prepare_dataloaders(
    args, tokenizer, world_size, rank, logger=None, file_logger=None, collate_fn=None
):
    """
    Build train and validation DataLoaders from the training arguments.
    It returns a DataLoaderBundle containing the dataloaders and metadata about the datasets.
    """

    if args.sanity_check:
        train_dataset, val_dataset = _load_sanity_check_datasets(args)
    else:
        train_dataset, val_dataset = _load_disk_datasets(
            args,
            logger=logger,
            file_logger=file_logger,
        )

    mask_token_ids = set()

    if collate_fn is None:
        # Always mask pad, eos, and bos tokens when they are defined in the tokenizer.
        if tokenizer is not None:
            for token_id in (
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
                tokenizer.bos_token_id,
            ):
                if token_id is not None:
                    mask_token_ids.add(token_id)

        # Add any user-specified additional token IDs to mask.
        if args.additional_mask_token_ids:
            mask_token_ids.update(args.additional_mask_token_ids)

        collate_fn = create_collate_fn(mask_token_ids=mask_token_ids)

        if logger:
            logger.info(f"Collate function will mask token IDs: {sorted(mask_token_ids)}")
        if file_logger:
            file_logger.log_metadata(
                f"Collate function will mask token IDs: {sorted(mask_token_ids)}"
            )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=args.shuffle_dataset,
        drop_last=False,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=collate_fn,
        batch_size=args.micro_batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers_for_dataloader,
        generator=generator,
        prefetch_factor=args.prefetch_factor,
    )

    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        collate_fn=collate_fn,
        batch_size=args.eval_micro_batch_size,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers_for_dataloader,
        prefetch_factor=args.prefetch_factor,
    )

    return DataLoaderBundle(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_sampler=train_sampler,
        num_train_samples=len(train_dataset),
        num_val_samples=len(val_dataset),
        mask_token_ids=mask_token_ids,
    )
