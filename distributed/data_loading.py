"""
Dataset loading and DataLoader creation for the distributed trainers.

Provides:
    - create_collate_fn:          factory for the default collate function with token masking
    - prepare_dataloaders:        main entry point; returns fully configured dataloaders
    - DataLoaderBundle:           return type bundling dataloaders and metadata
"""
from dataclasses import dataclass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch

import glob
import os

from transformers import default_data_collator
import numpy as np
import datasets


# Map user-facing format names to HuggingFace datasets format names.
_FORMAT_MAP = {
    "parquet": "parquet",
    "jsonl": "json",
}

SUPPORTED_FORMATS = set(_FORMAT_MAP)


@dataclass
class DataLoaderBundle:
    """Everything the training loop needs from the data pipeline."""
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    train_sampler: DistributedSampler
    num_train_samples: int
    num_val_samples: int


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
    _mask_ids_tensor = torch.tensor(sorted(mask_token_ids), dtype=torch.long) if mask_token_ids else None

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


def _collect_dataset_files(paths, dataset_type):
    """Discover dataset files from a list of paths (files or directories)."""
    if isinstance(paths, str):
        paths = [paths]

    files = []
    for path in paths:
        if os.path.isfile(path) and path.endswith(f".{dataset_type}"):
            files.append(path)
        elif os.path.isdir(path):
            files += glob.glob(f"{path}/*.{dataset_type}")
    return sorted(files)


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
            0, self.vocab_size, (self.seq_len,), generator=g, dtype=self.dtype,
        )

        # Overwrite a contiguous slice with a learnable pattern.
        pattern_len = max(2, int(self.seq_len * self.pattern_ratio))
        start = torch.randint(
            0, self.seq_len - pattern_len + 1, (1,), generator=g,
        ).item()

        # Alternate between two simple patterns based on idx parity.
        if idx % 2 == 0:
            # Copy-next: each token is (previous + 1) mod vocab_size.
            anchor = torch.randint(0, self.vocab_size, (1,), generator=g, dtype=self.dtype)
            input_ids[start:start + pattern_len] = (
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
    """Load train and validation datasets from disk."""
    dataset_type = args.dataset_type

    assert dataset_type in SUPPORTED_FORMATS, (
        f"Dataset type must be one of {SUPPORTED_FORMATS}, got '{dataset_type}'."
    )

    # Collect training files.
    train_files = _collect_dataset_files(args.train_dataset_dir, dataset_type)
    assert len(train_files) > 0, (
        f"No {dataset_type} files found in train_dataset_dir: {args.train_dataset_dir}"
    )

    if args.shuffle_dataset:
        if logger:
            logger.info(f"Shuffling enabled. Shuffling {len(train_files)} dataset files.")
        if file_logger:
            file_logger.log_metadata(f"Shuffling enabled. Shuffling {len(train_files)} dataset files.")
        np.random.seed(args.seed)
        np.random.shuffle(train_files)

    # Validation files.
    val_files = sorted(glob.glob(f"{args.val_dataset_dir}/*.{dataset_type}"))
    assert len(val_files) > 0, (
        f"No {dataset_type} files found in val_dataset_dir: {args.val_dataset_dir}"
    )

    hf_format = _FORMAT_MAP[dataset_type]

    train_dataset = datasets.load_dataset(
        hf_format,
        data_files=train_files,
        split="train",
        num_proc=len(train_files),
        cache_dir=args.cache_dir,
    )

    val_dataset = datasets.load_dataset(
        hf_format,
        data_files=val_files,
        split="train",
        num_proc=len(val_files),
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


def prepare_dataloaders(args, tokenizer, world_size, rank, logger=None, file_logger=None, collate_fn=None):
    """
    Build train and validation DataLoaders from the training arguments.
    It returns a DataLoaderBundle containing the dataloaders and metadata about the datasets.
    """

    if args.sanity_check:
        train_dataset, val_dataset = _load_sanity_check_datasets(args)
    else:
        train_dataset, val_dataset = _load_disk_datasets(
            args, logger=logger, file_logger=file_logger,
        )

    if collate_fn is None:
        # Always mask pad, eos, and bos tokens when they are defined in the tokenizer.
        mask_token_ids = set()
        if tokenizer is not None:
            for token_id in (tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id):
                if token_id is not None:
                    mask_token_ids.add(token_id)

        # Add any user-specified additional token IDs to mask.
        if args.additional_mask_token_ids:
            mask_token_ids.update(args.additional_mask_token_ids)

        collate_fn = create_collate_fn(mask_token_ids=mask_token_ids)

        if logger:
            logger.info(f"Collate function will mask token IDs: {sorted(mask_token_ids)}")
        if file_logger:
            file_logger.log_metadata(f"Collate function will mask token IDs: {sorted(mask_token_ids)}")

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
    )
