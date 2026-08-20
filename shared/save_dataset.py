"""Shared dataset saving utilities."""

import os

import numpy as np


def save_dataset(dataset, output_dir, output_type, tokens_per_chunk, token_count, *, n_chunks=None):
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
