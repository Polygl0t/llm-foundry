"""
Validation Split Creation

Creates a validation split by extracting a specified number of samples from a set of training
data files and consolidating them into a separate validation file. Useful for preparing
train/validation splits from pre-tokenized datasets.

Output:
- Single validation file with extracted samples
- Updated source files with remaining training samples (only the files that actually
  contributed; the rest are left untouched)
- .metadata file containing validation split statistics

The `n_samples` rows are apportioned across the selected files **in proportion to their
row counts**, using the largest-remainder method: each file takes
`floor(n_samples * rows_i / total)` and the left-over units go to the files with the
largest fractional parts.  Every file is therefore within one row of its exact quota.

Usage:
    python make_validation_split.py \\
        --input_dirs data/train_chunks \\
        --output_dir data/validation \\
        --input_type parquet \\
        --output_file validation_split \\
        --n_samples 20000 \\
        --n_files 10

    # Multiple source directories, spread over 16 worker processes:
    python make_validation_split.py \\
        --input_dirs data/train_en data/train_de data/train_fr \\
        --output_dir data/validation \\
        --input_type parquet \\
        --n_samples 20000 \\
        --num_proc 16
"""

import argparse
import os
import random
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor

import datasets
import pyarrow.parquet as pq
from utils import get_logger, list_matching_files, read_metadata

logger = get_logger("MakeValidationSplit")


def get_files_from_dirs(input_dirs, input_type, n_files=None):
    """Get files from one or more folders, optionally randomly selecting n_files total."""
    pattern = "*.parquet" if input_type == "parquet" else "*.jsonl"
    all_files: list[str] = []
    for d in input_dirs:
        found = list_matching_files(d, pattern)
        if not found:
            raise FileNotFoundError(f"No {pattern} files found in '{d}'.")
        all_files.extend(found)
    all_files = sorted(set(all_files))

    if n_files is not None and n_files < len(all_files):
        all_files = random.sample(all_files, n_files)

    return sorted(all_files)


def _row_count(task):
    """Worker: row count of one file. Parquet is read from the footer only."""
    path, input_type = task
    if input_type == "parquet":
        return pq.ParquetFile(path).metadata.num_rows
    datasets.disable_caching()
    return len(datasets.load_dataset(input_type, data_files=path, split="train"))


def allocate_samples(lengths, n_samples):
    """Apportion `n_samples` across files in proportion to their row counts.

    Largest-remainder method: a first pass gives every file
    `floor(n_samples * length_i / total)`, then the left-over units go to the files
    with the largest fractional parts.  Every file ends up within one row of its exact
    quota, so no single file absorbs a disproportionate share.

    Args:
        lengths: Row count of each candidate file.
        n_samples: Total rows to take.

    Returns:
        Per-file row counts, summing to exactly `n_samples`.

    Raises:
        ValueError: If `n_samples` is negative or exceeds the total number of rows.
    """
    total = sum(lengths)
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0.")
    if n_samples > total:
        raise ValueError("n_samples is greater than total number of rows in all files.")
    if n_samples == 0 or total == 0:
        return [0] * len(lengths)

    quotas = [n_samples * length / total for length in lengths]
    take = [min(int(quota), length) for quota, length in zip(quotas, lengths, strict=False)]

    leftover = n_samples - sum(take)
    if leftover > 0:
        # Largest fractional parts first; ties broken by index so it stays deterministic.
        order = sorted(range(len(lengths)), key=lambda i: (-(quotas[i] - int(quotas[i])), i))
        for i in order:
            if leftover == 0:
                break
            if take[i] < lengths[i]:
                take[i] += 1
                leftover -= 1
    if leftover > 0:
        raise ValueError("Could not apportion n_samples across the given files.")

    return take


def _carve_one(task):
    """Worker: move the first `n_remove` rows of one file into the validation set.

    The source file is rewritten without those rows and the removed rows go to a
    per-file staging file that the parent process concatenates.  Returns `None`,
    touching nothing at all, when the file contributes no rows.

    Args:
        task: `(index, path, n_remove, input_type, tmp_dir)`.  `index` is the file's
            position in the selected list; it is what keeps the staging filenames
            unique, because shards in different `--input_dirs` routinely share a
            basename and would otherwise overwrite each other.
    """
    index, path, n_remove, input_type, tmp_dir = task
    if n_remove <= 0:
        return None

    # Each file is read once and immediately rewritten, so the HuggingFace cache buys
    # nothing here -- and several workers writing into one cache directory is only a
    # source of contention.
    datasets.disable_caching()
    dataset = datasets.load_dataset(input_type, data_files=path, split="train")

    removed = dataset.select(range(n_remove))  # head rows -> validation
    kept = dataset.select(range(n_remove, len(dataset)))  # the rest stays in training

    if input_type == "parquet":
        kept.to_parquet(path)
        tmp_path = os.path.join(tmp_dir, f"{index:05d}.removed.parquet")
        removed.to_parquet(tmp_path)
    else:
        kept.to_json(path)
        tmp_path = os.path.join(tmp_dir, f"{index:05d}.removed.jsonl")
        removed.to_json(tmp_path)
    return tmp_path


def _block_size(files, input_type):
    """Token length of the first row of the first file (0 when it cannot be read)."""
    if not files:
        return 0
    datasets.disable_caching()
    dataset = datasets.load_dataset(input_type, data_files=files[0], split="train")
    if len(dataset) == 0 or "input_ids" not in dataset.features:
        return 0
    return len(dataset[0]["input_ids"])


def main(
    input_dirs,
    output_dir,
    input_type,
    output_file,
    n_samples,
    n_files=None,
    num_proc=1,
    seed=42,
):
    """
    Removes n_samples rows (as evenly as possible) from randomly selected files across
    one or more `input_dirs`, saves the removed rows to a single file, and overwrites
    the source files with the remaining rows.
    """
    random.seed(seed)
    datasets.disable_caching()  # one-shot reads: the cache only costs writes and races

    # Get files from all input folders
    files = get_files_from_dirs(input_dirs, input_type, n_files)
    logger.info(
        f"Selected {len(files)} files for sampling from {len(input_dirs)} director{'y' if len(input_dirs) == 1 else 'ies'}"
    )

    # Read tokenizer name from the first source folder metadata
    source_metadata_path = os.path.join(input_dirs[0], ".metadata")
    source_metadata = read_metadata(source_metadata_path) or {}
    tokenizer_name = source_metadata.get("Tokenizer", "")

    # Read existing metadata from output dir if it exists
    output_metadata_path = os.path.join(output_dir, ".metadata")
    existing_metadata = read_metadata(output_metadata_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    num_proc = max(1, min(int(num_proc or 1), len(files)))

    # Row counts come from the parquet footers, so this pass reads no column data.
    with ProcessPoolExecutor(max_workers=num_proc) as pool:
        lengths = list(pool.map(_row_count, [(f, input_type) for f in files]))
    total_rows = sum(lengths)

    samples_to_remove = allocate_samples(lengths, n_samples)
    contributing = [n for n in samples_to_remove if n > 0]
    logger.info(
        f"Apportioned {n_samples:,} of {total_rows:,} rows to {len(contributing)} of "
        f"{len(files)} file(s); per contributing file min "
        f"{min(contributing) if contributing else 0:,}, max "
        f"{max(contributing) if contributing else 0:,}"
    )

    # Staging stays on the output filesystem so the final write is a local move.
    tmp_dir = tempfile.mkdtemp(prefix=".val-split-staging-", dir=output_dir)
    try:
        tasks = [
            (index, path, n_remove, input_type, tmp_dir)
            for index, (path, n_remove) in enumerate(zip(files, samples_to_remove, strict=False))
        ]
        with ProcessPoolExecutor(max_workers=num_proc) as pool:
            tmp_paths = [p for p in pool.map(_carve_one, tasks) if p]

        if not tmp_paths:
            logger.warning("No rows were removed; nothing was written.")
            return

        removed_tables = [
            datasets.load_dataset(input_type, data_files=p, split="train") for p in tmp_paths
        ]
        concat = datasets.concatenate_datasets(removed_tables)
        logger.info(str(concat))

        # Get the block size from the first source file. Calculate the length of the
        # first entry in 'input_ids'.
        block_size = _block_size(files, input_type)
        sample_count = len(concat)
        token_count = sample_count * block_size
        logger.info(f"Number of samples: {sample_count:,}")
        logger.info(f"Number of tokens: {token_count:,}")

        if input_type == "parquet":
            concat.to_parquet(os.path.join(output_dir, output_file + ".parquet"))
        else:
            concat.to_json(os.path.join(output_dir, output_file + ".jsonl"))

        # Combine with existing metadata if present
        prev_samples = int(existing_metadata.get("Samples", 0) or 0)
        prev_tokens = int(existing_metadata.get("Tokens", 0) or 0)
        prev_chunks = int(existing_metadata.get("Chunks", 0) or 0)
        total_samples = prev_samples + sample_count
        total_tokens = prev_tokens + token_count
        total_chunks = prev_chunks + 1

        # Write metadata about the validation split
        with open(output_metadata_path, "w") as meta_file:
            meta_file.write(f"Samples: {total_samples}\n")
            meta_file.write(f"Tokens: {total_tokens}\n")
            meta_file.write(f"Tokens per chunk: {total_tokens // total_chunks}\n")
            meta_file.write(f"Block size: {block_size}\n")
            meta_file.write(f"Chunks: {total_chunks}\n")
            meta_file.write(f"Tokenizer: {tokenizer_name}\n")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing input files to sample from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to save the validation split and metadata.",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="parquet",
        choices=["parquet", "json"],
        help="Input file type.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="validation_split",
        help="Filename for the validation split file.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20000,
        help="Total number of samples to remove for validation split.",
    )
    parser.add_argument(
        "--n_files",
        type=int,
        default=None,
        help="Total number of files to randomly select across all input directories (default: use all files).",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Worker processes used to read row counts and rewrite the source files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for file selection.")
    args = parser.parse_args()

    main(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        input_type=args.input_type,
        output_file=args.output_file,
        n_samples=args.n_samples,
        n_files=args.n_files,
        num_proc=args.num_proc,
        seed=args.seed,
    )
