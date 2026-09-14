"""
Prebuilt Dataset Builder

Materialises a tokenized train/validation dataset to disk once, so that distributed
training jobs can memory-map it instead of rebuilding it on every rank at start-up.

Running this tool once on a login node (or in a small CPU-only job) produces the
directory that the trainer memory-maps, speeding up distributed training startup.

Output layout (the contract lives in `distributed/data_loading.py`, which is also the
module that reads a prebuilt dataset back at training time):

    <output_dir>/
        train/          # output of datasets.save_to_disk()
        validation/     # output of datasets.save_to_disk()
        .metadata       # build provenance (counts, worker counts, seed)

Usage:

    python prebuilt_dataset.py \\
        --train_dataset_dir data/packed_4096/gigaverbo_v2_3 \\
        --val_dataset_dir   data/packed_4096/validation \\
        --output_dir        data/packed_4096/prebuilt \\
        --dataset_type      parquet \\
        --num_proc          16 \\
        --shuffle --seed 1337

Then point a run at it from the specs file:

    prebuilt_dataset_dir: /data/packed_4096/prebuilt"

See `distributed/slurm/prebuilt_dataset.sh` for a SLURM job template.
"""

import argparse
import logging
import os
import sys

import numpy as np

# The dataset contract and helpers live in `data_loading.py`, next to this script and
# alongside the code that reads a prebuilt dataset back at training time.
from data_loading import (
    DEFAULT_MAX_NUM_PROC,
    SUPPORTED_FORMATS,
    TRAIN_SUBDIR,
    VAL_SUBDIR,
    discover_dataset_files,
    effective_num_proc,
    load_raw_dataset,
    save_dataset_to_disk,
    write_metadata,
)


def _get_logger(name="PrebuiltDataset"):
    """Create a console logger for the CLI."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
        logger.addHandler(handler)

    return logger


def build_prebuilt_dataset(
    train_dataset_dir,
    val_dataset_dir,
    output_dir,
    *,
    dataset_type="parquet",
    num_proc=DEFAULT_MAX_NUM_PROC,
    max_num_proc=DEFAULT_MAX_NUM_PROC,
    cache_dir=None,
    shuffle=False,
    seed=None,
    overwrite=False,
    validate_column="input_ids",
):
    """Build both splits from raw shards and materialise them to disk.

    Writes `<output_dir>/train`, `<output_dir>/validation` and `<output_dir>/.metadata`.


    Args:
        train_dataset_dir: Path or list of paths to the raw training shards. Repeat a
            directory to up-weight it.
        val_dataset_dir: Path or list of paths to the raw validation shards.
        output_dir: Root directory of the prebuilt dataset.
        dataset_type: `'parquet'` or `'jsonl'`.
        num_proc: Requested worker count per split (clamped by `max_num_proc`).
        max_num_proc: Hard upper bound on workers. Pass `None` to disable clamping.
        cache_dir: HuggingFace cache directory for the intermediate Arrow files.
        shuffle: Shuffle the list of training shards before building.
        seed: Random seed for `shuffle`.
        overwrite: Replace existing split directories.
        validate_column: Column that must exist in both splits.

    Returns:
        A dict describing the build (paths, example counts, effective worker counts).

    Raises:
        ValueError: If no shards are found for a split, or a required column is missing.
    """
    train_files = discover_dataset_files(train_dataset_dir, dataset_type)
    if not train_files:
        raise ValueError(f"No '.{dataset_type}' files found in: {train_dataset_dir}")

    train_num_proc = effective_num_proc(num_proc, len(train_files), max_num_proc)

    if shuffle:
        # Mirror `shuffle_dataset` in the trainer: same seeded permutation of paths.
        np.random.seed(seed)
        np.random.shuffle(train_files)

    train_dataset = load_raw_dataset(
        train_files,
        dataset_type,
        num_proc=train_num_proc,
        max_num_proc=max_num_proc,
        cache_dir=cache_dir,
    )
    if validate_column not in train_dataset.column_names:
        raise ValueError(
            f"Training dataset is missing the required column '{validate_column}'. "
            f"Found: {train_dataset.column_names}."
        )

    val_files = discover_dataset_files(val_dataset_dir, dataset_type)
    if not val_files:
        raise ValueError(f"No '.{dataset_type}' files found in: {val_dataset_dir}")

    val_num_proc = effective_num_proc(num_proc, len(val_files), max_num_proc)

    val_dataset = load_raw_dataset(
        val_files,
        dataset_type,
        num_proc=val_num_proc,
        max_num_proc=max_num_proc,
        cache_dir=cache_dir,
    )
    if validate_column not in val_dataset.column_names:
        raise ValueError(
            f"Validation dataset is missing the required column '{validate_column}'. "
            f"Found: {val_dataset.column_names}."
        )

    train_path = save_dataset_to_disk(
        train_dataset,
        os.path.join(output_dir, TRAIN_SUBDIR),
        overwrite=overwrite,
    )
    val_path = save_dataset_to_disk(
        val_dataset,
        os.path.join(output_dir, VAL_SUBDIR),
        overwrite=overwrite,
    )

    result = {
        "train_path": train_path,
        "validation_path": val_path,
        "num_train_files": len(train_files),
        "num_validation_files": len(val_files),
        "num_train_examples": len(train_dataset),
        "num_validation_examples": len(val_dataset),
        "num_proc_train": train_num_proc,
        "num_proc_validation": val_num_proc,
        "dataset_type": dataset_type,
        "shuffled": shuffle,
        "seed": seed,
    }

    write_metadata(output_dir, **result)

    return result


def main(args):
    logger = _get_logger()

    logger.info("Building prebuilt dataset...")
    result = build_prebuilt_dataset(
        train_dataset_dir=args.train_dataset_dir,
        val_dataset_dir=args.val_dataset_dir,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        num_proc=args.num_proc,
        max_num_proc=args.max_num_proc,
        cache_dir=args.cache_dir,
        shuffle=args.shuffle,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    logger.info(
        f"Train      : {result['num_train_examples']:,} examples "
        f"from {result['num_train_files']:,} file(s) "
        f"(num_proc={result['num_proc_train']})"
    )
    logger.info(
        f"Validation : {result['num_validation_examples']:,} examples "
        f"from {result['num_validation_files']:,} file(s) "
        f"(num_proc={result['num_proc_validation']})"
    )
    logger.info(f"Written    : {result['train_path']}")
    logger.info(f"             {result['validation_path']}")
    logger.info(
        f"Point the trainer at it with `prebuilt_dataset_dir: {os.path.abspath(args.output_dir)}`."
    )
    logger.info("Prebuilt dataset build complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O stuff
    io = parser.add_argument_group("Input / Output")
    io.add_argument(
        "--train_dataset_dir",
        nargs="+",
        required=True,
        help=(
            "One or more directories (or files) holding the raw training shards. "
            "Repeat a directory to up-weight it."
        ),
    )
    io.add_argument(
        "--val_dataset_dir",
        nargs="+",
        required=True,
        help="One or more directories (or files) holding the raw validation shards.",
    )
    io.add_argument(
        "--output_dir",
        required=True,
        help=(
            "Root directory for the prebuilt dataset. `train/` and `validation/` "
            "subdirectories are created inside it."
        ),
    )
    io.add_argument(
        "--dataset_type",
        choices=sorted(SUPPORTED_FORMATS),
        default="parquet",
        help="Format of the raw shards.",
    )

    # Build options
    build = parser.add_argument_group("Build options")
    build.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the training shard paths before building (mirrors `shuffle_dataset`).",
    )
    build.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used by --shuffle.",
    )
    build.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output directories.",
    )

    # Performance
    perf = parser.add_argument_group("Performance")
    perf.add_argument(
        "--num_proc",
        type=int,
        default=DEFAULT_MAX_NUM_PROC,
        help="Requested number of worker processes per split.",
    )
    perf.add_argument(
        "--max_num_proc",
        type=int,
        default=DEFAULT_MAX_NUM_PROC,
        help="Hard upper bound on worker processes. Smaller values are safer on shared filesystems.",
    )
    perf.add_argument(
        "--cache_dir",
        default=None,
        help="Cache directory for the intermediate HuggingFace Arrow files.",
    )

    main(parser.parse_args())
