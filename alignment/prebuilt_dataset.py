"""
Prebuilt Dataset Builder (alignment)

Materialises a raw (untokenised) alignment dataset to disk once, so that SFT / DPO /
GRPO / reward jobs memory-map it at start-up instead of rebuilding it from the raw
shards on every rank.

Usage:

    python prebuilt_dataset.py \\
        --train_dataset_dir data/sft \\
        --output_dir        data/sft_prebuilt \\
        --dataset_type      jsonl \\
        --test_size         0.02 \\
        --seed              42 \\
        --max_token_count   32768 \\
        --num_proc          64

Then point a trainer at it:

    python sft_trainer.py ... --prebuilt_dataset_dir data/sft_prebuilt

See `alignment/slurm/prebuilt_dataset.sh` for a SLURM job template.
"""

import argparse
import datetime
import math
import os

import datasets
from utils import (
    PREBUILT_TRAIN_SUBDIR,
    PREBUILT_VAL_SUBDIR,
    get_logger,
    has_column,
    load_training_dataset,
    resolve_prebuilt_dataset,
    save_dataset_to_disk,
    setup_distributed_state,
    write_metadata,
)

# Raw shard formats the alignment trainers can read (see their `--dataset_type`).
DATASET_TYPES = ("jsonl", "parquet")

# How the validation budget is spread over the training folders.
VAL_SPLIT_MODES = ("uniform", "global")

# Conservative default worker count, matching the trainers' `--num_proc` default.
DEFAULT_NUM_PROC = 16

# In `uniform` mode no folder is ever asked for more than this share of itself: taking
# more than half would leave that folder's remaining rows in the training minority. A
# folder that cannot supply its equal share is clamped to this and reported.
MAX_VAL_SHARE_PER_FOLDER = 0.5


def parse_test_size(value):
    """Parse `--test_size` as a row count (`5000`) or a fraction (`0.02`).

    `Dataset.train_test_split` accepts either, but argparse would force one type, so the
    raw string is parsed here: an integer literal is a row count, anything else is
    interpreted as a fraction.

    Args:
        value: The raw CLI value.

    Returns:
        `int` for a row count, `float` for a fraction.

    Raises:
        ValueError: If the value is neither a positive row count nor a fraction in
            (0, 1). argparse turns this into a proper CLI error.
    """
    text = str(value).strip()

    try:
        size = int(text)
    except ValueError:
        try:
            size = float(text)
        except ValueError:
            raise ValueError(
                f"--test_size must be a row count (e.g. 5000) or a fraction (e.g. 0.02), "
                f"got '{value}'."
            ) from None

    if isinstance(size, float):
        if not 0.0 < size < 1.0:
            raise ValueError(
                f"A fractional --test_size must be in (0, 1), got '{value}'. Use a value "
                ">= 1 to give a row count instead."
            )
    elif size < 1:
        raise ValueError(f"A row-count --test_size must be >= 1, got '{value}'.")

    return size


def uniform_val_quotas(split_sizes, test_size):
    """Distribute a validation budget uniformly over the input folders.

    Args:
        split_sizes: Row count of each folder, in input order.
        test_size: Total validation size. An `int` is a row count, a `float` a fraction
            of the concatenated rows (rounded up, as `Dataset.train_test_split` does).

    Returns:
        `(quotas, budget)`: the per-folder validation row counts, aligned with
        `split_sizes`, and the total budget they were drawn from.

        The quotas sum to AT MOST `budget`. A folder too small for its equal share is
        clamped to `MAX_VAL_SHARE_PER_FOLDER` of itself, and the rows it could not supply
        are deliberately NOT pushed onto the other folders: concentrating them in one
        folder is exactly the imbalance this mode exists to avoid. The shortfall is
        visible in the returned quotas (and logged by the caller) instead of being
        hidden.
    """
    num_folders = len(split_sizes)
    if num_folders == 0:
        return [], 0

    total_rows = sum(split_sizes)
    budget = math.ceil(test_size * total_rows) if isinstance(test_size, float) else int(test_size)

    # One row per folder while the budget lasts, then one more each: a budget smaller
    # than the folder count simply leaves the last folders without a share.
    base, remainder = divmod(budget, num_folders)

    quotas = []
    for index, size in enumerate(split_sizes):
        quota = base + (1 if index < remainder else 0)
        quotas.append(min(quota, int(size * MAX_VAL_SHARE_PER_FOLDER)))

    return quotas, budget


def filter_by_token_count(
    dataset,
    column,
    max_tokens,
    *,
    num_proc=None,
    label="dataset",
    logger=None,
    allow_empty=False,
):
    """Drop rows whose `column` (a token count) exceeds `max_tokens`.

    Args:
        dataset: The loaded dataset to filter.
        column: Name of the column holding the token count.
        max_tokens: Rows with MORE tokens than this are removed. `None` disables the
            filter and returns `dataset` untouched.
        num_proc: Worker processes for the filter.
        label: What `dataset` is, for the log lines and error messages (e.g. a folder
            path).
        logger: Optional logger.
        allow_empty: Whether removing EVERY row is acceptable. `False` (the default)
            raises instead, because an empty split is always a build mistake; the
            uniform per-folder path passes `True` so it can skip such a folder instead.

    Returns:
        `(dataset, num_dropped)`.

    Raises:
        ValueError: If `max_tokens` is set but `column` is missing, or if nothing
            survives and `allow_empty` is False.
    """
    if max_tokens is None:
        return dataset, 0

    if not has_column(dataset, column):
        raise ValueError(
            f"The token-count column '{column}' is not in {label}. Found: "
            f"{dataset.column_names}. Point --token_count_column at the right column, "
            "or drop --max_token_count."
        )

    # Guard the comparison below: filtering on a string/list column would otherwise fail
    # deep inside `datasets` with an opaque TypeError instead of naming the mistake.
    numeric = ("int", "uint", "float")
    feature = dataset.features.get(column)
    if feature is not None and not (
        isinstance(feature, datasets.Value) and feature.dtype.startswith(numeric)
    ):
        raise ValueError(
            f"The token-count column '{column}' in {label} holds {feature}, not numbers. "
            "Point --token_count_column at the column holding the token count."
        )

    before = len(dataset)
    # `None` counts are KEPT: "unknown" is not "over the limit", and silently dropping
    # them would discard rows the filter was never asked to touch. A run that drops
    # nothing because the column is unpopulated shows up as "dropped 0" in the log.
    dataset = dataset.filter(
        lambda example: example[column] is None or example[column] <= max_tokens,
        num_proc=num_proc,
        desc=f"Dropping rows over {max_tokens:,} tokens ({label})",
    )
    dropped = before - len(dataset)

    if logger:
        logger.info(
            f"  {label}: kept {len(dataset):,} of {before:,} rows "
            f"(dropped {dropped:,} over {max_tokens:,} tokens)"
        )

    if len(dataset) == 0 and not allow_empty:
        raise ValueError(
            f"Filtering {label} on '{column}' <= {max_tokens:,} removed every row. "
            "Raise --max_token_count, or drop it."
        )

    return dataset, dropped


def _concatenate(parts):
    """Concatenate the per-folder datasets back into a single `Dataset`."""
    if len(parts) == 1:
        return parts[0]
    return datasets.concatenate_datasets(parts)


def _as_dirs(paths):
    """Normalise a path or a list of paths to a list of paths."""
    if isinstance(paths, list | tuple):
        return list(paths)
    return [paths]


def build_prebuilt_dataset(
    train_dataset_dir,
    output_dir,
    *,
    val_dataset_dir=None,
    test_size=None,
    val_split_mode="uniform",
    seed=42,
    dataset_type="jsonl",
    num_proc=DEFAULT_NUM_PROC,
    max_num_proc=DEFAULT_NUM_PROC,
    cache_dir=None,
    validate_column=None,
    max_token_count=None,
    token_count_column="token_count",
    overwrite=False,
    logger=None,
):
    """Build the train/validation pair from raw shards and materialise it to disk.

    Writes `<output_dir>/train`, `<output_dir>/validation` and `<output_dir>/.metadata`.

    Args:
        train_dataset_dir: Path or list of paths to the raw training shards. Repeat a
            directory to include its shards more than once.
        output_dir: Root directory of the prebuilt dataset.
        val_dataset_dir: Path or list of paths to the raw validation shards. The
            alternative to `test_size`; the two are mutually exclusive.
        test_size: Size of the validation split carved out of the training data. An
            `int` is a row count, a `float` a fraction in (0, 1). Required unless
            `val_dataset_dir` is given.
        val_split_mode: How that budget is spent. `'uniform'` (the default) gives every
            training folder the same number of validation rows; `'global'` concatenates
            the folders first and draws one random sample, so a folder's share follows
            its size. Ignored when `val_dataset_dir` is given, and degraded to `'global'`
            when there is only one training folder to spread over.
        seed: Seed for `train_test_split`.
        dataset_type: `'jsonl'` or `'parquet'`.
        num_proc: Requested worker count per split. Clamped by `max_num_proc`, and by the
            loader to one worker per shard.
        max_num_proc: Hard upper bound on workers.
        cache_dir: HuggingFace cache directory for the intermediate Arrow files.
        validate_column: Optional column that must exist in BOTH splits.
        max_token_count: Optional row filter applied BEFORE the splits are built: rows
            whose `token_count_column` is greater than this are dropped from both the
            training data and the validation set. `None` (the default) keeps every row.
            A folder emptied by the filter is skipped in `uniform` mode; a build that
            keeps nothing at all is an error.
        token_count_column: Column holding the token count used by `max_token_count`.
        overwrite: Replace existing split directories.
        logger: Optional logger used to report progress.

    Returns:
        A dict describing the build (paths, example counts, split config).

    Raises:
        ValueError: If neither or both of `test_size` and `val_dataset_dir` is given, no
            shards are found, a required column is missing, or `uniform` sampling cannot
            allocate a single validation row (every folder holds at most one row).
        FileExistsError: If a split directory already exists and `overwrite` is False.
    """
    if test_size is not None and val_dataset_dir:
        raise ValueError(
            "--test_size and --val_dataset_dir are mutually exclusive: either carve the "
            "validation split out of the training data, or give it explicitly."
        )
    if test_size is None and not val_dataset_dir:
        raise ValueError(
            "No validation split configured. Pass --test_size (carved out of the training "
            "data) or --val_dataset_dir (an explicit held-out set): a prebuilt dataset "
            "always holds both <root>/train and <root>/validation."
        )

    if logger is None:
        logger = get_logger("PrebuiltDataset")

    output_dir = os.path.abspath(os.fspath(output_dir))
    train_path = os.path.join(output_dir, PREBUILT_TRAIN_SUBDIR)
    val_path = os.path.join(output_dir, PREBUILT_VAL_SUBDIR)

    # Fail before loading anything: a wrong --output_dir, or a forgotten --overwrite,
    # should not cost a full build first.
    existing = [path for path in (train_path, val_path) if os.path.exists(path)]
    if existing and not overwrite:
        raise FileExistsError(
            f"Already exists: {', '.join(existing)}. Pass --overwrite to replace it, or "
            "point --output_dir somewhere else."
        )

    if val_split_mode not in VAL_SPLIT_MODES:
        raise ValueError(
            f"Unsupported val_split_mode '{val_split_mode}'. Expected one of: "
            f"{sorted(VAL_SPLIT_MODES)}."
        )

    if max_token_count is not None:
        max_token_count = int(max_token_count)
        if max_token_count < 1:
            raise ValueError(f"--max_token_count must be >= 1 (got {max_token_count}).")

    # Rows removed by `--max_token_count`, recorded in the build metadata below.
    dropped_total = 0

    num_proc = min(int(num_proc), int(max_num_proc))

    state, _ = setup_distributed_state(logger)

    train_dirs = _as_dirs(train_dataset_dir)
    val_dirs = _as_dirs(val_dataset_dir) if val_dataset_dir else []
    val_per_folder = None

    # `uniform` only means something when there is more than one folder to spread the
    # budget over; with a single folder it is exactly the global draw.
    uniform = val_split_mode == "uniform" and len(train_dirs) > 1 and not val_dirs

    if val_dirs:
        # Explicit held-out set: both sides are materialised as they are, no splitting.
        effective_mode = "val_dataset_dir"
        logger.info(f"Loading raw {dataset_type} shards: {val_dirs}")
        val_dataset = load_training_dataset(
            val_dirs, dataset_type, num_proc, cache_dir, state, logger=logger
        )
        val_dataset, dropped = filter_by_token_count(
            val_dataset,
            token_count_column,
            max_token_count,
            num_proc=num_proc,
            label="validation shards",
            logger=logger,
        )
        dropped_total += dropped

        logger.info(f"Loading raw {dataset_type} shards: {train_dirs}")
        train_dataset = load_training_dataset(
            train_dirs, dataset_type, num_proc, cache_dir, state, logger=logger
        )
        train_dataset, dropped = filter_by_token_count(
            train_dataset,
            token_count_column,
            max_token_count,
            num_proc=num_proc,
            label="train shards",
            logger=logger,
        )
        dropped_total += dropped
    elif uniform:
        # Load each folder on its own: a single concatenated dataset has already lost the
        # folder boundaries the uniform draw needs.
        effective_mode = "uniform"
        logger.info(f"Loading raw {dataset_type} shards from {len(train_dirs)} folders")
        parts = [
            load_training_dataset([path], dataset_type, num_proc, cache_dir, state, logger=logger)
            for path in train_dirs
        ]

        # Filter BEFORE the quotas are computed, so a folder's share is based on the
        # rows that will actually be used (a filtered-away row must not reserve a
        # validation slot). A folder the filter empties is skipped and reported.
        if max_token_count is not None:
            remaining = []
            for path, part in zip(train_dirs, parts, strict=False):
                part, dropped = filter_by_token_count(
                    part,
                    token_count_column,
                    max_token_count,
                    num_proc=num_proc,
                    label=os.fspath(path),
                    logger=logger,
                    allow_empty=True,
                )
                dropped_total += dropped
                if len(part) == 0:
                    logger.warning(
                        f"  {os.fspath(path)}: every row is over {max_token_count:,} "
                        "tokens; the folder is skipped"
                    )
                    continue
                remaining.append(part)

            if not remaining:
                raise ValueError(
                    f"--max_token_count={max_token_count} removed every row from all "
                    f"{len(train_dirs)} folders."
                )
            parts = remaining

        quotas, budget = uniform_val_quotas([len(part) for part in parts], test_size)
        equal_share = budget // len(parts)
        logger.info(
            f"Carving the validation split out of the training data "
            f"(test_size={test_size}, seed={seed}, {budget:,} rows over "
            f"{len(parts)} folders)"
        )

        train_parts, val_parts = [], []
        val_per_folder = {}
        for path, part, quota in zip(train_dirs, parts, quotas, strict=False):
            val_per_folder[os.fspath(path)] = quota

            clamped = " (clamped: less than its share)" if quota < equal_share else ""
            logger.info(f"  {os.fspath(path)}: {quota:,} of {len(part):,} rows{clamped}")

            if quota <= 0:
                train_parts.append(part)
                continue

            split = part.train_test_split(test_size=quota, seed=seed, shuffle=True)
            train_parts.append(split["train"])
            val_parts.append(split["test"])

        if not val_parts:
            raise ValueError(
                f"Uniform sampling could not allocate a single validation row: each of the "
                f"{len(parts)} folders holds at most one row. Give an explicit validation "
                "set with --val_dataset_dir, or use a larger --test_size."
            )

        train_dataset = _concatenate(train_parts)
        val_dataset = _concatenate(val_parts)
    else:
        # `global`: one draw over everything, so a folder's share follows its size.
        # `train_test_split` shuffles by default, so shard order does not bias the
        # validation set; `seed` makes the draw reproducible.
        effective_mode = "global"
        if val_split_mode != "global":
            logger.info(
                "A single training folder leaves nothing to spread the validation budget "
                "over: falling back to a global draw."
            )
        logger.info(f"Loading raw {dataset_type} shards: {train_dirs}")
        train_dataset = load_training_dataset(
            train_dirs, dataset_type, num_proc, cache_dir, state, logger=logger
        )
        train_dataset, dropped = filter_by_token_count(
            train_dataset,
            token_count_column,
            max_token_count,
            num_proc=num_proc,
            label="train shards",
            logger=logger,
        )
        dropped_total += dropped
        logger.info(
            f"Carving the validation split out of the training data "
            f"(test_size={test_size}, seed={seed}, global draw)"
        )
        split = train_dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)
        train_dataset, val_dataset = split["train"], split["test"]

    if validate_column is not None:
        for label, dataset in (("train", train_dataset), ("validation", val_dataset)):
            if validate_column not in dataset.column_names:
                raise ValueError(
                    f"The {label} split is missing the required column "
                    f"'{validate_column}'. Found: {dataset.column_names}."
                )

    train_path = save_dataset_to_disk(train_dataset, train_path, overwrite=overwrite)
    val_path = save_dataset_to_disk(val_dataset, val_path, overwrite=overwrite)

    result = {
        "train_path": train_path,
        "validation_path": val_path,
        "num_train_examples": len(train_dataset),
        "num_validation_examples": len(val_dataset),
        "val_split_mode": effective_mode,
        "test_size": test_size,
        "seed": seed,
        "val_per_folder": (
            " ".join(f"{path}={quota}" for path, quota in val_per_folder.items())
            if val_per_folder
            else None
        ),
        "dataset_type": dataset_type,
        "validate_column": validate_column,
        "max_token_count": max_token_count,
        "token_count_column": token_count_column if max_token_count is not None else None,
        "num_dropped_over_max_tokens": dropped_total,
        "num_proc": num_proc,
        "train_dataset_dir": " ".join(os.fspath(path) for path in train_dirs),
        "val_dataset_dir": " ".join(os.fspath(path) for path in val_dirs) if val_dirs else None,
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    write_metadata(output_dir, **result)

    # Cheap self-check: what was just written must be readable by the same resolver the
    # trainers use, so a builder/reader layout drift fails here instead of 40 minutes
    # into a GPU job.
    resolve_prebuilt_dataset(output_dir)

    return result


def main(args):
    logger = get_logger("PrebuiltDataset")

    logger.info("Building prebuilt alignment dataset...")
    result = build_prebuilt_dataset(
        train_dataset_dir=args.train_dataset_dir,
        output_dir=args.output_dir,
        val_dataset_dir=args.val_dataset_dir,
        test_size=args.test_size,
        val_split_mode=args.val_split_mode,
        seed=args.seed,
        dataset_type=args.dataset_type,
        num_proc=args.num_proc,
        max_num_proc=args.max_num_proc,
        cache_dir=args.cache_dir,
        validate_column=args.validate_column,
        max_token_count=args.max_token_count,
        token_count_column=args.token_count_column,
        overwrite=args.overwrite,
        logger=logger,
    )

    logger.info(f"Train      : {result['num_train_examples']:,} examples -> {result['train_path']}")
    logger.info(
        f"Validation : {result['num_validation_examples']:,} examples "
        f"({result['val_split_mode']}) -> {result['validation_path']}"
    )
    logger.info(
        f"Point a trainer at it with `--prebuilt_dataset_dir {os.path.abspath(args.output_dir)}`."
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
            "One or more directories (or files) holding the raw training shards. Repeat a "
            "directory to include its shards more than once."
        ),
    )
    io.add_argument(
        "--val_dataset_dir",
        nargs="+",
        default=None,
        help=(
            "One or more directories (or files) holding an explicit validation set. The "
            "alternative to --test_size; the two are mutually exclusive."
        ),
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
        choices=sorted(DATASET_TYPES),
        default="jsonl",
        help="Format of the raw shards.",
    )

    # Validation split
    split = parser.add_argument_group("Validation split")
    split.add_argument(
        "--test_size",
        type=parse_test_size,
        default=None,
        help=(
            "Size of the validation split carved out of the training data with "
            "Dataset.train_test_split. A value in (0, 1) is a fraction, a value >= 1 a row "
            "count. Required unless --val_dataset_dir is given."
        ),
    )
    split.add_argument(
        "--val_split_mode",
        choices=sorted(VAL_SPLIT_MODES),
        default="uniform",
        help=(
            "How --test_size is spread over --train_dataset_dir. 'uniform' gives every "
            "folder the same number of validation rows (a folder too small for its share "
            "is clamped to half of itself and reported); 'global' concatenates the folders "
            "first and draws one random sample, so a folder's share follows its size. "
            "Ignored with --val_dataset_dir."
        ),
    )
    split.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/validation split.",
    )

    # Build options
    build = parser.add_argument_group("Build options")
    build.add_argument(
        "--validate_column",
        default=None,
        help=(
            "Optional column that must be present in BOTH splits (e.g. 'messages' for SFT, "
            "'chosen' for DPO). Omit to skip the check."
        ),
    )
    build.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output directories.",
    )

    # Filtering
    filt = parser.add_argument_group("Filtering")
    filt.add_argument(
        "--max_token_count",
        type=int,
        default=None,
        help=(
            "Drop rows whose token count exceeds this value (e.g. 32768) BEFORE the "
            "splits are built, so over-long samples reach neither the training data nor "
            "the validation set. Omit to keep every row."
        ),
    )
    filt.add_argument(
        "--token_count_column",
        default="token_count",
        help="Column holding the token count read by --max_token_count.",
    )

    # Performance
    perf = parser.add_argument_group("Performance")
    perf.add_argument(
        "--num_proc",
        type=int,
        default=DEFAULT_NUM_PROC,
        help=(
            "Requested number of worker processes per split. Also clamped by the loader to "
            "one worker per shard."
        ),
    )
    perf.add_argument(
        "--max_num_proc",
        type=int,
        default=DEFAULT_NUM_PROC,
        help="Hard upper bound on worker processes. Smaller values are safer on shared filesystems.",
    )
    perf.add_argument(
        "--cache_dir",
        default=None,
        help="Cache directory for the intermediate HuggingFace Arrow files.",
    )

    main(parser.parse_args())
