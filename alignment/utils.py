"""
Shared utilities for alignment training scripts.
"""

import glob
import logging
import os
import shutil
import sys
from pathlib import Path

import accelerate
import datasets
import transformers

_REPO_ROOT = Path(__file__).resolve()
while not (_REPO_ROOT / "pyproject.toml").exists() and _REPO_ROOT.parent != _REPO_ROOT:
    _REPO_ROOT = _REPO_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.logging import get_logger as get_logger  # noqa: E402


def setup_distributed_state(logger: logging.Logger):
    """Initialize distributed training state.

    Args:
        logger: Logger instance used to report the process state.

    Returns:
        state: accelerate.PartialState instance.
        master_process: True if this is the main (rank-0) process.
    """
    state = accelerate.PartialState()
    master_process = int(state.process_index) == 0
    if master_process:
        logger.info(str(state))
    return state, master_process


#############################################
# Prebuilt datasets
#############################################
# A prebuilt dataset is the `<root>/train` + `<root>/validation` pair produced by
# materialising a `datasets.Dataset` with `Dataset.save_to_disk`, so training jobs
# memory-map Arrow instead of rebuilding from raw shards on every rank.
#
# This is deliberately a SELF-CONTAINED copy of that contract rather than an import
# from `distributed/data_loading.py`: the alignment and pretraining pipelines are
# kept independent on purpose, so the alignment scripts do not pick up an import
# edge onto the pretraining package. The two layouts are the same, so a directory
# built for either side can be read by the other -- but if one side ever changes
# the layout, the other must be updated by hand.

PREBUILT_TRAIN_SUBDIR = "train"
PREBUILT_VAL_SUBDIR = "validation"

# Files `Dataset.save_to_disk()` leaves behind, enough for a cheap "is this a
# prebuilt dataset?" check before attempting to load.
_PREBUILT_REQUIRED_FILES = ("dataset_info.json", "state.json")


def is_prebuilt_dataset(path):
    """Whether `path` looks like a dataset materialised with `Dataset.save_to_disk`."""
    if not path:
        return False

    path = os.path.abspath(os.fspath(path))
    if not os.path.isdir(path):
        return False

    return all(os.path.isfile(os.path.join(path, name)) for name in _PREBUILT_REQUIRED_FILES)


def resolve_prebuilt_dataset(
    root, *, train_subdir=PREBUILT_TRAIN_SUBDIR, val_subdir=PREBUILT_VAL_SUBDIR
):
    """Return the `(train, validation)` directories inside `root`.

    Args:
        root: The prebuilt root. `None` (or an empty string) means "no prebuilt
            dataset": the caller then builds from the raw shards instead.
        train_subdir: Name of the training subdirectory.
        val_subdir: Name of the validation subdirectory.

    Returns:
        `(train_path, val_path)`, or None when `root` is not set.

    Raises:
        ValueError: If `root` is set but does not hold a materialised dataset pair.
            Failing loudly keeps a typo from silently triggering an expensive
            rebuild from the raw shards.
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
            "with a prebuilt-dataset builder, or unset `prebuilt_dataset_dir` to build "
            "from the raw shards."
        )

    return train_path, val_path


def load_dataset_from_disk(path, *, validate_column=None):
    """Read a dataset materialised with `Dataset.save_to_disk`.

    Args:
        path: Directory holding the output of `Dataset.save_to_disk()`.
        validate_column: Optional column that must be present (e.g. `input_ids`).

    Returns:
        The loaded `datasets.Dataset`.

    Raises:
        FileNotFoundError: If `path` is not an existing directory.
        ValueError: If `path` holds multiple splits (one split per path is expected)
            or is missing `validate_column`.
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

    The writing half of the prebuilt contract, for the builder script that produces
    the `<root>/train` + `<root>/validation` pair this module reads back.

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


def has_column(dataset, column):
    """Whether `column` is present in `dataset`.

    Accepts a bare `datasets.Dataset` or a `datasets.DatasetDict` (which has no
    `.column_names` of its own); for a `DatasetDict` the column must be present in
    EVERY split.
    """
    if isinstance(dataset, datasets.DatasetDict):
        return all(column in split.column_names for split in dataset.values())
    return column in dataset.column_names


def load_prebuilt_dataset(prebuilt_dataset_dir, logger=None):
    """Memory-map a prebuilt dataset, or return None when none is configured.

    `prebuilt_dataset_dir` is the `<root>/train` + `<root>/validation` pair
    materialised with `Dataset.save_to_disk` -- the same layout the distributed
    pretraining pipeline uses (see the note above the layout constants for why this
    module implements it locally instead of importing it). Both splits are read
    straight off disk
    as memory-mapped Arrow, so no `datasets` build runs on the compute nodes and no
    per-rank worker pool is forked at start-up.

    The on-disk `validation/` split is returned under the `"test"` key, so the
    resulting `DatasetDict` is indistinguishable from the one `split_dataset`
    produces for the raw path and callers never need to know which path ran.

    Nothing is asserted about the columns: a prebuilt dataset may hold raw
    `messages` / `prompt` / `chosen` / `rejected`, or pre-tokenised `input_ids`.
    Each trainer validates what it needs.

    Args:
        prebuilt_dataset_dir: The `<root>` holding train/ and validation/, or
            None/empty for "no prebuilt dataset".
        logger: Optional logger.

    Returns:
        A `datasets.DatasetDict` with `train`/`test` keys, or None when
        `prebuilt_dataset_dir` is not set.

    Raises:
        ValueError: If the path is set but does not hold a prebuilt dataset pair.
            Failing loudly keeps a typo from silently triggering an expensive rebuild
            from the raw shards.
    """
    resolved = resolve_prebuilt_dataset(prebuilt_dataset_dir)
    if resolved is None:
        return None

    train_path, val_path = resolved

    if logger:
        logger.info(f"Loading prebuilt dataset (memory-mapped): {train_path}")

    return datasets.DatasetDict(
        {
            "train": load_dataset_from_disk(train_path),
            "test": load_dataset_from_disk(val_path),
        }
    )


def load_training_dataset(
    train_dirs,
    dataset_type,
    num_proc,
    cache_dir,
    state,
    prebuilt_dataset_dir=None,
    logger=None,
):
    """Collect dataset files from directories/paths and load them into a Dataset.

    Args:
        train_dirs: A string path or list of string paths pointing to dataset
            files or directories containing dataset files. May be None when
            `prebuilt_dataset_dir` is set.
        dataset_type: 'jsonl' or 'parquet'.
        num_proc: Number of processes for dataset loading.
        cache_dir: Optional cache directory.
        state: accelerate.PartialState instance.
        prebuilt_dataset_dir: Optional `<root>` of a prebuilt dataset (see
            `load_prebuilt_dataset`). When set, the raw shards are not touched at all.
        logger: Optional logger.

    Returns:
        A `datasets.Dataset` loaded from the discovered files, or -- when
        `prebuilt_dataset_dir` is used -- a `datasets.DatasetDict` with
        `train`/`test` keys. The `DatasetDict` is already split, so the subsequent
        `split_dataset` call passes it straight through.

    Raises:
        ValueError: If neither `prebuilt_dataset_dir` nor `train_dirs` is given.
    """
    prebuilt = load_prebuilt_dataset(prebuilt_dataset_dir, logger=logger)
    if prebuilt is not None:
        return prebuilt

    if not train_dirs:
        raise ValueError(
            "No training data given. Pass --train_dataset_dir (raw shards) or "
            "--prebuilt_dataset_dir (a dataset materialised with Dataset.save_to_disk)."
        )

    assert dataset_type in ["jsonl", "parquet"], (
        f"Dataset type must be either 'jsonl' or 'parquet', got {dataset_type}."
    )

    if isinstance(train_dirs, str):
        train_dirs = [train_dirs]

    train_dataset_files = []
    for train_dir in train_dirs:
        if os.path.isfile(train_dir) and train_dir.endswith(f".{dataset_type}"):
            train_dataset_files.append(train_dir)
        elif os.path.isdir(train_dir):
            train_dataset_files += glob.glob(f"{train_dir}/*.{dataset_type}")
    train_dataset_files = sorted(train_dataset_files)

    # Ensure all processes are in sync before loading
    state.wait_for_everyone()

    def _load():
        return datasets.load_dataset(
            "json" if dataset_type == "jsonl" else dataset_type,
            data_files=train_dataset_files,
            split="train",
            num_proc=min(len(train_dataset_files), num_proc),
            cache_dir=cache_dir,
        )

    # Building the Arrow cache is the only step here that writes to disk, and on a
    # shared filesystem (multi-node) letting every rank build the SAME cache at the
    # same time is a classic start-up failure: the ranks race to write the same Arrow
    # files. Rank 0 builds it, the barrier waits for that, and the other ranks then
    # read the finished cache.
    if state.is_main_process:
        dataset = _load()
    state.wait_for_everyone()
    if not state.is_main_process:
        dataset = _load()

    return dataset


def split_dataset(dataset, test_size, seed, checkpoint_dir, save_test_set, master_process, state):
    """Optionally split a dataset into train/test sets.

    If test_size is None the dataset is returned unchanged. Otherwise it is
    split and, when save_test_set is True, the test split is written to
    `<checkpoint_dir>/test_set.jsonl` by the master process.

    A `DatasetDict` is already split -- that is what a prebuilt dataset comes back
    as -- and is returned untouched.

    Args:
        dataset: datasets.Dataset to split (or a DatasetDict to pass through).
        test_size: Number or fraction of samples for the test set, or None.
        seed: Random seed for the split.
        checkpoint_dir: Directory where the optional test JSONL is saved.
        save_test_set: Whether to save the test split to disk.
        master_process: True if this is the rank-0 process.
        state: accelerate.PartialState instance.

    Returns:
        The original dataset when test_size is None, or a DatasetDict with
        'train' and 'test' keys.

    Raises:
        ValueError: If an already-split dataset is combined with test_size.
    """
    if isinstance(dataset, datasets.DatasetDict):
        # Already split: a prebuilt dataset defines both splits on disk.
        if test_size is not None:
            raise ValueError(
                "--test_size cannot be combined with --prebuilt_dataset_dir: a prebuilt "
                "dataset already defines its train and validation splits (train/ and "
                "validation/). Drop --test_size."
            )
        return dataset

    if test_size is None:
        return dataset

    dataset = dataset.train_test_split(test_size=test_size, seed=seed)

    if master_process and save_test_set:
        test_file = os.path.join(checkpoint_dir, "test_set.jsonl")
        if not os.path.exists(test_file):
            dataset["test"].to_json(test_file, orient="records", lines=True)

    # Wait for master process to finish saving before other processes continue
    state.wait_for_everyone()

    return dataset


def load_tokenizer(
    model_name_or_path,
    max_length,
    cache_dir,
    chat_template_path=None,
    allow_eos_pad_token=False,
    require_chat_template=True,
):
    """Load a tokenizer, optionally apply a custom chat template, and validate it.

    By default, asserts that the tokenizer has a pad token distinct from the EOS
    token, which is required by the SFT/DPO trainers in this repository. Reward
    modeling can relax that constraint by setting `allow_eos_pad_token=True`.

    Args:
        model_name_or_path: Model identifier or local path.
        max_length: Maximum sequence length (model_max_length).
        cache_dir: Optional cache directory.
        chat_template_path: Path to a Jinja chat template file. Required when
            the tokenizer does not already have a chat_template set, unless
            `require_chat_template` is False.
        allow_eos_pad_token: Whether to allow using EOS as the pad token.
        require_chat_template: Whether a missing chat template is an error. Set it to
            False for data that is not conversational (e.g. plain-text preference
            pairs for reward modeling), where the template is never applied; the
            caller is then responsible for rejecting conversational data itself.

    Returns:
        A configured AutoTokenizer instance.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=max_length,
        cache_dir=cache_dir,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.chat_template is None and chat_template_path is not None:
        with open(chat_template_path) as f:
            tokenizer.chat_template = f.read()

    if require_chat_template:
        assert tokenizer.chat_template is not None, (
            "Tokenizer does not have a chat template. Please provide a chat template path."
        )

    if tokenizer.pad_token is None:
        assert allow_eos_pad_token and tokenizer.eos_token is not None, (
            "The tokenizer does not have a pad token. Please set a pad token before training."
        )
        tokenizer.pad_token = tokenizer.eos_token

    if not allow_eos_pad_token:
        assert tokenizer.pad_token != tokenizer.eos_token, (
            "The tokenizer's pad token is the same as the eos token. Please set a different pad token before training."
        )

    return tokenizer


def resolve_checkpoint_path(resume_from_checkpoint, master_process, logger: logging.Logger):
    """Resolve the most recent checkpoint inside a checkpoint directory.

    Given a path that may point to either an exact checkpoint directory or a
    parent directory containing multiple `checkpoint-<step>` sub-directories,
    returns the path to the latest checkpoint.

    Args:
        resume_from_checkpoint: Path to a checkpoint or a directory of
            checkpoints.
        master_process: True if this is the rank-0 process (controls logging).
        logger: Logger instance used to report the resolved checkpoint path.

    Returns:
        Resolved path to the checkpoint to resume from.
    """
    checkpoint_path = resume_from_checkpoint

    try:
        checkpoint_dirs = os.listdir(checkpoint_path)
        checkpoint_dirs = [d for d in checkpoint_dirs if d.startswith("checkpoint-")]
        checkpoint_path = os.path.join(
            checkpoint_path,
            sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[-1].split(".")[0]))[-1],
        )
    except Exception:
        # resume_from_checkpoint already points directly to a checkpoint
        pass

    if master_process:
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")

    return checkpoint_path


def run_training(
    trainer, resume_from_checkpoint, checkpoint_dir, master_process, logger: logging.Logger
):
    """Run trainer.train() with a fallback save on error.

    On success the final model is saved to `<checkpoint_dir>/final`.
    On failure the model is saved to `<checkpoint_dir>/last` and the
    exception is logged before re-raising.

    Args:
        trainer: A Hugging Face Trainer (or TRL Trainer) instance.
        resume_from_checkpoint: Checkpoint path to resume from, or None.
        checkpoint_dir: Directory for saving the final / last model.
        master_process: True if this is the rank-0 process (controls logging).
        logger: Logger instance used to report errors and save paths.
    """
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except Exception as e:
        save_path = os.path.join(checkpoint_dir, "last")
        trainer.save_model(save_path)
        if master_process:
            logger.error(f"Training failed with error: {e}")
            logger.info(f"Model saved to 'last' checkpoint at {save_path}")
        raise

    trainer.save_model(os.path.join(checkpoint_dir, "final"))
