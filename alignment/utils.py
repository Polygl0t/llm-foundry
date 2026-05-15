"""
Shared utilities for alignment training scripts.
"""
import os
import glob
import logging
import datasets
import transformers
import accelerate


def get_logger(name: str) -> logging.Logger:
    """Create and return a logger with a standard console handler.

    Args:
        name: Logger name (e.g. 'SFT-Trainer', 'DPO-Trainer').

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


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


def load_training_dataset(train_dirs, dataset_type, num_proc, cache_dir, state):
    """Collect dataset files from directories/paths and load them into a Dataset.

    Args:
        train_dirs: A string path or list of string paths pointing to dataset
            files or directories containing dataset files.
        dataset_type: 'jsonl' or 'parquet'.
        num_proc: Number of processes for dataset loading.
        cache_dir: Optional cache directory.
        state: accelerate.PartialState instance.

    Returns:
        A datasets.Dataset loaded from the discovered files.
    """
    assert dataset_type in ["jsonl", "parquet"], \
        f"Dataset type must be either 'jsonl' or 'parquet', got {dataset_type}."

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

    return datasets.load_dataset(
        "json" if dataset_type == "jsonl" else dataset_type,
        data_files=train_dataset_files,
        split="train",
        num_proc=min(len(train_dataset_files), num_proc),
        cache_dir=cache_dir,
    )


def split_dataset(dataset, test_size, seed, checkpoint_dir, save_test_set, master_process, state):
    """Optionally split a dataset into train/test sets.

    If test_size is None the dataset is returned unchanged. Otherwise it is
    split and, when save_test_set is True, the test split is written to
    `<checkpoint_dir>/test_set.jsonl` by the master process.

    Args:
        dataset: datasets.Dataset to split.
        test_size: Number or fraction of samples for the test set, or None.
        seed: Random seed for the split.
        checkpoint_dir: Directory where the optional test JSONL is saved.
        save_test_set: Whether to save the test split to disk.
        master_process: True if this is the rank-0 process.
        state: accelerate.PartialState instance.

    Returns:
        The original dataset when test_size is None, or a DatasetDict with
        'train' and 'test' keys.
    """
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


def load_tokenizer(model_name_or_path, max_length, cache_dir, chat_template_path=None, allow_eos_pad_token=False):
    """Load a tokenizer, optionally apply a custom chat template, and validate it.

    By default, asserts that the tokenizer has a pad token distinct from the EOS
    token, which is required by the SFT/DPO trainers in this repository. Reward
    modeling can relax that constraint by setting `allow_eos_pad_token=True`.

    Args:
        model_name_or_path: Model identifier or local path.
        max_length: Maximum sequence length (model_max_length).
        cache_dir: Optional cache directory.
        chat_template_path: Path to a Jinja chat template file. Required when
            the tokenizer does not already have a chat_template set.
        allow_eos_pad_token: Whether to allow using EOS as the pad token.

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

    if tokenizer.chat_template is None:
        assert chat_template_path is not None, \
            "Tokenizer does not have a chat template. Please provide a chat template path."
        with open(chat_template_path, "r") as f:
            tokenizer.chat_template = f.read()

    if tokenizer.pad_token is None:
        assert allow_eos_pad_token and tokenizer.eos_token is not None, \
            "The tokenizer does not have a pad token. Please set a pad token before training."
        tokenizer.pad_token = tokenizer.eos_token

    if not allow_eos_pad_token:
        assert tokenizer.pad_token != tokenizer.eos_token, \
            "The tokenizer's pad token is the same as the eos token. Please set a different pad token before training."

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


def run_training(trainer, resume_from_checkpoint, checkpoint_dir, master_process, logger: logging.Logger):
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
