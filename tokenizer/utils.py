"""
Shared utilities for tokenizer training scripts.
"""
import glob
import json
import logging
import os
from typing import Any, Optional

# Additional tokens beyond BOS, EOS, UNK, and PAD.
EXTRA_TOKENS = [
    "<tools>",
    "</tools>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
    "<context>",
    "</context>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|repo_name|>",
    "<|image|>",
    "<|image_pad|>",
    "<|image_placeholder|>",
    # The indented tokens are a trick from the Olmo tokenizer.
    # (note: Pythia / GPT-NeoX also did something like this)
    # This helps make the tokenizer more efficient when dealing with code data.
    "                        ",
    "                       ",
    "                      ",
    "                     ",
    "                    ",
    "                   ",
    "                  ",
    "                 ",
    "                ",
    "               ",
    "              ",
    "             ",
    "            ",
    "           ",
    "          ",
    "         ",
    "        ",
    "       ",
    "      ",
    "     ",
    "    ",
    "   ",
    "  ",
]


def get_logger(name: str) -> logging.Logger:
    """Create and return a logger with a standard console handler.

    Args:
        name: Logger name (e.g. 'My-Cool-Script')

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


def load_text_dataset(path: str, data_type: str, cache_dir: Optional[str] = None, num_proc: int = 8):
    """Load a text dataset from a file or directory.

    Supports txt, jsonl, parquet, and csv formats. When *path* is a directory,
    all files matching `*.{data_type}` are collected.

    Args:
        path: Path to a single data file or a directory of data files.
        data_type: File extension / format (`"txt"`, `"jsonl"`, `"parquet"`, `"csv"`).
        cache_dir: Directory used by the Hugging Face datasets cache.
        num_proc: Number of parallel workers passed to `load_dataset`.

    Returns:
        A `datasets.Dataset` with `split="train"`.
    """
    import datasets as _datasets

    valid = ["txt", "jsonl", "parquet", "csv"]
    if data_type not in valid:
        raise ValueError(f"Invalid data_type '{data_type}'. Must be one of {valid}.")

    if os.path.isdir(path):
        data_files = glob.glob(os.path.join(path, f"*.{data_type}"))
        if not data_files:
            raise ValueError(f"No .{data_type} files found in '{path}'.")
    elif os.path.isfile(path):
        data_files = [path]
    else:
        raise ValueError(f"Invalid path: '{path}'. Must be an existing file or directory.")

    hf_type = "text" if data_type == "txt" else "json" if data_type == "jsonl" else data_type
    return _datasets.load_dataset(hf_type, data_files=data_files, split="train", cache_dir=cache_dir, num_proc=num_proc)


def update_tokenizer_config(output_dir: str, **fields: Any) -> None:
    """Patch `tokenizer_config.json` in *output_dir* with the given *fields*.

    Reads the existing file, merges *fields* into it, then writes it back.

    Args:
        output_dir: Directory containing `tokenizer_config.json`.
        **fields: Key/value pairs to set (or overwrite) in the config.
    """
    config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config.update(fields)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def write_special_tokens_map(output_dir: str, *, bos_token: str, eos_token: str, unk_token: str, pad_token: str) -> None:
    """Write the canonical special-token map for saved tokenizer artifacts."""
    special_tokens_map_path = os.path.join(output_dir, "special_tokens_map.json")
    special_tokens_map = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "unk_token": unk_token,
        "pad_token": pad_token,
    }
    with open(special_tokens_map_path, "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2)


def validate_saved_tokenizer(output_dir: str) -> None:
    """Assert that the tokenizer in *output_dir* loads correctly in both slow and fast modes.

    Args:
        output_dir: Directory containing the saved tokenizer files.

    Raises:
        AssertionError: If either the slow or fast tokenizer fails to load.
    """
    from transformers import AutoTokenizer
    assert AutoTokenizer.from_pretrained(output_dir, use_fast=False), \
        f"Failed to load slow tokenizer from '{output_dir}'."
    assert AutoTokenizer.from_pretrained(output_dir, use_fast=True), \
        f"Failed to load fast tokenizer from '{output_dir}'."


def push_tokenizer_to_hub(output_dir: str, repo_id: str, token: str, private: bool = True) -> None:
    """Create (if needed) a Hub repository and upload the tokenizer folder to it.

    Args:
        output_dir: Local directory containing the saved tokenizer files.
        repo_id: Hugging Face Hub repository ID (e.g. `"username/my-tokenizer"`).
        token: Hugging Face authentication token.
        private: Whether to create the repository as private.
    """
    from huggingface_hub import create_repo, HfApi
    _logger = get_logger("push_tokenizer_to_hub")
    _logger.info(f"Pushing tokenizer to the hub at '{repo_id}'...")
    create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True, private=private)
    HfApi(token=token).upload_folder(repo_id=repo_id, folder_path=output_dir)
    _logger.info("Tokenizer uploaded to the hub.")
