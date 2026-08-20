"""
Shared utilities for filtering and annotation scripts.
"""

import glob
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve()
while not (_REPO_ROOT / "pyproject.toml").exists() and _REPO_ROOT.parent != _REPO_ROOT:
    _REPO_ROOT = _REPO_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def read_metadata(metadata_file: str) -> dict | None:
    """
    Read metadata from a file in YAML-like key: value format.

    Args:
        metadata_file: Path to the metadata file.

    Returns:
        Dictionary with metadata values, or None if the file does not exist.
    """
    if not os.path.exists(metadata_file):
        return None

    metadata = {}
    with open(metadata_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    if "." in value:
                        metadata[key] = float(value)
                    else:
                        metadata[key] = int(value)
                except ValueError:
                    metadata[key] = value
    return metadata


def write_metadata(metadata_file: str, metadata: dict) -> None:
    """
    Write metadata to a file in YAML-like key: value format.

    Args:
        metadata_file: Path to the metadata file to write.
        metadata: Dictionary of metadata values to persist.
    """
    with open(metadata_file, "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")


def initialize_or_load_metadata(lang_output_path: str) -> dict:
    """
    Return metadata for a language output folder.

    Loads from an existing `.metadata` file if present. Otherwise scans all
    JSONL files in the folder to rebuild the statistics and writes the result
    to `.metadata` for future calls.

    Args:
        lang_output_path: Path to the language-specific output folder.

    Returns:
        Dictionary with at least `lines` and `tokens` keys.
    """
    metadata_file = os.path.join(lang_output_path, ".metadata")

    metadata = read_metadata(metadata_file)
    if metadata is not None:
        return metadata

    all_jsonl_files = glob.glob(os.path.join(lang_output_path, "*.jsonl"))

    if not all_jsonl_files:
        return {"lines": 0, "tokens": 0}

    total_lines = 0
    total_tokens = 0

    for jsonl_file in all_jsonl_files:
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    total_lines += 1
                    total_tokens += data.get("token_count", 0)
                except json.JSONDecodeError:
                    continue

    metadata = {"lines": total_lines, "tokens": total_tokens}
    write_metadata(metadata_file, metadata)
    return metadata


def is_messages_column(dataset, column_name):
    """Heuristically determine if a column contains a list of message dicts with 'content' fields."""
    if column_name not in dataset.column_names or len(dataset) == 0:
        return False
    value = dataset[0].get(column_name)
    if isinstance(value, list) and len(value) > 0:
        return isinstance(value[0], dict) and "content" in value[0]
    return False


def flatten_messages(messages):
    """Convert a list of message dicts to a single string by concatenating the 'content' fields."""
    if not messages:
        return ""
    contents = []
    for msg in messages:
        if isinstance(msg, dict) and "content" in msg:
            content = msg["content"]
            if content:
                contents.append(str(content))
    return "\n".join(contents)


def apply_chat_template_to_dataset(dataset, tokenizer, text_column, num_proc):
    """
    Apply chat template formatting to the text column of the dataset.

    Args:
        dataset: The input dataset
        tokenizer: The tokenizer with chat template
        text_column: Name of the column containing text
        num_proc: Number of processes for parallel processing

    Returns:
        tuple: (formatted_dataset, new_text_column_name)
    """
    if tokenizer.chat_template is None:
        raise ValueError(
            "The tokenizer does not have a chat template. "
            "Please use a tokenizer that supports chat templates."
        )

    def format_messages(example):
        formatted_text = tokenizer.apply_chat_template(
            example[text_column],
            tokenize=False,  # Returns string instead of token IDs
        )
        example["formatted_text"] = formatted_text
        return example

    formatted_dataset = dataset.map(
        format_messages, num_proc=num_proc, desc="Formatting messages with chat template"
    )

    return formatted_dataset, "formatted_text"
