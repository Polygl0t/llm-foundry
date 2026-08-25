"""
Shared utilities for filtering and annotation scripts.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve()
while not (_REPO_ROOT / "pyproject.toml").exists() and _REPO_ROOT.parent != _REPO_ROOT:
    _REPO_ROOT = _REPO_ROOT.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.dataset_loader import DatasetLoader as DatasetLoader  # noqa: E402
from shared.logging import get_logger as get_logger  # noqa: E402
from shared.save_dataset import save_dataset as save_dataset  # noqa: E402


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
