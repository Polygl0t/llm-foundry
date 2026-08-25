import glob
import json
import os

from shared.read_metadata import read_metadata
from shared.write_metadata import write_metadata


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
