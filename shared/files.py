"""Shared file/shard helpers."""

import glob
import json
import os


def list_matching_files(directory: str, *patterns: str) -> list[str]:
    """Return a sorted list of files in `directory` matching any glob pattern."""
    matches: set[str] = set()
    for pattern in patterns:
        matches.update(glob.glob(os.path.join(directory, pattern)))
    return sorted(matches)


def infer_file_features(file_path: str, output_type: str) -> list[str]:
    """Infer output feature names from a written parquet or jsonl shard."""
    if output_type == "parquet":
        import pyarrow.parquet as pq

        return list(pq.read_schema(file_path).names)

    with open(file_path, encoding="utf-8") as fh:
        line = fh.readline()
        if not line.strip():
            return []
        return list(json.loads(line).keys())
