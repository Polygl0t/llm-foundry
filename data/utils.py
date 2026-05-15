"""
Utilities for data preprocessing.
"""
import importlib.util
import glob
import json
import os
import sys
import logging
from dataclasses import dataclass
from types import ModuleType
from typing import Any

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a logger with a consistent format.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default: logging.INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid adding duplicate handlers if the logger was already configured.
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent log records from propagating to the root logger.
    logger.propagate = False

    return logger


def save_metadata(output_dir, **kwargs):
    """Write key-value metadata to `<output_dir>/.metadata`."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, ".metadata"), "w") as f:
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")


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

    with open(file_path, "r", encoding="utf-8") as fh:
        line = fh.readline()
        if not line.strip():
            return []
        return list(json.loads(line).keys())


@dataclass(frozen=True)
class ParseResult:
    subset: str
    row: dict[str, Any]


def document_to_row(doc) -> dict[str, Any]:
    """Reconstruct the original flat row from a DataTrove document."""
    return {"text": doc.text, "id": doc.id, **doc.metadata}


def sanitize_subset_name(value: Any, fallback: str) -> str:
    """Convert arbitrary parser output into a safe subset folder name."""
    if value is None:
        return fallback

    subset = str(value).strip()
    if not subset:
        return fallback

    subset = subset.replace(os.sep, "_")
    if os.altsep:
        subset = subset.replace(os.altsep, "_")
    return subset


def load_parser_config(raw_value: str | None) -> dict[str, Any]:
    """Load parser config from a JSON string or a JSON file path."""
    if raw_value is None:
        return {}

    if os.path.isfile(raw_value):
        with open(raw_value, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    else:
        loaded = json.loads(raw_value)

    if not isinstance(loaded, dict):
        raise ValueError("--parser_config must decode to a JSON object.")
    return loaded


def load_parser_module(parser_path: str) -> ModuleType:
    """Load a parser module from a Python file path."""
    module_path = os.path.abspath(parser_path)
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load parser module from: {parser_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_input_files(datasets_dir: str) -> list[str]:
    """Return input parquet files from the dataset directory."""
    input_files = list_matching_files(datasets_dir, "*.parquet")
    if not input_files:
        raise FileNotFoundError(f"No parquet files found in: {datasets_dir}")
    return input_files


def validate_parquet_column(input_files: list[str], column_name: str) -> None:
    """Validate that a column exists in the parquet schema."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(input_files[0])
    if column_name not in schema.names:
        raise ValueError(
            f"Column '{column_name}' not found in dataset. "
            f"Available columns: {list(schema.names)}"
        )


def discover_unique_column_values(datasets_dir: str, column_name: str, logger) -> list[Any]:
    """Discover unique values from a parquet column via columnar reads."""
    import pyarrow.parquet as pq

    input_files = list_input_files(datasets_dir)
    validate_parquet_column(input_files, column_name)

    logger.info(f"Scanning unique values for '{column_name}'...")
    unique_values: set[Any] = set()
    for file_path in input_files:
        column = pq.read_table(file_path, columns=[column_name]).column(column_name)
        unique_values.update(value.as_py() for value in column if value.is_valid)

    discovered = sorted(unique_values, key=lambda value: (value is None, str(value)))
    logger.info(f"Found {len(discovered)} unique values: {discovered}")
    return discovered


def normalize_parse_result(result: Any, doc, default_subset_name: str) -> ParseResult | None:
    """Normalize parser output into a routed row or a dropped sample."""
    if result is None:
        return None

    base_row = document_to_row(doc)

    if isinstance(result, str):
        return ParseResult(sanitize_subset_name(result, default_subset_name), base_row)

    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError("Parser tuple results must be of the form (subset, row).")
        subset_name, row = result
        if not isinstance(row, dict):
            raise ValueError("Parser tuple results must provide a dict row.")
        return ParseResult(sanitize_subset_name(subset_name, default_subset_name), row)

    if not isinstance(result, dict):
        raise TypeError(
            "Parser must return None, a subset string, a (subset, row) tuple, or a dict."
        )

    subset_name = sanitize_subset_name(result.get("subset"), default_subset_name)
    row = result.get("row")
    if row is not None:
        if not isinstance(row, dict):
            raise ValueError("Parser result 'row' must be a dict.")
        return ParseResult(subset_name, row)

    row = dict(base_row)
    if "text" in result:
        row["text"] = result["text"]
    if "id" in result:
        row["id"] = result["id"]

    metadata_updates = result.get("metadata")
    if metadata_updates is not None:
        if not isinstance(metadata_updates, dict):
            raise ValueError("Parser result 'metadata' must be a dict.")
        row.update(metadata_updates)

    return ParseResult(subset_name, row)


def get_subset_stats(pipeline_stats, subset_name: str) -> tuple[int, int]:
    """Extract routed document and token counts for one subset."""
    stats_by_name = {stat.name: stat for stat in pipeline_stats.stats}
    routing = stats_by_name.get("RoutingWriter")
    if routing is None:
        return 0, 0

    document_stat = routing.stats.get(f"{subset_name}_documents")
    token_stat = routing.stats.get(f"{subset_name}_tokens")
    total_docs = int(document_stat.total) if document_stat is not None else 0
    total_tokens = int(token_stat.total) if token_stat is not None else 0
    return total_docs, total_tokens


def write_subset_metadata(output_dir: str, output_type: str, pipeline_stats, subset_name: str, logger) -> tuple[int, int]:
    """Write per-subset metadata after pipeline execution."""
    output_files = list_matching_files(output_dir, f"*.{output_type}")
    features = infer_file_features(output_files[0], output_type) if output_files else []
    total_docs, total_tokens = get_subset_stats(pipeline_stats, subset_name)

    save_metadata(
        output_dir,
        samples=total_docs,
        tokens=total_tokens,
        chunks=len(output_files),
        features=features,
        subset=subset_name,
    )
    logger.info(f"Saved metadata to: {output_dir}/.metadata")
    return total_docs, total_tokens


def discover_written_subsets(output_dir: str) -> list[str]:
    """Discover subset folders created by a parser run."""
    if not os.path.isdir(output_dir):
        return []

    subset_names = []
    for entry in os.scandir(output_dir):
        if entry.is_dir():
            subset_names.append(entry.name)
    return sorted(subset_names)
