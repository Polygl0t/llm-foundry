"""
Dataset Parsing / Preprocessing

This script runs a single-pass DataTrove pipeline over a parquet dataset and lets
you route, filter, or enrich samples with a user-defined parser.

Use this script when you need to:
  - split a dataset into subsets based on metadata values
  - drop samples that do not satisfy some rule
  - add or rewrite metadata columns before writing the output rows

The parser logic is intentionally user-defined. This script provides the runner,
output routing, and metadata bookkeeping.

Parser API:
Pass `--parser_path` to a Python file that defines:

    def parse_document(doc, args):
        ...

`parse_document` receives a DataTrove document and must return one of:
  - `None`: drop the sample
  - `"subset_name"`: keep the original row and route it to that subset
  - `{"subset": "subset_name", "row": {...}}`: write a fully specified row
  - `{"subset": "subset_name", "metadata": {...}}`: merge metadata updates into
    the original flat row
  - `("subset_name", {...})`: shorthand for `{"subset": ..., "row": ...}`

Optional parser hooks:
  - `setup(args)` for parser-side validation or initialization
  - `resolve_subsets(args)` to declare expected subset names for logging

Examples:
1. Built-in stratification by a column:

    python preprocess.py \\
        --datasets_dir ./raw_data \\
        --output_dir ./parsed_data \\
        --stratify_by_column language

2. External parser that adds a UUID to every sample:

    python preprocess.py \\
        --datasets_dir ./raw_data \\
        --output_dir ./parsed_data \\
        --parser_path ./data/parsers/add_uuid_parser.py

3. Stratify by edu_int_score and add a UUID in the same pass:

    python preprocess.py \\
        --datasets_dir ./raw_data \\
        --output_dir ./parsed_data \\
        --stratify_by_column edu_int_score \\
        --parser_path ./data/parsers/add_uuid_parser.py

4. External parser that drops low-scoring samples:

    python preprocess.py \\
        --datasets_dir ./raw_data \\
        --output_dir ./parsed_data \\
        --parser_path ./data/parsers/score_threshold_filter_parser.py \\
        --parser_config '{"score_column": "edu_int_score", "minimum_score": 3}'
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader

from utils import (
    ParseResult,
    discover_unique_column_values,
    discover_written_subsets,
    document_to_row,
    get_logger,
    load_parser_config,
    load_parser_module,
    normalize_parse_result,
    sanitize_subset_name,
    write_subset_metadata,
)

logger = get_logger("Preprocess")


def apply_stratification(parse_result: ParseResult, doc, args) -> ParseResult:
    """Override the output subset when built-in stratification is requested."""
    if not args.stratify_by_column:
        return parse_result

    subset_name = sanitize_subset_name(
        doc.metadata.get(args.stratify_by_column),
        args.default_subset_name,
    )
    return ParseResult(subset_name, parse_result.row)


def build_builtin_parser(args) -> tuple[Callable[[Any], ParseResult | None], list[str]]:
    """Build the built-in parser behavior."""
    if args.stratify_by_column:
        discovered = discover_unique_column_values(
            args.datasets_dir,
            args.stratify_by_column,
            logger,
        )
        expected_subsets = [
            sanitize_subset_name(value, args.default_subset_name) for value in discovered
        ]

        def parse_document(doc):
            subset_name = sanitize_subset_name(
                doc.metadata.get(args.stratify_by_column),
                args.default_subset_name,
            )
            return ParseResult(subset_name, document_to_row(doc))

        return parse_document, expected_subsets

    def parse_document(doc):
        return ParseResult(args.default_subset_name, document_to_row(doc))

    return parse_document, [args.default_subset_name]


def build_active_parser(args) -> tuple[Callable[[Any], ParseResult | None], str, list[str]]:
    """Resolve the active parser and optionally compose it with stratification."""
    if not args.parser_path:
        parser_fn, expected_subsets = build_builtin_parser(args)
        return parser_fn, "built-in parser", expected_subsets

    parser_module = load_parser_module(args.parser_path)
    if hasattr(parser_module, "setup"):
        parser_module.setup(args)

    parse_document = getattr(parser_module, "parse_document", None)
    if not callable(parse_document):
        raise AttributeError(
            f"Parser module '{args.parser_path}' must define a callable parse_document(doc, args)."
        )

    resolve_subsets = getattr(parser_module, "resolve_subsets", None)
    expected_subsets = resolve_subsets(args) if callable(resolve_subsets) else []
    parser_name = getattr(parser_module, "PARSER_NAME", os.path.basename(args.parser_path))

    if args.stratify_by_column:
        discovered = discover_unique_column_values(
            args.datasets_dir,
            args.stratify_by_column,
            logger,
        )
        expected_subsets = [
            sanitize_subset_name(value, args.default_subset_name) for value in discovered
        ]

    def wrapped(doc):
        parsed = normalize_parse_result(parse_document(doc, args), doc, args.default_subset_name)
        if parsed is None:
            return None
        return apply_stratification(parsed, doc, args)

    return wrapped, parser_name, [sanitize_subset_name(name, args.default_subset_name) for name in expected_subsets]


class RoutingWriter(PipelineStep):
    """Terminal pipeline step that applies a parser and writes subset outputs."""

    name = "RoutingWriter"

    def __init__(
        self,
        parse_document: Callable[[Any], ParseResult | None],
        output_dir: str,
        output_type: str,
        token_count_column: str,
        write_batch_size: int = 1_000,
    ):
        super().__init__()
        self.parse_document = parse_document
        self.output_dir = output_dir
        self.output_type = output_type
        self.token_count_column = token_count_column
        self.write_batch_size = write_batch_size

    def _flush_batch(self, subset_name, rows, rank, pq_writers, jsonl_files):
        """Write buffered rows to the corresponding subset shard."""
        out_dir = os.path.join(self.output_dir, subset_name)
        os.makedirs(out_dir, exist_ok=True)

        if self.output_type == "parquet":
            table = pa.Table.from_pylist(rows)
            if subset_name not in pq_writers:
                file_path = os.path.join(out_dir, f"{rank:05d}.parquet")
                pq_writers[subset_name] = pq.ParquetWriter(file_path, table.schema)
            pq_writers[subset_name].write_table(table)
            return

        if subset_name not in jsonl_files:
            file_path = os.path.join(out_dir, f"{rank:05d}.jsonl")
            jsonl_files[subset_name] = open(file_path, "w", encoding="utf-8")

        for row in rows:
            jsonl_files[subset_name].write(json.dumps(row, ensure_ascii=False) + "\n")

    def run(self, data, rank: int = 0, world_size: int = 1):
        buffers: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        pq_writers: dict[str, Any] = {}
        jsonl_files: dict[str, Any] = {}

        try:
            for doc in data:
                parsed = self.parse_document(doc)
                if parsed is None:
                    self.stat_update("filtered_documents")
                    continue

                subset_name = parsed.subset
                row = parsed.row
                tokens = row.get(self.token_count_column, doc.metadata.get(self.token_count_column) or 0)

                self.stat_update(f"{subset_name}_documents")
                self.stat_update(f"{subset_name}_tokens", value=tokens)

                buffers[subset_name].append(row)
                if len(buffers[subset_name]) >= self.write_batch_size:
                    self._flush_batch(subset_name, buffers[subset_name], rank, pq_writers, jsonl_files)
                    buffers[subset_name] = []

            for subset_name, rows in buffers.items():
                if rows:
                    self._flush_batch(subset_name, rows, rank, pq_writers, jsonl_files)
        finally:
            for writer in pq_writers.values():
                writer.close()
            for handle in jsonl_files.values():
                handle.close()

        if False:
            yield


def main(args):
    if args.output_type not in {"jsonl", "parquet"}:
        raise ValueError("Output type must be either 'jsonl' or 'parquet'.")

    args.parser_config = load_parser_config(args.parser_config)
    parser_fn, parser_name, expected_subsets = build_active_parser(args)

    if expected_subsets:
        logger.info(f"Using parser '{parser_name}' with expected subsets: {expected_subsets}")
    else:
        logger.info(f"Using parser '{parser_name}'")

    pipeline = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder=args.datasets_dir,
                glob_pattern="*.parquet",
            ),
            RoutingWriter(
                parse_document=parser_fn,
                output_dir=args.output_dir,
                output_type=args.output_type,
                token_count_column=args.token_count_column,
                write_batch_size=args.write_batch_size,
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=args.logs_folder,
    )

    logger.info("Starting parsing pipeline...")
    pipeline_stats = pipeline.run()
    logger.info("Parsing pipeline completed.")

    written_subsets = discover_written_subsets(args.output_dir)
    if not written_subsets:
        logger.warning("No output subsets were written. The parser may have filtered every sample.")
        return

    logger.info("=" * 60)
    for subset_name in written_subsets:
        subset_output_dir = os.path.join(args.output_dir, subset_name)
        total_docs, total_tokens = write_subset_metadata(
            subset_output_dir,
            args.output_type,
            pipeline_stats,
            subset_name=subset_name,
            logger=logger,
        )
        logger.info(f"[{subset_name}] Samples: {total_docs:,} | Tokens: {total_tokens:,}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        required=True,
        help="Directory containing the input parquet files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output directory; one subfolder per subset is created inside.",
    )
    parser.add_argument(
        "--output_type",
        choices=["jsonl", "parquet"],
        default="parquet",
        help="Output file format.",
    )
    parser.add_argument(
        "--token_count_column",
        type=str,
        default="token_count",
        help="Metadata column used to accumulate per-subset token counts.",
    )
    parser.add_argument(
        "--parser_path",
        type=str,
        default=None,
        help="Optional Python file that defines parse_document(doc, args). Can be combined with --stratify_by_column.",
    )
    parser.add_argument(
        "--parser_config",
        type=str,
        default=None,
        help="Optional parser config as a JSON string or path to a JSON file.",
    )
    parser.add_argument(
        "--default_subset_name",
        type=str,
        default="default",
        help="Fallback subset name when a parser does not specify one.",
    )
    parser.add_argument(
        "--stratify_by_column",
        type=str,
        default=None,
        help="Create one subset per unique value of this column. Can be combined with --parser_path.",
    )
    parser.add_argument(
        "--write_batch_size",
        type=int,
        default=1_000,
        help="Number of rows buffered before writing each subset shard.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=64,
        help="Number of tasks used by DataTrove.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of local workers used by DataTrove.",
    )
    parser.add_argument(
        "--logs_folder",
        type=str,
        default="./logs",
        help="Folder used by DataTrove for logs.",
    )

    main(parser.parse_args())
