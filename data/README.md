# Data Processing

This folder contains data preparation code used before training. It covers raw corpus processing, dataset parsing and splitting, parser examples, Common Crawl utilities, filtering code, and the downstream tokenization pipeline.

## Contents

- [`cc/`](./cc) contains Common Crawl-related processing code.
- [`filters/`](./filters) contains dataset filtering utilities.
- [`parsers/`](./parsers) contains parser modules used by `preprocess.py`.
- [`tokenization/`](./tokenization) contains tokenization, packing, decontamination, and validation split utilities.
- [`preprocess.py`](./preprocess.py) — Runner for a DataTrove-based preprocessing pipeline.
- [`utils.py`](./utils.py) — Shared helper functions used by data preprocessing scripts.

## Usage Summary

### `preprocess.py`

`preprocess.py` is intended for dataset parsing tasks such as:

- splitting a dataset into subsets based on a metadata column
- dropping samples that do not satisfy a rule
- adding or rewriting metadata columns before output

It supports two main modes:

- built-in stratification with `--stratify_by_column`
- external parser mode with `--parser_path`

These modes can be combined, so a parser can enrich or filter rows while built-in stratification decides the final subset name.

Examples:
```bash
# 1. Stratify by a metadata column:
python data/preprocess.py \
  --datasets_dir ./raw_data \
  --output_dir ./parsed_data \
  --stratify_by_column


# 2. Run a custom parser module:
python data/preprocess.py \
  --datasets_dir ./raw_data \
  --output_dir ./parsed_data \
  --parser_path ./data/parsers/add_uuid_parser.py

# 3. Stratify and use an external parser in the same pass:

python data/preprocess.py \
  --datasets_dir ./raw_data \
  --output_dir ./parsed_data \
  --stratify_by_column edu_int_score \
  --parser_path ./data/parsers/add_uuid_parser.py

# 4. Pass parser configuration as JSON:
python data/preprocess.py \
  --datasets_dir ./raw_data \
  --output_dir ./parsed_data \
  --parser_path ./data/parsers/score_threshold_filter_parser.py \
  --parser_config '{"score_column": "edu_int_score", "minimum_score": 3}'
```

Main parameters:
- `--datasets_dir` — Directory containing the input parquet files.
- `--output_dir` — Root output directory; one subfolder is created per output subset.
- `--output_type` — Output format, either `jsonl` or `parquet`.
- `--token_count_column` — Metadata column used to accumulate per-subset token counts.
- `--parser_path` — Optional Python parser module path; the module must define `parse_document(doc, args)`.
- `--parser_config` — Optional parser config as a JSON string or a path to a JSON file.
- `--default_subset_name` — Default subset name used when no subset is specified by the parser.
- `--stratify_by_column` — Create one output subset for each unique value in this metadata column.
- `--write_batch_size` — Number of rows buffered before writing each subset shard.
- `--tasks` — Number of DataTrove tasks.
- `--workers` — Number of local worker processes used by DataTrove.
- `--logs_folder` — Folder used by DataTrove for logs.

### Parser modules

See [`data/parsers/README.md`](./parsers/README.md) for parser-specific details and examples.

A parser module should export:

- `parse_document(doc, args)` — return `None`, a subset name, or a dict with `subset`, `row`, and/or `metadata`.
- Optional hooks: `setup(args)` and `resolve_subsets(args)`.

## SLURM Cluster Jobs

The `.sh` helper scripts are configured for SLURM-based GPU clusters. Before submitting, update the values in each script for:

- `--account`
- `--partition`
- `username`, `file_system`, `workspace_name`

```bash
sbatch data/preprocess.sh
```

## Notes

- [`preprocess.py`](./preprocess.py) expects parquet input files and writes one subfolder per output subset.
- Per-subset `.metadata` files are written after the pipeline completes.
- If both `--parser_path` and `--stratify_by_column` are provided, the parser logic runs first and stratification determines the final subset name.
