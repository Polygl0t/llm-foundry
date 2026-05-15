# Data Processing

Dataset preprocessing, parsing, filtering, and tokenization utilities.

## Overview

This folder contains data preparation code used before training. It covers raw corpus processing, dataset parsing and splitting, parser examples, Common Crawl utilities, filtering code, and the downstream tokenization pipeline.

At a high level:

- `cc/` contains Common Crawl-related processing code.
- `filters/` contains dataset filtering utilities.
- `parsers/` contains parser modules used by `preprocess.py`.
- `tokenization/` contains tokenization, packing, decontamination, and validation split utilities.
- `preprocess.py` runs a single-pass DataTrove pipeline over parquet datasets.
- `utils.py` contains shared helpers for preprocessing scripts.

## Parsing with preprocess.py

`preprocess.py` is intended for dataset parsing tasks such as:

- splitting a dataset into subsets based on a metadata column
- dropping samples that do not satisfy a rule
- adding or rewriting metadata columns before output

The script can run in two modes:

- built-in stratification with `--stratify_by_column`
- external parser mode with `--parser_path`

These can also be combined in one pass, so a parser can enrich or filter rows while built-in stratification decides the output subset.

## Parser Modules

See `data/parsers/README.md` for parser-specific details.

## SLURM Cluster Jobs

`preprocess.sh` is an example SLURM job for running the parser pipeline on a cluster. Before submitting, update the following values:

- `--account` — Your SLURM account
- `--partition` — Your target partition
- `username`, `file_system`, `workspace_name` — Paths to your working directory
- dataset input and output paths in the main command

```bash
sbatch preprocess.sh
```

## Notes

- `preprocess.py` expects parquet input files and writes one subfolder per output subset.
- Per-subset `.metadata` files are written after the pipeline completes.
- If both `--parser_path` and `--stratify_by_column` are provided, parser logic runs first and stratification determines the final subset name.

