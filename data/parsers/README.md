# Data Parsers

Parser modules for `data/preprocess.py`.

## Overview

This folder contains example parser implementations used by the generic preprocessing runner in `data/preprocess.py`.

A parser is responsible for sample-level logic such as:

- filtering out unwanted samples
- adding metadata fields
- replacing or rewriting output rows
- optionally suggesting expected subset names

`preprocess.py` handles dataset reading, output routing, shard writing, and metadata generation. Parsers only define the sample transformation logic.

## Contents

- **add_uuid_parser.py** — Adds a UUID field to every sample.
- **score_threshold_filter_parser.py** — Drops samples whose score column is below a configured minimum.

## Parser API

Pass a parser file to `preprocess.py` with `--parser_path`.

Each parser must define:

```python
def parse_document(doc, args):
    ...
```

`parse_document(doc, args)` receives a DataTrove document and may return one of the following:

- `None` — drop the sample
- `"subset_name"` — keep the original row and route it to that subset
- `("subset_name", {...})` — route to a subset with a fully specified output row
- `{"subset": "subset_name", "row": {...}}` — same as above in dict form
- `{"subset": "subset_name", "metadata": {...}}` — merge metadata into the original row before writing

Optional parser hooks:

```python
def setup(args):
    ...


def resolve_subsets(args):
    ...
```

- `setup(args)` can validate parser configuration or initialize state.
- `resolve_subsets(args)` can declare expected subset names for logging.

## Example Parsers

### add_uuid_parser.py

Adds a UUID field to every sample.

Optional config:

```json
{
  "uuid_column": "uuid"
}
```

Example usage:

```bash
python data/preprocess.py \
    --datasets_dir data/raw_parquet \
    --output_dir data/parsed \
    --parser_path ./data/parsers/add_uuid_parser.py
```

### score_threshold_filter_parser.py

Drops samples when a configured score column is missing or below a configured threshold.

Required config:

```json
{
  "score_column": "edu_int_score",
  "minimum_score": 3
}
```

Example usage:

```bash
python data/preprocess.py \
    --datasets_dir data/raw_parquet \
    --output_dir data/parsed \
    --parser_path ./data/parsers/score_threshold_filter_parser.py \
    --parser_config '{"score_column": "edu_int_score", "minimum_score": 3}'
```

## Combining Parsers with Stratification

A parser can be combined with `--stratify_by_column` in `preprocess.py`.

In that mode:

- the parser still decides whether a sample is kept or dropped
- the parser can still add metadata fields
- the final output subset is derived from the stratification column

Example:

```bash
python data/preprocess.py \
    --datasets_dir data/raw_parquet \
    --output_dir data/parsed \
    --stratify_by_column edu_int_score \
    --parser_path ./data/parsers/add_uuid_parser.py
```

## Notes

- Parsers should be lightweight and deterministic unless randomness is explicitly intended.
- If a parser rewrites the full row, it is responsible for preserving any fields it still wants in the output.
- If a parser only needs to add metadata, returning a `metadata` dict is usually the simplest option.
- Parser config is passed through `--parser_config` as either a JSON string or a path to a JSON file.

## Adding New Parsers

To add a new parser:

1. Create a new Python file in this folder.
2. Implement `parse_document(doc, args)`.
3. Optionally add `setup(args)` and `resolve_subsets(args)`.
4. Run it through `data/preprocess.py` with `--parser_path`.
5. Document any required `--parser_config` fields in the parser docstring.