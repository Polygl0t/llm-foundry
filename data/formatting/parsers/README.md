# Data Parsers

Parsers and formatters for [`../preprocess.py`](../preprocess.py).

A parser is responsible for sample-level logic such as:

- filtering out unwanted samples
- adding or transforming metadata fields
- replacing or rewriting output rows
- optionally suggesting expected subset names

[`../preprocess.py`](../preprocess.py) handles dataset reading, output routing, shard writing, and metadata generation. Parsers only define the sample transformation logic.

## Contents

- [`add_uuid_parser.py`](./add_uuid_parser.py) — Adds a UUID field to every sample.
- [`score_threshold_filter_parser.py`](./score_threshold_filter_parser.py) — Drops samples whose score column is below a configured minimum.

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

- `setup(args)` can validate parser configuration or initialize parser state.
- `resolve_subsets(args)` can declare expected subset names for logging.

### Common parser arguments

The parser receives runtime arguments from `preprocess.py`. Common fields include:

- `args.parser_config` — parsed JSON config provided by `--parser_config`.
- `args.default_subset_name` — default subset name used when the parser returns a simple subset string.

## Usage Summary

### Running `data/formatting/preprocess.py` with a parser

```bash
python data/formatting/preprocess.py \
    --datasets_dir data/raw_parquet \
    --output_dir data/parsed \
    --parser_path ./data/formatting/parsers/<parser>.py \
    --parser_config '{...}'
```

`--parser_config` supports either:

- a JSON string, or
- a path to a JSON file

If no parser config is needed, omit `--parser_config`.

## Example Parsers

### `add_uuid_parser.py`

Adds a UUID field to every sample by writing a new metadata column.

Main parameters:
- `uuid_column` — optional string name of the UUID metadata column.
  - default: `uuid`

Behavior:
- `parse_document` always returns a dictionary with `subset` set to `args.default_subset_name`.
- it adds `metadata: {uuid_column: <generated-uuid>}` to each sample.

Optional config:
```json
{
  "uuid_column": "uuid"
}
```

Example:
```bash
python data/formatting/preprocess.py \
    --datasets_dir data/raw_parquet \
    --output_dir data/parsed \
    --parser_path ./data/formatting/parsers/add_uuid_parser.py
```

### `score_threshold_filter_parser.py`

Drops samples when a configured score column is missing or below a configured threshold.

Main parameters:
- `score_column` — name of the score field in `doc.metadata`.
- `minimum_score` — minimum acceptable score value.

Behavior:
- if `doc.metadata[score_column]` is missing or less than `minimum_score`, the parser returns `None` and the sample is dropped.
- otherwise, it returns `args.default_subset_name` to keep the original row.

Required config:
```json
{
  "score_column": "edu_int_score",
  "minimum_score": 3
}
```

Example:
```bash
python data/formatting/preprocess.py \
    --datasets_dir data/raw_parquet \
    --output_dir data/parsed \
    --parser_path ./data/formatting/parsers/score_threshold_filter_parser.py \
    --parser_config '{"score_column": "edu_int_score", "minimum_score": 3}'
```

## Combining Parsers with Stratification

A parser can be combined with `--stratify_by_column` in `preprocess.py`.

In that mode:

- the parser still decides whether a sample is kept or dropped
- the parser can still add metadata fields
- the final subset is derived from the value in the stratification column

Example:
```bash
python data/formatting/preprocess.py \
    --datasets_dir data/raw_parquet \
    --output_dir data/parsed \
    --stratify_by_column edu_int_score \
    --parser_path ./data/formatting/parsers/add_uuid_parser.py
```

## Notes

- Parsers should be lightweight and deterministic unless randomness is explicitly intended.
- If a parser rewrites the full row, it is responsible for preserving any fields it still wants in output.
- If a parser only needs to add metadata, returning a `metadata` dict is usually the simplest option.
- Parser config is passed through `--parser_config` as either a JSON string or a path to a JSON file.

## Adding New Parsers

To add a new parser:

1. Create a new Python file in this folder.
2. Implement `parse_document(doc, args)`.
3. Optionally add `setup(args)` and `resolve_subsets(args)`.
4. Run it through [`../preprocess.py`](../preprocess.py) with `--parser_path`.
5. Document any required `--parser_config` fields in the parser docstring.
