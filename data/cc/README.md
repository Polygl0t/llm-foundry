# Data Pipelines (Common Crawl)

CommonCrawl WARC archive processing for multilingual text extraction with optional quality filtering. This folder contains scripts to run single-stage and two-stage extraction pipelines.

## Contents

- [`/.configs`](./.configs) — per-language filter configuration files.
- [`/slurm`](./slurm) — Folder containing SLURM job scripts for cluster-managed environments. Before submitting, update the scripts with your cluster-specific settings and correct paths for your artifacts/workspace. **These are templates, not ready-to-run scripts.**
- [`process_cc_dump_all_languages.py`](process_cc_dump_all_languages.py) — single-stage extraction + language filtering.
- [`process_cc_dump_with_quality_filters.py`](process_cc_dump_with_quality_filters.py) — two-stage extraction + per-language quality filtering.
- [`splitter.py`](splitter.py) — chunk large JSONL files using token counts.
- [`utils.py`](utils.py) — shared WARC, metadata, and logging helpers.
- [`langcodes.py`](langcodes.py) — supported language codes and backend mappings.
- [`warc_files_download.sh`](warc_files_download.sh) — helper to download CommonCrawl WARC files.
- [`warc_paths_get.sh`](warc_paths_get.sh) — helper to retrieve WARC paths from CommonCrawl indexes.

## Pipeline Stages

### `process_cc_dump_all_languages.py`

This script runs a single-stage CommonCrawl extraction pipeline (useful to gather as many languages as possible with minimal filtering):

1. `WarcReader` reads `*.warc.gz` files from `--warc_files_folder`.
2. `URLFilter` removes pages that match blocklists or unwanted patterns.
3. `Trafilatura` extracts cleaned text from HTML.
4. `TokensCounter` counts tokens with a tokenizer from `--tokenizer_name_or_path`.
5. `LanguageFilter` detects language using `--language_filter_backend` and `--language_threshold`.
6. `JsonlWriter` writes output into language-separated directories under `--output_folder`.

### `process_cc_dump_with_quality_filters.py`

This script runs a two-stage extraction pipeline with a second quality filtering stage (useful for higher-quality outputs in a smaller set of languages):

- Stage 1: WARC extraction and intermediate JSONL output by language (using FT176 for LID).
- Stage 2: Per-language filtering using:
  - `LanguageFilter` with GlotLID labels and language score thresholds.
  - `GopherRepetitionFilter` for duplicate lines/n-grams.
  - `FineWebQualityFilter` for punctuation, newline, and duplicate character checks.
  - `GopherQualityFilter` for word-length, stop-word, and alpha-ratio checks.
  - `FTFYFormatter`, `PIIFormatter`, and `SymbolLinesFormatter` for cleanup.
  - `TokensCounter` and `JsonlWriter` to produce the final output.

### `splitter.py`

Splits large JSONL files in a folder into smaller chunks when they exceed `--size_threshold_gb`.

- Uses the `token_count` field from each JSONL object to allocate records into chunks.
- Writes chunk files named `{hash}-chunk-{N}.jsonl`.
- Skips files already containing `-chunk-`.

## Usage Summary

### Single-Stage Extraction

Example:
```bash
python data/cc/process_cc_dump_all_languages.py \
  --warc_files_folder /data/cc/CC-MAIN-2025-30/ \
  --output_folder output/all_languages/ \
  --dump CC-MAIN-2025-30 \
  --tasks 32 --workers 32
```

Main parameters:
- `--warc_files_folder`: folder containing CommonCrawl `*.warc.gz` files.
- `--dump`: CommonCrawl dump identifier, e.g. `CC-MAIN-2025-30`.
- `--output_folder`: final destination for extracted language folders.
- `--temp_output_folder`: temporary folder for intermediate output.
- `--logs_folder`: folder for pipeline logs.
- `--languages`: optional list of language ISO codes.
- `--language_filter_backend`: `ft176` or `glotlid`.
- `--language_threshold`: minimum language score for filtering.
- `--tokenizer_name_or_path`: tokenizer used for token counting (default `Qwen/Qwen3-0.6B`).
- `--tasks`, `--workers`: workflow parallelism settings.
- `--expand_metadata`: include expanded metadata in output JSONL.
- `--limit`: process only a subset of WARC files for debugging.

### Two-Stage Quality Filtering

Example:
```bash
python data/cc/process_cc_dump_with_quality_filters.py \
  --warc_files_folder /data/cc/CC-MAIN-2025-30/ \
  --config_folder .configs/ \
  --final_output_folder output/ \
  --dump CC-MAIN-2025-30 \
  --languages pt bn \
  --tasks 32 --workers 32
```

Main parameters:
- `--warc_files_folder`: folder containing CommonCrawl `*.warc.gz` files.
- `--config_folder`: folder with per-language filter config YAMLs.
- `--final_output_folder`: final destination for filtered output.
- `--output_file`: file path for combined output JSONL.
- `--warc_extraction_output`: intermediate folder for stage-1 extraction.
- `--quality_filter_output`: intermediate folder for stage-2 filtering.
- `--dump`: CommonCrawl dump identifier.
- `--tokenizer_name_or_path`: tokenizer used for token counting (default `Qwen/Qwen3-0.6B`).
- `--languages`: list of FT176 language codes to process.
- `--tasks`, `--workers`: workflow parallelism settings.
- `--expand_metadata`: include expanded metadata in output JSONL.
- `--limit`: process only a subset of WARC files for debugging.

### Splitting Large Files

Example:
```bash
python data/cc/splitter.py --directory output/pt/
```

Main parameters:
- `--directory`: directory containing JSONL files to split.
- `--max_tokens_per_chunk`: maximum number of tokens per chunk (default `100000000`).
- `--size_threshold_gb`: split only files larger than this size in GB (default `1.0`).

## Configuration

### Language configuration files

Language-specific configuration files live in [`.configs/`](./.configs) and are referenced by [`process_cc_dump_with_quality_filters.py`](./process_cc_dump_with_quality_filters.py).

To add a new language:

1. Add a YAML config to `.configs/{language_code}.yaml`.
2. Add the language entry to `LANG_CONFIG_MAPPING` in [`process_cc_dump_with_quality_filters.py`](./process_cc_dump_with_quality_filters.py).
3. Update `LANG_FILTER_MAPPING` with the language label.

### Output structure

- `--output_folder` for [`process_cc_dump_all_languages.py`](./process_cc_dump_all_languages.py) contains per-language directories and JSONL files.
- `--warc_extraction_output` and `--quality_filter_output` are intermediate folders for the two-stage pipeline.
- `--final_output_folder` contains the final filtered per-language outputs.

## Notes

- Ensure CommonCrawl WARC files are downloaded and available at the path passed to `--warc_files_folder`.
- Default tokenizer is `Qwen/Qwen3-0.6B`.
- `splitter.py` avoids re-splitting files already containing `-chunk-`.
- `langcodes.py` defines supported language codes for filtering.
- `utils.py` is used by both scripts for logging and metadata updates.
