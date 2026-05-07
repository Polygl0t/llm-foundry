# Data Pipelines (Common Crawl)

CommonCrawl WARC archive processing for multilingual text extraction with optional quality filtering.

## Overview

This folder contains pipelines for processing CommonCrawl WARC archives to extract and filter multilingual text data. The pipelines support language-specific quality filtering, token counting, and some other features. The output is organized into language-separated JSONL files with associated metadata.

## Contents

- **process_cc_dump_all_languages.py** — Single-stage pipeline for language identification and text extraction from CommonCrawl WARC archives. Extracts text for all languages or a subset of specified languages.
- **process_cc_dump_with_quality_filters.py** — Two-stage pipeline with quality filtering per language (repetition, punctuation, word quality, encoding fixes).
- **splitter.py** — Utility to split large JSONL files into manageable chunks based on token count thresholds.
- **utils.py** — Shared utility functions for logging, metadata management, and WARC file handling.
- **langcodes.py** — Language code mappings and identifiers.
- **warc_files_download.sh** — Helper script to download CommonCrawl WARC files.
- **warc_paths_get.sh** — Helper script to retrieve WARC file paths from CommonCrawl index.

## Pipeline Stages

### Single-Stage Extraction (process_cc_dump_all_languages.py)

1. **WARC Reading** — Parse CommonCrawl WARC.gz files
2. **URL Filtering** — Apply optional blocklists for spam/adult content
3. **Text Extraction** — Clean HTML using Trafilatura (precision mode)
4. **Token Counting** — Count tokens using specified tokenizer
5. **Language Detection** — FT176 (176 languages) or GlotLID (1665 languages)
6. **Output** — Write to language-separated JSONL files

### Two-Stage Filtering (process_cc_dump_with_quality_filters.py)

**Stage 1 — WARC Extraction:**
1. Read WARC files from CommonCrawl dumps
2. Filter URLs (optional blocklists for spam/adult content)
3. Extract clean text using Trafilatura (precision-focused)
4. Perform initial language detection (FT176 — 176 languages)
5. Write language-separated intermediate files

**Stage 2 — Quality Filtering (per-language):**
1. Secondary language detection (GlotLID — 1665 languages)
2. Language score thresholding (custom per language)
3. Gopher Repetition Filter (line/n-gram deduplication)
4. FineWeb Quality Filter (punctuation, newlines, char duplicates)
5. Gopher Quality Filter (word length, stop words, alpha ratio)
6. Formatting cleanup (FTFY encoding fixes, PII removal, symbol lines)
7. Token counting with selected tokenizer
8. Output to final JSONL files

## Running Pipelines

### Single-Stage Extraction

```bash
python data/cc/process_cc_dump_all_languages.py \
  --warc_files_folder /data/cc/CC-MAIN-2025-30/ \
  --output_folder all_languages/ \
  --dump CC-MAIN-2025-30 \
  --tasks 32 --workers 32
```

With language filtering:

```bash
python data/cc/process_cc_dump_all_languages.py \
  --warc_files_folder /data/cc/warc/ \
  --output_folder extracted/ \
  --dump CC-MAIN-2025-30 \
  --languages pt \
  --language_filter_backend glotlid \
  --language_threshold 0.7
```

### Two-Stage Quality Filtering

```bash
python data/cc/process_cc_dump_with_quality_filters.py \
  --warc_files_folder /data/cc/CC-MAIN-2025-30/ \
  --config_folder .configs/ \
  --final_output_folder output/ \
  --dump CC-MAIN-2025-30 \
  --languages pt \
  --tasks 32 --workers 32
```

### Splitting Large Files

```bash
python data/cc/splitter.py --directory output/pt/
```

Splits files over 1 GB into chunks of 100M tokens by default.

### With SLURM

```bash
sbatch data/cc/process_cc_dump_all_languages.sh
sbatch data/cc/process_cc_dump_with_quality_filters.sh
```

## Configuration

### Language Configuration

Language-specific quality filter configurations are stored in `.configs/` directory. To add support for additional languages:

1. Create a config file in `.configs/{language_code}.yaml`
2. Add the language to `LANG_CONFIG_MAPPING` in the processing script
3. Update `LANG_FILTER_MAPPING` with filter thresholds

## Notes

- Ensure WARC files are downloaded and accessible in the specified folder
- Adjust language filter backend and threshold based on desired precision/recall
- Scripts can be re-submitted if jobs fail; they support resuming from checkpoints
- Output fields are customizable via `KEEP_KEYS` in processing scripts
- The splitter automatically skips files already chunked (ending with `-chunk-`)