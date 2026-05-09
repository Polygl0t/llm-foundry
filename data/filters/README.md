# Data Filters

Dataset filtering and annotation pipelines for text corpus curation.

## Overview

This folder contains scripts for language filtering, deduplication, quality filtering, SFT dataset cleaning, and annotator training/inference — covering the full filtering workflow from raw web data to curated training sets.

## Contents

- **langdetect_language_filter.py** — Filters datasets by language using the `langdetect` library.
- **unicode_language_filter.py** — Filters datasets by character set validation using Unicode ranges for 18+ languages.
- **minhash.py** — MinHash-based fuzzy deduplication pipeline using DataTrove and LSH.
- **quality_filters.py** — Multi-stage quality filtering pipeline (FastText, GlotLID, Gopher, FineWeb) using DataTrove.
- **sft_filters.py** — Filters and cleans instruction-tuning (SFT) datasets (malformed code, repetition loops, Unicode issues, etc.).
- **train_annotator.py** — Trains regression-based sequence classification models for annotation tasks.
- **run_annotator.py** — Runs inference with a trained annotator on a dataset.

## Running Filters

```bash
# Language filtering with langdetect
python langdetect_language_filter.py --input_dir data/ --output_dir filtered/ \
    --languages portuguese english --num_proc 16

# Language filtering with Unicode ranges
python unicode_language_filter.py \
    --input_dir data/ --output_dir filtered/ \
    --languages portuguese --text_column text

# MinHash deduplication
python minhash.py \
    --data_folder raw_data/ --language pt \
    --output_deduplication_final deduplicated/ \
    --tokenizer_name_or_path Qwen/Qwen3-0.6B \
    --tasks 32 --workers 32

# Quality filtering
python quality_filters.py \
    --data_folder raw/ --final_output_folder filtered/ \
    --language pt --config_folder .configs/ \
    --tokenizer_name_or_path Qwen/Qwen3-0.6B \
    --tasks 32 --workers 32

# SFT dataset filtering
python sft_filters.py \
    --input_dir ./raw_data --output_dir ./filtered_data \
    --filter_malformed_code_blocks --filter_repetition_loops \
    --filter_undecoded_sequences --remove_system_messages

# Train an annotator classifier
python train_annotator.py \
    --train_dataset_dir scored_data.jsonl \
    --model_name microsoft/deberta-v3-base \
    --text_column text --target_column score \
    --checkpoint_dir checkpoints/ --num_train_epochs 20

# Run annotator inference
python run_annotator.py \
    --model_name username/edu-classifier \
    --dataset_path data/ --text_column text \
    --output_folder scored/ --batch_size 32
```

## SLURM Cluster Jobs

The `.sh` scripts are configured for SLURM-based GPU clusters. Before submitting, update the following variables in each script:

- `--account` — Your SLURM account
- `--partition` — Your target partition
- `username`, `file_system`, `workspace_name` — Paths to your working directory

```bash
sbatch langdetect_language_filter.sh
sbatch ... # (same command for other scripts)
```

