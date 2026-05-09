# Data Tokenization

Tokenization, packing, decontamination, and validation split utilities for pretraining and SFT datasets.

## Overview

This folder contains scripts for the full post-filtering, pre-training data preparation pipeline: tokenizing raw text or chat datasets, packing sequences into fixed-length chunks, removing benchmark contamination, and creating validation splits.

## Contents

- **tokenize.py** — Tokenizes datasets for pretraining or SFT, with optional chat template formatting and assistant mask generation.
- **pack.py** — Packs pre-tokenized sequences into fixed-length chunks using concatenation or Best-Fit Decreasing (BFD) strategies.
- **decontaminate.py** — Removes training examples that overlap with reference/evaluation sets via k-token matching.
- **make_validation_split.py** — Extracts a validation split from tokenized training files.
- **utils.py** — Shared utilities (dataset loading, logging, saving) used across scripts.

## Running Scripts

```bash
# Pretraining tokenization
python tokenize.py \
    --input_path data/pretrain_raw \
    --output_dir data/pretrain_tokenized \
    --tokenizer_name Qwen/Qwen3-0.6B \
    --add_bos_token --add_eos_token \
    --return_seq_lengths \
    --max_length 8192

# SFT tokenization (with chat template and assistant masks)
python tokenize.py \
    --input_path data/sft_raw \
    --output_dir data/sft_tokenized \
    --tokenizer_name Qwen/Qwen3-0.6B \
    --text_column messages \
    --apply_chat_template \
    --return_seq_lengths \
    --return_labels \
    --return_assistant_masks

# Pack with concatenation strategy
python pack.py \
    --input_path data/data_tokenized \
    --output_dir data/data_packed \
    --strategy concatenate \
    --block_size 4096

# Pack with Best-Fit Decreasing strategy
python pack.py \
    --input_path data/data_tokenized \
    --output_dir data/data_packed \
    --strategy bfd \
    --block_size 4096 \
    --pad_token_id 0

# Decontaminate against evaluation sets
python decontaminate.py \
    --input_pattern "data/*.jsonl" \
    --reference_files eval_set.jsonl test_set.jsonl \
    --output_dir cleaned_data \
    --min_k 8 --max_k 32 --allow_one_token_mismatch

# Create a validation split
python make_validation_split.py \
    --input_dir data/train_chunks \
    --output_dir data/validation \
    --input_type parquet \
    --output_file validation_split \
    --n_samples 20000
```

## SLURM Cluster Jobs

The `.sh` scripts are configured for SLURM-based GPU clusters. Before submitting, update the following variables in each script:

- `--account` — Your SLURM account
- `--partition` — Your target partition
- `username`, `file_system`, `workspace_name` — Paths to your working directory

```bash
sbatch tokenize.sh
sbatch ... # (same command for other scripts)
```

## Notes

- Decontamination operates at the **token level** — both training and reference datasets must be pre-tokenized with `input_ids`.

