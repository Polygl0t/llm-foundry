
# Data Tokenization

Tokenization, packing, decontamination, and validation split utilities for pretraining and SFT datasets.

## Contents

- [`decontaminate.py`](decontaminate.py) — Removes training examples that overlap with reference/evaluation sets via k-token matching.
- [`make_validation_split.py`](make_validation_split.py) — Extracts a validation split from tokenized training files.
- [`pack.py`](pack.py) — Packs pre-tokenized sequences into fixed-length chunks using concatenation or Best-Fit Decreasing (BFD) strategies.
- [`run_tokenization.py`](run_tokenization.py) — Tokenizes datasets for pretraining or SFT, with optional chat template formatting and assistant mask generation.
- [`utils.py`](utils.py) — Shared utilities (dataset loading, logging, saving) used across scripts.

---

## Usage Summary

### `decontaminate.py`

Removes examples from a training dataset that contain k-token sequences found in reference datasets (e.g., test/validation sets) to prevent data leakage.

Example:
```bash
python decontaminate.py \
    --input_dir data/tokenized_train \
    --reference_path eval_set/ \
    --output_dir cleaned_data \
    --min_k 8 --max_k 32 --allow_one_token_mismatch
```

Main parameters:
- `--input_dir` — Directory with contaminated dataset shards.
- `--reference_path` — Reference dataset (file, directory, or HF id).
- `--output_dir` — Output directory for cleaned data.
- `--min_k` / `--max_k` — Min/max token span for matching.
- `--allow_one_token_mismatch` — Allow 1-token substitutions in matches.
- `--approx_max_k` — Max k for masked matching (default: 10).
- `--output_type` — Output format: `jsonl` or `parquet`.
- `--num_proc` — Number of processes for dataset operations.

### `make_validation_split.py`

Creates validation splits by extracting a specified number of samples from multiple training data files and consolidating them into a separate validation file.

Example:
```bash
python make_validation_split.py \
    --input_dirs data/train_chunks \
    --output_dir data/validation \
    --input_type parquet \
    --output_file validation_split \
    --n_samples 20000
```

Main parameters:
- `--input_dirs` — One or more directories containing input files.
- `--output_dir` — Directory to save the validation split and metadata.
- `--input_type` — Input file type: `parquet` or `json`.
- `--output_file` — Filename for the validation split.
- `--n_samples` — Total number of samples to remove for validation.
- `--n_files` — Number of files to randomly select (default: all).
- `--seed` — Random seed for reproducibility.

### `pack.py`

Packs a pre-tokenized dataset into fixed-length chunks using either concatenation or Best-Fit Decreasing (BFD) strategies.

Examples:
```bash
# Concatenation
python pack.py \
    --input_path data/data_tokenized \
    --output_dir data/data_packed \
    --strategy concatenate \
    --block_size 4096

# Best-Fit Decreasing (BFD)
python pack.py \
    --input_path data/data_tokenized \
    --output_dir data/data_packed \
    --strategy bfd \
    --block_size 4096 \
    --pad_token_id 0
```

Main parameters:
- `--input_path` — Tokenized dataset source.
- `--output_dir` — Directory for packed dataset.
- `--strategy` — Packing strategy: `concatenate` or `bfd`.
- `--block_size` — Target sequence length for each chunk.
- `--pad_token_id` — Token ID for padding (required for `bfd`).
- `--max_tokens` — Truncate output to at most this many tokens.
- `--output_type` — Output format: `parquet` or `jsonl`.
- `--num_proc` — Number of parallel workers.

### `run_tokenization.py`

Tokenizes datasets for both pretraining (causal LM) and supervised fine-tuning (SFT). Supports standard text tokenization and chat-template formatting.

Examples:
```bash
# Pretraining tokenization
python run_tokenization.py \
    --input_path data/pretrain_raw \
    --output_dir data/pretrain_tokenized \
    --tokenizer_name Qwen/Qwen3-0.6B \
    --add_bos_token --add_eos_token \
    --return_seq_lengths \
    --max_length 8192

# SFT tokenization
python run_tokenization.py \
    --input_path data/sft_raw \
    --output_dir data/sft_tokenized \
    --tokenizer_name Qwen/Qwen3-0.6B \
    --text_column messages \
    --apply_chat_template \
    --return_seq_lengths \
    --return_labels \
    --return_assistant_masks
```

Main parameters:
- `--input_path` — Dataset source: local file, directory, or HuggingFace Hub id.
- `--output_dir` — Directory to write the tokenized dataset.
- `--tokenizer_name` — Name or path of the tokenizer.
- `--text_column` — Column containing text/messages to tokenize (default: `text`).
- `--apply_chat_template` — Apply chat template (for SFT).
- `--add_bos_token` / `--add_eos_token` — Add BOS/EOS tokens (not with chat template).
- `--return_seq_lengths` — Include sequence lengths in output.
- `--return_labels` — Include labels for loss computation.
- `--return_assistant_masks` — Include assistant masks (requires chat template).
- `--max_length` — Discard sequences longer than this.
- `--max_tokens` — Truncate output to at most this many tokens.
- `--output_type` — Output format: `parquet` or `jsonl` (default: `parquet`).
- `--num_proc` — Number of parallel workers (default: 8).

### `utils.py`

Shared utilities for tokenization and packing scripts, including dataset loading, logging, saving, and file management.

## SLURM Cluster Jobs

The `.sh` scripts are configured for SLURM-based GPU clusters. Before submitting, update the following variables in each script:

- `--account` — Your SLURM account
- `--partition` — Your target partition
- `username`, `file_system`, `workspace_name` — Paths to your working directory

```bash
sbatch run_tokenization.sh
```

## Notes

- Decontamination operates at the **token level** — both training and reference datasets must be pre-tokenized with `input_ids`.
