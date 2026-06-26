# Data Filters

Dataset filtering and annotation pipelines for text corpus curation. This folder contains scripts for language filtering, deduplication, quality filtering, SFT dataset cleaning, and annotator training/inference — covering the full filtering workflow from raw web data to curated training sets.

## Contents

- [`language_filter.py`](./language_filter.py) — Filters datasets by language using one of two backends: `langdetect` (probabilistic) or `unicode` (Unicode character-range heuristics).
- [`minhash.py`](./minhash.py) — MinHash-based fuzzy deduplication pipeline using DataTrove and LSH.
- [`quality_filters.py`](./quality_filters.py) — Multi-stage quality filtering pipeline using FastText, GlotLID, Gopher, and FineWeb quality checks.
- [`sft_filters.py`](./sft_filters.py) — Filters and cleans instruction-tuning datasets (malformed code, repetition loops, Unicode issues, etc.).
- [`train_annotator.py`](./train_annotator.py) — Trains regression-based sequence classification models for annotation tasks.
- [`run_annotator.py`](./run_annotator.py) — Runs inference with a trained annotator on a dataset.

## Usage Summary

### `language_filter.py`

Language-based filtering with two backends selectable via `--backend`:

- **`langdetect`** — probabilistic detection using the `langdetect` library. More accurate on short or ambiguous text; requires `langdetect` to be installed.
- **`unicode`** — heuristic filtering using Unicode character-range matching. Fast, deterministic, and requires no extra dependencies. Use `--unicode_threshold` (default `0.85`) to control strictness.

Examples:
```bash
# langdetect backend
python data/filters/language_filter.py \
    --backend langdetect \
    --input_dir data/ --output_dir filtered/ \
    --languages portuguese english \
    --input_type jsonl --output_type jsonl \
    --text_column text --num_proc 16

# unicode backend
python data/filters/language_filter.py \
    --backend unicode \
    --input_dir data/ --output_dir filtered/ \
    --languages portuguese \
    --unicode_threshold 0.85 \
    --text_column text --num_proc 16
```

Main parameters:
- `--backend` — filtering backend: `langdetect` or `unicode` (required).
- `--input_dir` — directory containing the input dataset files.
- `--output_dir` — directory to save filtered dataset files.
- `--languages` — list of languages to keep (e.g. `portuguese`, `english`).
- `--input_type` / `--output_type` — choose `jsonl` or `parquet`.
- `--text_column` — name of the text field in the dataset.
- `--cache_dir` — optional cache directory for dataset loading.
- `--num_proc` — number of processes to use (default: all available CPUs).
- `--save_excluded` — save excluded samples instead of included ones (useful for debugging).
- `--unicode_threshold` — minimum fraction of in-range characters required to keep a sample (`unicode` backend only, default: `0.85`).

### `minhash.py`

Scalable fuzzy deduplication using MinHash signatures and Locality-Sensitive Hashing.

Example:
```bash
python data/filters/minhash.py \
    --data_folder raw_data/ --language pt \
    --output_deduplication_final deduplicated/ \
    --tokenizer_name_or_path Qwen/Qwen3-0.6B \
    --tasks 32 --workers 32
```

Main parameters:
- `--data_folder` — path to the input data folder.
- `--language` — language code.
- `--tokenizer_name_or_path` — tokenizer used for token counting and deduplication.
- `--tasks` — number of pipeline tasks.
- `--workers` — number of worker processes.
- `--cache_dir` — cache directory for datasets.
- `--expand_metadata` — expand output metadata during processing.
- `--output_minhash_signatures` — output folder for MinHash signatures.
- `--output_minhash_bucket` — output folder for bucket clustering.
- `--output_removed_ids` — output folder for removed duplicate IDs.
- `--output_duplicated_samples` — output folder for duplicate examples.
- `--output_deduplication_final` — output folder for final deduplicated dataset.

### `quality_filters.py`

Multi-stage quality filtering for large corpus cleaning with language-specific configuration files.

Example:
```bash
python data/filters/quality_filters.py \
    --data_folder raw/ --final_output_folder filtered/ \
    --language pt --config_folder .configs/ \
    --tokenizer_name_or_path Qwen/Qwen3-0.6B \
    --tasks 32 --workers 32
```

Main parameters:
- `--data_folder` — path to raw input data.
- `--final_output_folder` — directory for filtered output.
- `--language` — language code (`pt`, `bn`, `hi`).
- `--config_folder` — folder containing language configuration files.
- `--tokenizer_name_or_path` — tokenizer used for quality scoring and token counts.
- `--tasks` — number of pipeline tasks.
- `--workers` — number of worker processes.
- `--cache_dir` — cache directory for datasets.
- `--logs_folder` — directory for logs.
- `--expand_metadata` — include additional metadata fields in output.

### `sft_filters.py`

Instruction-tuning dataset filtering and cleaning for SFT corpora.

Example:
```bash
python data/filters/sft_filters.py \
    --input_dir ./raw_data --output_dir ./filtered_data \
    --input_type jsonl --output_type jsonl \
    --filter_malformed_code_blocks \
    --filter_repetition_loops \
    --filter_undecoded_sequences \
    --remove_system_messages
```

Main parameters:
- `--input_dir` — directory containing dataset files.
- `--output_dir` — directory to save cleaned dataset.
- `--input_type` / `--output_type` — choose `jsonl` or `parquet`.
- `--cache_dir` — optional cache directory for datasets.
- `--max_tokens_per_chunk` — maximum token count per output chunk.
- `--filter_incomplete_sentences` — remove samples whose final message lacks terminal punctuation.
- `--max_token_count` / `--min_token_count` — filter by token count range.
- `--filter_malformed_code_blocks` — remove malformed or invalid code block samples.
- `--filter_corrupted_code` — remove code with corrupted characters or invalid source text.
- `--filter_undecoded_sequences` — remove samples containing undecoded Unicode escape sequences.
- `--filter_invalid_markers` — remove samples with invalid structural markers.
- `--remove_system_messages` — strip system messages before processing.
- `--filter_repetition_loops` — remove samples stuck in repetitive model loops.
- `--quality_score_column` / `--min_quality_score` — filter by quality score fields if present.
- `--messages_column` — column name for message arrays.
- `--token_count_column` — column name for token count metadata.
- `--num_proc` — number of processes to use.


### `train_annotator.py`

Trains sequence classification / regression annotator models for dataset scoring.

Example:
```bash
python data/filters/train_annotator.py \
    --train_dataset_dir scored_data.jsonl \
    --model_name microsoft/deberta-v3-base \
    --text_column text --target_column score \
    --checkpoint_dir checkpoints/ --num_train_epochs 20
```

Main parameters:
- `--dataset_type` — `jsonl` or `parquet` input format.
- `--train_dataset_dir` — path(s) to training dataset directories or files.
- `--shuffle_dataset` — shuffle dataset files before loading.
- `--cache_dir` — path for dataset/model cache.
- `--num_proc` — number of parallel dataset workers.
- `--target_column` — name of the target score column.
- `--text_column` — name of the text input column.
- `--test_size` — number of evaluation samples.
- `--seed` — randomness seed.
- `--model_name` — model name or path for training.
- `--chat_template_path` — optional Jinja template for chat formatting.
- `--id_label` — label for the classification task.
- `--checkpoint_dir` — output checkpoint directory.
- `--resume_from_checkpoint` — resume training from an existing checkpoint.
- `--max_length` — maximum tokenizer sequence length.
- `--freeze` — freeze model weights except the classifier head.
- `--learning_rate`, `--weight_decay`, `--adam_beta1`, `--adam_beta2`, `--adam_epsilon` — optimizer settings.
- `--num_train_epochs` — number of training epochs.
- `--eval_steps`, `--save_steps`, `--logging_steps` — logging/evaluation frequency.
- `--bf16`, `--tf32`, `--gradient_checkpointing` — precision and memory options.
- `--per_device_train_batch_size`, `--per_device_eval_batch_size`, `--gradient_accumulation_steps` — batch and accumulation settings.
- `--report_to`, `--wandb_project` — experiment reporting configuration.

### `run_annotator.py`

Runs inference with a trained annotator model and writes scores back to output files.

Example:
```bash
python data/filters/run_annotator.py \
    --model_name username/edu-classifier \
    --dataset_path data/ --text_column text \
    --output_folder scored/ --batch_size 32
```

Main parameters:
- `--model_name` — annotator model name or path.
- `--apply_chat_template` — apply a chat template to each text input.
- `--dataset_path` — dataset directory or file path (`jsonl`/`parquet`).
- `--token` — optional token for protected datasets.
- `--cache_dir` — cache directory for the dataset.
- `--text_column` — name of the text column.
- `--num_proc` — number of processing workers.
- `--batch_size` — inference batch size.
- `--max_length` — maximum token length for tokenization.
- `--float_score` / `--int_score` — output column names for scores.
- `--output_folder` — output directory for annotated results.

## SLURM Cluster Jobs

The `.sh` scripts in this folder are configured for SLURM-based GPU clusters. Before submitting, update the following variables in each script:

- `--account` — your SLURM account
- `--partition` — target partition
- `username`, `file_system`, `workspace_name` — workspace-specific paths

Example:

```bash
sbatch data/filters/minhash.sh
```
