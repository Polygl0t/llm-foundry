# Tokenizer Training and Evaluation

This folder contains the tokenizer training and evaluation stack for creating custom tokenizers and managing chat templates. Supports both [SentencePiece](https://github.com/google/sentencepiece) and HuggingFace [tokenizers](https://github.com/huggingface/tokenizers) library implementations.

## Contents

- [`assets/`](assets/) — Directory containing Jinja2 chat template configurations for various use cases (reasoning, non-reasoning, text, hybrid).
- [`chat_template.ipynb`](chat_template.ipynb) — Jupyter notebook for configuring and testing chat templates with different tokenizers.
- [`tokenizer_eval.py`](tokenizer_eval.py) — Evaluation script for comparing and analyzing tokenizer performance, compression ratios, and token efficiency.
- [`train_tokenizer_sentencepiece.py`](train_tokenizer_sentencepiece.py) — SentencePiece tokenizer training script for creating custom subword tokenizers with configurable vocabulary and merge operations.
- [`train_tokenizer_tokenizers.py`](train_tokenizer_tokenizers.py) — HuggingFace `tokenizers` library-based tokenizer training for BPE, WordPiece, and other tokenization algorithms.
- [`utils.py`](utils.py) — Utility functions for loading datasets, updating tokenizer configs, and validating saved tokenizers.


## Usage Summary

### `train_tokenizer_sentencepiece.py`

SentencePiece tokenizer training for creating custom subword tokenizers from raw text data.

Example:
```bash
python tokenizer/train_tokenizer_sentencepiece.py \
  --input_file data/raw_text.txt \
  --output_dir outputs/tokenizers \
  --vocab_size 32000 \
  --model_type bpe
```

Main parameters:
- `--input_file` — Path to input text file for training (required).
- `--output_dir` — Directory to save tokenizer model (required).
- `--vocab_size` — Vocabulary size for the tokenizer (default: 32000).
- `--model_type` — Tokenizer model type: `bpe`, `unigram`, or `char` (default: `bpe`).
- `--character_coverage` — Character coverage threshold for training (default: 0.9995).
- `--unk_id` — Unknown token ID (default: 0).
- `--bos_id` — Beginning of sequence token ID (default: 1).
- `--eos_id` — End of sequence token ID (default: 2).
- `--pad_id` — Padding token ID (default: 3).
- `--model_name` — Name for the output model file (default: `sentencepiece`).

### `train_tokenizer_tokenizers.py`

HuggingFace `tokenizers` library-based tokenizer training for modern tokenization algorithms.

Example:
```bash
python tokenizer/train_tokenizer_tokenizers.py \
  --input_file data/raw_text.txt \
  --output_dir outputs/tokenizers \
  --vocab_size 32000 \
  --tokenizer_type bpe
```

Main parameters:
- `--input_file` — Path to input text file for training (required).
- `--output_dir` — Directory to save tokenizer model (required).
- `--vocab_size` — Vocabulary size for the tokenizer (default: 32000).
- `--tokenizer_type` — Tokenizer algorithm: `bpe`, `wordpiece`, or `unigram` (default: `bpe`).
- `--min_frequency` — Minimum frequency threshold for tokens (default: 2).
- `--special_tokens` — List of special tokens to add (default: `['[UNK]', '[CLS]', '[SEP]', '[MASK]', '[PAD]']`).
- `--model_name` — Name for the output model file (default: `tokenizer`).

### `tokenizer_eval.py`

Evaluation and comparison tool for analyzing tokenizer performance.

Example:
```bash
python tokenizer/tokenizer_eval.py \
  --tokenizer_path outputs/tokenizers/sentencepiece.model \
  --eval_dataset data/eval_text.txt \
  --output_file outputs/tokenizer_metrics.json
```

Main parameters:
- `--tokenizer_path` — Path to trained tokenizer model (required).
- `--eval_dataset` — Path to evaluation dataset (required).
- `--output_file` — Output file for evaluation metrics (default: `tokenizer_metrics.json`).
- `--compute_compression_ratio` — Compute compression ratio vs raw bytes (default: enabled).
- `--compute_entropy` — Compute token distribution entropy (default: enabled).
- `--max_samples` — Maximum samples to evaluate (-1 = all) (default: -1).

### `chat_template.ipynb`

Interactive Jupyter notebook for configuring and testing chat templates with different tokenizers and models.

Features:
- Load and test custom tokenizers
- Apply chat templates to conversation samples
- Visualize token sequences and special tokens
- Compare template outputs across different formats

## SLURM Cluster Jobs

The `.sh` scripts are configured for SLURM-based GPU clusters (when applicable). Before submitting, update the following variables in each script:

- `--account` — Your SLURM account
- `--partition` — Your target partition
- `username`, `file_system`, `workspace_name` — Paths to your working directory

```bash
sbatch tokenizer/train_tokenizer_sentencepiece.sh
```

## Chat Templates

Chat template files in `assets/` define Jinja2 templates for formatting conversations:

- `chat_template.jinja` — Default chat template for standard conversations
- `chat_template_reasoning.jinja` — Template optimized for reasoning/thinking chains
- `chat_template_non_reasoning.jinja` — Template for non-reasoning conversational data
- `chat_template_hybrid_reasoning.jinja` — Template supporting mixed reasoning and conversational modes
- `chat_template_text.jinja` — Basic text template (i.e., concatenates all messages without special formatting)

