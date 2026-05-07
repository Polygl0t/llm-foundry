# Synthetic Data Generation

Synthetic data generation stack for creating training data using vLLM-powered inference on Hugging Face language models.

## Overview

This folder contains the synthetic data generation stack for creating training data using vLLM-powered inference on Hugging Face language models.

## Contents

- **generate.py** — General-purpose synthetic text generation with vLLM inference.
- **generate_cai.py** — Constitutional AI (CAI) based synthetic data generation with critique and revision.
- **generate_datatrove.py** — Datatrove-based inference pipeline for large-scale synthetic data generation.
- **utils.py** — Shared utility functions including dataset loading, model initialization, and sampling utilities.

## Running Generation

From the repository root:

### General Synthesis

```bash
python synthetic/generate.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_path data/seed_texts.jsonl \
  --output_dir outputs/synthetic
```

### Constitutional AI Synthesis

```bash
python synthetic/generate_cai.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_path data/prompts.jsonl \
  --constitution_file constitutions/helpful_honest.md \
  --output_dir outputs/cai_data
```

### Datatrove-Based Inference

```bash
python synthetic/generate_datatrove.py \
  --input-path /data/documents \
  --model-name-or-path Qwen/Qwen3-0.6B \
  --output-path /data/summaries
```

### With SLURM

```bash
sbatch synthetic/generate.sh
sbatch synthetic/generate_cai.sh
sbatch synthetic/generate_datatrove.sh
```

Before submitting jobs, update `account`, `partition`, and `workdir` values in the respective `.sh` files.

## Configuration

### Common Arguments

Key arguments for generation scripts (common across all generators):

- **Model**: `--model_name_or_path` (HuggingFace model identifier)
- **Dataset**: `--dataset_path` (local file, directory, or HuggingFace Hub identifier)
- **Output**: `--output_dir` and `--output_file` (where to write generated data)
- **Generation**: `--max_length`, `--temperature`, `--top_p`, `--top_k` (sampling parameters)
- **Inference**: `--tensor_parallel_size`, `--gpu_memory_utilization` (vLLM configuration)
- **Dataset-specific**: `--text_column`, `--prompt_column`, `--metadata_columns` (field mapping)

### CAI-Specific Arguments

- `--constitution_file` — Path to constitution with principles/rules for guided generation
- `--enable_think` — Enable extended thinking for deeper reasoning
- `--enable_critique` — Enable critique-revision loop for output refinement
- `--max_revisions` — Maximum revision iterations per sample

### Datatrove-Specific Arguments

- `--prompt-template` — Template with `[[DOCUMENT]]` placeholder for inference
- **Input format** — JSONL or Parquet files
- **Auto-resume** — Automatically resumes from checkpoints on job interruption

## Environment Notes

- All generators use vLLM for high-throughput inference with tensor parallelism support.
- The SLURM launchers configure per-job cache directories to avoid cache collisions.
- Triton cache is automatically configured during execution for optimal performance.
- Dataset loading supports local files (JSONL/Parquet), local directories, and HuggingFace Hub datasets.
- Output files are JSONL format for easy downstream processing.
- Resume capability: Interrupted jobs can continue from the last completed sample by re-running the same command.
