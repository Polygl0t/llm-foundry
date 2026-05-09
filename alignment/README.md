# Alignment 

Alignment-related training scripts and utilities for the post-training phase/fine-tuning of large language models.

## Overview

This folder contains scripts to perform supervised fine-tuning (SFT), direct preference optimization (DPO), and (WIP) Group Relative Policy Optimization (GRPO) on language models.

## Contents

- **sft_trainer.py** — Supervised fine-tuning of LLMs using Hugging Face Transformers and TRL.
- **dpo_trainer.py** — Direct Preference Optimization (DPO) training using chosen/rejected response pairs.
- **vibes.py** — Inference testing across multiple task types (useful for quick vibe checks).
- **gym/** — Generation, validation, and verification utilities for gym-based (RL-style) pipelines.

## Running Trainers

```bash
# Supervised Fine-Tuning
python sft_trainer.py \
    --model_name_or_path Qwen/Qwen3-0.6B-Base \
    --train_dataset_dir data/train \
    --checkpoint_dir checkpoints/llama-sft \
    --max_length 4096 \
    --packing --assistant_only_loss \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3

# Direct Preference Optimization
python dpo_trainer.py \
    --train_dataset_dir data/preferences.jsonl \
    --model_name_or_path Qwen/Qwen3-0.6B-Base \
    --checkpoint_dir checkpoints/ \
    --loss_type sigmoid --beta 0.1 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 1

# Inference Testing
python vibes.py \
    --model_path MyModel \
    --output_file "MyModel/inference_samples.json" \
    --samples_file "samples.json" \
    --max_new_tokens 1024 \
    --temperature 0.2
```

## SLURM Cluster Jobs

The `.sh` scripts are configured for SLURM-based GPU clusters. Before submitting, update the following variables in each script:

- `--account` — Your SLURM account
- `--partition` — Your target partition
- `username`, `file_system`, `workspace_name` — Paths to your working directory

```bash
sbatch sft_trainer.sh
sbatch ... # (same command for other scripts)
```

## Dataset Formats

**SFT** expects chat-formatted messages or pre-tokenized input:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**DPO** expects chosen/rejected response pairs:
```json
{"prompt": "...", "chosen": [{"role": "assistant", "content": "..."}], "rejected": [{"role": "assistant", "content": "..."}]}
```

**Vibes** expects a JSON array of sample objects with `messages` and `task_type` fields.

## Notes

- See `gym/README.md` for details on the gym-based generation and verification pipelines.
