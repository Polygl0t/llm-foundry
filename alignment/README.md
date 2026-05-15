# Alignment 

Alignment-related training scripts and utilities for the post-training phase/fine-tuning of large language models.

## Overview

This folder contains scripts to perform supervised fine-tuning (SFT), direct preference optimization (DPO), and (WIP) Group Relative Policy Optimization (GRPO) on language models.

## Contents

- **.*_config.yaml** — Example configuration files for distributed training via Accelerate.
- **gym/** — Generation, validation, and verification utilities for gym-based (RL-style) pipelines. See `gym/README.md` for details.
- **dpo_trainer.py** — Direct Preference Optimization (DPO) training using chosen/rejected response pairs.
- **grpo_trainer.py** — Group Relative Policy Optimization (GRPO) training using verifier-based rewards.
- **reward_trainer.py** — Reward model training using TRL's RewardTrainer.
- **sft_trainer.py** — Supervised fine-tuning of LLMs using Hugging Face Transformers and TRL.
- **utils.py** — Helper functions for the alignment scripts.

## Running Trainers

```bash
# Direct Preference Optimization
python dpo_trainer.py \
    --train_dataset_dir data/preferences.jsonl \
    --model_name_or_path Qwen/Qwen3-0.6B-Base \
    --checkpoint_dir checkpoints/ \
    --loss_type sigmoid --beta 0.1 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 1

# GRPO with verifier rewards
python grpo_trainer.py \
    --train_dataset_dir alignment/0.jsonl \
    --dataset_type jsonl \
    --model_name_or_path Qwen/Qwen3-0.6B-Instruct \
    --checkpoint_dir checkpoints/grpo \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 1

# Supervised Fine-Tuning
python sft_trainer.py \
    --model_name_or_path Qwen/Qwen3-0.6B-Base \
    --train_dataset_dir data/train \
    --checkpoint_dir checkpoints/llama-sft \
    --max_length 4096 \
    --packing --assistant_only_loss \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3

# Reward model training
python reward_trainer.py \
        --train_dataset_dir data/preferences.jsonl \
        --model_name_or_path Qwen/Qwen3-0.6B \
        --checkpoint_dir checkpoints/reward-model \
        --per_device_train_batch_size 4 \
        --num_train_epochs 1
```

## SLURM Cluster Jobs

The `.sh` scripts are configured for SLURM-based GPU clusters. Before submitting, update the following variables in each script:

- `--account` — Your SLURM account
- `--partition` — Your target partition
- `username`, `file_system`, `workspace_name` — Paths to your working directory

```bash
sbatch dpo_trainer.sh
sbatch ... # (same command for other scripts)
```

## Dataset Formats

**SFT** expects chat-formatted messages or pre-tokenized input:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**DPO** and **Reward Model** expect chosen/rejected response pairs:
```json
{"prompt": "...", "chosen": [{"role": "assistant", "content": "..."}], "rejected": [{"role": "assistant", "content": "..."}]}
```

**GRPO** expects prompts with verifier IDs and verifier kwargs:
```json
{"prompt": "...", "verifier_id_list": ["math:answer_check"], "kwargs": ["{\"expected_answer\": \"42\", \"relaxed\": true}"]}
```
