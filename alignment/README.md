# Alignment

Alignment-related training scripts and utilities for the post-training phase and fine-tuning of large language models. This folder includes trainers for Direct Preference Optimization (DPO), Group Relative Policy Optimization (GRPO), reward model training, and supervised fine-tuning (SFT), along with shared utilities and a subfolder for gym generation and verification used in GRPO.

## Contents

- [`/gym`](./gym) — Codebase to generate tasks with verifiable rewards for RLVR-style training.
- [`/slurm`](./slurm) — Folder containing SLURM job scripts for cluster-managed environments. Before submitting, update the scripts with your cluster-specific settings and correct paths for your artifacts/workspace. **These are templates, not ready-to-run scripts.**
- [`dpo_trainer.py`](./dpo_trainer.py) — DPO training with chosen/rejected response pairs.
- [`grpo_trainer.py`](./grpo_trainer.py) — GRPO training using verifier-based rewards from `alignment/gym/`.
- [`reward_trainer.py`](./reward_trainer.py) — Reward model training using TRL's `RewardTrainer`.
- [`sft_trainer.py`](./sft_trainer.py) — Supervised fine-tuning of LLMs using Transformers and TRL.
- [`utils.py`](./utils.py) — Shared helper functions used by the alignment scripts.

## Usage Summary

### `dpo_trainer.py`

Direct Preference Optimization training pipeline.

Example:
```bash
python alignment/dpo_trainer.py \
  --train_dataset_dir data/preferences.jsonl \
  --model_name_or_path Qwen/Qwen3-0.6B-Base \
  --checkpoint_dir checkpoints/dpo \
  --loss_type sigmoid --beta 0.1 \
  --per_device_train_batch_size 4 \
  --num_train_epochs 1
```

Main parameters:
- `--train_dataset_dir` — Path(s) to the training dataset directory or file.
- `--dataset_type` — `jsonl` or `parquet`.
- `--model_name_or_path` — Model identifier or local path.
- `--ref_model_name_or_path` — Optional reference model for DPO loss computation.
- `--checkpoint_dir` — Directory to save checkpoints.
- `--loss_type` — Loss variant(s) to use, e.g. `sigmoid`, `apo_zero`, `hinge`, `nca_pair`.
- `--beta` — KL coefficient or reference deviation weight for certain DPO losses.
- `--precompute_ref_log_probs` — Precompute reference log probabilities for efficiency.
- `--max_length` — Maximum sequence length for tokenization/model input.
- `--max_prompt_length` — Maximum prompt length before tokenization.
- `--padding_free` — Use padding-free batches to reduce memory usage.
- `--per_device_train_batch_size` — Training batch size per device.
- `--gradient_accumulation_steps` — Number of steps to accumulate gradients.
- `--learning_rate`, `--weight_decay`, `--adam_beta1`, `--adam_beta2`, `--adam_epsilon` — Optimizer settings.
- `--num_train_epochs` — Number of training epochs.
- `--bf16`, `--tf32`, `--gradient_checkpointing` — Mixed-precision and memory settings.

### `sft_trainer.py`

Supervised fine-tuning trainer.

Example:
```bash
python alignment/sft_trainer.py \
  --model_name_or_path Qwen/Qwen3-0.6B-Base \
  --train_dataset_dir data/train \
  --checkpoint_dir checkpoints/llama-sft \
  --max_length 4096 \
  --packing --assistant_only_loss \
  --per_device_train_batch_size 4 \
  --num_train_epochs 3
```

Main parameters:
- `--train_dataset_dir` — Path(s) to the training dataset directory or file.
- `--dataset_type` — `jsonl` or `parquet`.
- `--model_name_or_path` — Model identifier or local path.
- `--checkpoint_dir` — Output checkpoint directory.
- `--packing` — Pack variable-length samples to improve training efficiency.
- `--assistant_only_loss` — Compute loss only on assistant responses.
- `--pad_to_multiple_of` — Pad sequences to a multiple of this value.
- `--max_length` — Maximum sequence length for tokenization/model input.
- `--per_device_train_batch_size` — Training batch size per device.
- `--per_device_eval_batch_size` — Evaluation batch size per device.
- `--learning_rate`, `--weight_decay`, `--adam_beta1`, `--adam_beta2`, `--adam_epsilon` — Optimizer settings.
- `--num_train_epochs` — Number of training epochs.
- `--bf16`, `--tf32`, `--activation_offloading`, `--gradient_checkpointing` — Memory and precision options.

### `reward_trainer.py`

Reward model training pipeline.

Example:
```bash
python alignment/reward_trainer.py \
  --train_dataset_dir data/preferences.jsonl \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --checkpoint_dir checkpoints/reward-model \
  --per_device_train_batch_size 4 \
  --num_train_epochs 1
```

Main parameters:
- `--train_dataset_dir` — Path(s) to the training dataset directory or file.
- `--dataset_type` — `jsonl` or `parquet`.
- `--model_name_or_path` — Model identifier or local path.
- `--checkpoint_dir` — Output checkpoint directory.
- `--chat_template_path` — Optional chat template for conversational reward datasets.
- `--max_length` — Maximum sequence length for tokenization/model input.
- `--center_rewards_coefficient` — Reward centering scale.
- `--per_device_train_batch_size` — Training batch size per device.
- `--learning_rate`, `--weight_decay`, `--adam_beta1`, `--adam_beta2`, `--adam_epsilon` — Optimizer settings.
- `--num_train_epochs` — Number of training epochs.
- `--bf16`, `--tf32`, `--gradient_checkpointing` — Mixed-precision and memory optimization.

### `grpo_trainer.py`

Group Relative Policy Optimization trainer using verifier-based rewards from [`alignment/gym/`](./gym/).

Example:
```bash
python alignment/grpo_trainer.py \
  --train_dataset_dir path/to/dataset.jsonl \
  --dataset_type jsonl \
  --model_name_or_path Qwen/Qwen3-0.6B-Instruct \
  --checkpoint_dir checkpoints/grpo \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --num_generations 8 \
  --per_device_train_batch_size 4 \
  --num_train_epochs 1 \
  --verifier_enable_thinking --no-verifier_strict
```

Main parameters:
- `--train_dataset_dir` — Path(s) to the training dataset directory or file.
- `--dataset_type` — `jsonl` or `parquet`.
- `--model_name_or_path` — Model identifier or local path.
- `--checkpoint_dir` — Output checkpoint directory.
- `--max_prompt_length` — Maximum prompt token length used by the tokenizer.
- `--max_completion_length` — Maximum generated completion length.
- `--num_generations` — Number of completions sampled per prompt.
- `--num_iterations` — Optimization iterations per batch.
- `--beta` — KL coefficient for GRPO.
- `--loss_type` — GRPO loss variant (`dapo`, `grpo`, `bnpo`, `dr_grpo`, `sapo`).
- `--scale_rewards` — Reward scaling mode: `group`, `batch`, or `none`.
- `--verifier_enable_thinking` — Require a `<think>...</think>` reasoning block before verifier checks.
- `--verifier_strict` / `--no-verifier_strict` — Strict vs. relaxed verifier checking.
- `--mask_truncated_completions` — Mask completions that hit `max_completion_length` without EOS.
- `--temperature`, `--top_p`, `--top_k`, `--repetition_penalty` — Sampling settings.
- `--use_vllm`, `--vllm_mode` — Enable vLLM-based rollout generation.
- `--per_device_train_batch_size` — Training batch size per device.
- `--learning_rate`, `--weight_decay`, `--adam_beta1`, `--adam_beta2`, `--adam_epsilon` — Optimizer settings.

## Dataset Formats

### `sft_trainer.py`

Expected chat-formatted messages or pre-tokenized input:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

If the dataset is pre-tokenized, it may include:

```json
{"input_ids": [...], "seq_lengths": [...], "assistant_tokens_mask": [...]}
```

### `dpo_trainer.py` and `reward_trainer.py`

Expected chosen/rejected preference pairs:

```json
{
  "prompt": "...",
  "chosen": [{"role": "assistant", "content": "Good response"}],
  "rejected": [{"role": "assistant", "content": "Bad response"}]
}
```

### `grpo_trainer.py`

Expected verifier-driven prompts:

```json
{
  "prompt": "...",
  "verifier_id_list": ["math:answer_check"],
  "kwargs": ["{\"expected_answer\": \"42\", \"relaxed\": true}"]
}
```

[`grpo_trainer.py`](./grpo_trainer.py) uses [`alignment/gym/verifier.py`](./gym/verifier.py) to compute a reward from verifier results.
