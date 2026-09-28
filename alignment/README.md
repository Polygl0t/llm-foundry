# Alignment

Alignment-related training scripts and utilities for the post-training phase and fine-tuning of large language models. This folder includes trainers for Direct Preference Optimization (DPO), Group Relative Policy Optimization (GRPO), reward model training, and supervised fine-tuning (SFT), along with shared utilities and a subfolder for gym generation and verification used in GRPO.

## Contents

- [`/configs`](./configs) — Configuration files for distributed training with Accelerate.
- [`/gym`](./gym) — Codebase to generate tasks with verifiable rewards for RLVR-style training.
- [`/slurm`](./slurm) — Folder containing SLURM job scripts for cluster-managed environments. Before submitting, update the scripts with your cluster-specific settings and correct paths for your artifacts/workspace. **These are templates, not ready-to-run scripts.**
- [`dpo_trainer.py`](./dpo_trainer.py) — DPO training with chosen/rejected response pairs.
- [`grpo_trainer.py`](./grpo_trainer.py) — GRPO training using verifier-based rewards from `alignment/gym/`.
- [`prebuilt_dataset.py`](./prebuilt_dataset.py) — Materialise a raw alignment dataset to disk once, so SFT/DPO/GRPO/reward jobs can memory-map it instead of rebuilding it from shards on every rank.
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
- `--truncation_mode` — Which end survives truncation: `keep_start` (default/future-proof) or `keep_end`.
- `--padding_free` — Use padding-free batches to reduce memory usage.
- `--per_device_train_batch_size` — Training batch size per device.
- `--gradient_accumulation_steps` — Number of steps to accumulate gradients.
- `--learning_rate`, `--weight_decay`, `--adam_beta1`, `--adam_beta2`, `--adam_epsilon` — Optimizer settings.
- `--num_train_epochs` — Number of training epochs.
- `--bf16`, `--tf32`, `--gradient_checkpointing` — Mixed-precision and memory settings.

Notes:
- **Liger kernel (`--use_liger_kernel`) cannot be combined with `--precompute_ref_log_probs` (the fused loss never materialises logits, so TRL rejects the pair and the reference model is forwarded on the fly instead), and it only implements the `--loss_type` values `sigmoid`, `hinge`, `apo_zero`, `apo_down`, `robust`, `exo_pair`, `nca_pair`, `bco_pair`, `sppo_hard` and `discopop`. TRL's `ipo`, `aot`, `aot_pair` and `sft` have no kernel.

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

### `prebuilt_dataset.py`

Build a train/validation split from raw shards and materialise it to disk once, so `sft_trainer.py` / `dpo_trainer.py` / `grpo_trainer.py` / `reward_trainer.py` can memory-map it at start-up via `--prebuilt_dataset_dir` instead of rebuilding it from the raw shards on every rank.

Example:
```bash
python alignment/prebuilt_dataset.py \
  --train_dataset_dir data/sft \
  --output_dir        data/sft_prebuilt \
  --dataset_type      jsonl \
  --test_size         0.02 \
  --seed              42 \
  --max_token_count   32768 \
  --num_proc          64
```

Then point a trainer at it with `--prebuilt_dataset_dir data/sft_prebuilt`. See [`alignment/slurm/prebuilt_dataset.sh`](./slurm/prebuilt_dataset.sh) for a SLURM job template.

Main parameters:
- `--train_dataset_dir` — One or more directories/files with the raw training shards. Repeat a directory to include its shards more than once.
- `--val_dataset_dir` — Explicit validation shards, mutually exclusive with `--test_size`.
- `--output_dir` — Root directory for the prebuilt dataset; `train/` and `validation/` subdirectories are created inside it.
- `--dataset_type` — `jsonl` or `parquet`.
- `--test_size` — Size of the validation split carved out of the training data: a fraction in (0, 1), or a row count if >= 1. Required unless `--val_dataset_dir` is given.
- `--val_split_mode` — How `--test_size` is spread over `--train_dataset_dir`: `uniform` (default) gives every folder the same number of validation rows; `global` draws one random sample after concatenating the folders. Ignored with `--val_dataset_dir`.
- `--seed` — Random seed for the train/validation split.
- `--validate_column` — Optional column that must be present in both splits (e.g. `messages` for SFT, `chosen` for DPO).
- `--overwrite` — Replace existing output directories.
- `--max_token_count` — Drop rows whose token count exceeds this value before the splits are built.
- `--token_count_column` — Column holding the token count read by `--max_token_count` (default: `token_count`).
- `--num_proc` / `--max_num_proc` — Requested and maximum worker processes per split.
- `--cache_dir` — Cache directory for the intermediate HuggingFace Arrow files.

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

Besides the dataset format specific to each fine-tuning method, it is also important to use the correct chat template. See [`tokenizer/chat_template.ipynb`](../tokenizer/chat_template.ipynb) for an explanation on chat templates.

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
