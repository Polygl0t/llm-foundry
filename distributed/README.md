# Distributed Training

This folder contains distributed training scripts for large language models using PyTorch's DDP (Distributed Data Parallel) and FSDP (Fully Sharded Data Parallel) strategies. Both are optimized for multi-GPU, multi-node SLURM clusters and support standard AdamW or hybrid Muon + Adam optimizers.

## Contents

- [train_ddp.py](train_ddp.py) — Distributed Data Parallel (DDP) training script for transformer-based causal language models. Handles multi-GPU synchronization with gradient accumulation and checkpointing.
- [train_fsdp.py](train_fsdp.py) — Fully Sharded Data Parallel (FSDP) training script for larger models requiring parameter and optimizer state sharding across nodes.
- [trainer.py](trainer.py) — Contains `DDPTrainer` and `FSDPTrainer` classes that encapsulate the training and validation loops, checkpointing, and per-step logging.
- [model_setup.py](model_setup.py) — Pre-DDP/FSDP model and tokenizer initialization, including architecture setup and optional context extension for continual pretraining.
- [data_loading.py](data_loading.py) — Dataset loading and DataLoader creation with support for multiple data formats (JSONL, Parquet).
- [optimizers.py](optimizers.py) — Optimizer and learning rate scheduler creation for both AdamW and Muon + Adam configurations.
- [mfu.py](mfu.py) — Model FLOPs Utilization (MFU) calculation utilities for performance monitoring and benchmarking.
- [specifications.py](specifications.py) — Dataclass definitions and type hints for all training arguments.
- [specifications.yaml](specifications.yaml) — Example YAML configuration file for training settings.
- [utils.py](utils.py) — Logging, checkpointing, distributed environment setup, and miscellaneous utilities.

## Usage Summary

### `train_ddp.py`

Distributed Data Parallel (DDP) training for transformer-based causal language models using PyTorch DDP with multi-GPU/multi-node synchronization, gradient accumulation, and checkpointing support.

Examples:

```bash
torchrun --nproc_per_node=4 distributed/train_ddp.py \
    --specs distributed/specifications.yaml \
    --slurm-job-id my_job_001 \
    --hardware a100

# DDP training with multi-node setup via SLURM
sbatch distributed/train_ddp.sh
```

Main parameters:
- See [specifications.py](specifications.py) files for detailed argument definitions and defaults.

### `train_fsdp.py`

Fully Sharded Data Parallel (FSDP2) training for large language models using PyTorch FSDP with parameter and optimizer state sharding across nodes. Supports zero-stage 2 (parameter sharding) and zero-stage 3 (full sharding).

Examples:

```bash
# Basic FSDP training on 4 GPUs
torchrun --nproc_per_node=4 distributed/train_fsdp.py \
    --specs distributed/specifications.yaml \
    --slurm-job-id my_job_001 \
    --hardware a100

# FSDP training with multi-node setup via SLURM
sbatch distributed/train_fsdp.sh
```

Main parameters:
- See [specifications.py](specifications.py) files for detailed argument definitions and defaults.

### Validation-only runs

To submit a job that only evaluates a model on the validation split, set `eval_only: true` in `specifications.yaml` and launch the usual DDP or FSDP job script. When `resume_from_checkpoint` is set, the checkpoint is loaded first and the validation log uses the checkpoint's restored step; otherwise the initialized or base model is evaluated at step 0.

```yaml
eval_only: true
resume_from_checkpoint: /path/to/checkpoint
```

Validation-only jobs run one validation pass, write validation stats, and exit without consuming training batches, running backward, taking optimizer steps, or saving a new checkpoint.

## SLURM Cluster Jobs

The `.sh` scripts are configured for SLURM-based GPU clusters. Key configuration variables:

```bash
# Example SLURM directives in train_ddp.sh / train_fsdp.sh
#SBATCH --account=your_account
#SBATCH --partition=sgpu_devel
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=480GB
#SBATCH --time=01:00:00
```

Update the following in each shell script before submission:
- `--account` — Your SLURM account
- `--partition` — Target partition/queue
- `--nodes` — Number of compute nodes
- `--ntasks-per-node` — Number of GPUs per node
- `username`, `workspace_name` — Paths to your working directory and model checkpoint locations

## Example Architecture Configs

Here we have toy examples of model config files covering the supported architectures. Each config is a `transformers`-compatible JSON that can be passed directly to `path_to_model_config` in `specifications.yaml`.

- **NOTE**: For Qwen3.5 hybrids that mix linear-attention layers with full attention (`layer_types` containing `"linear_attention"`), the fast path in the modeling code requires **both** `flash-linear-attention` (gated-delta-rule chunk / fused kernels) **and** `causal-conv1d` (short-conv branch of `GatedDeltaNet`). Install both with `pip install flash-linear-attention causal-conv1d`. If either is missing, training still runs but falls back to a slow PyTorch reference path.

<details>
<summary><strong>Dense Transformer</strong> — <code>LlamaForCausalLM</code> · Dense transformer · <a href="https://huggingface.co/docs/transformers/model_doc/llama">HF Docs</a></summary>

```json
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "hidden_act": "silu",
  "hidden_size": 512,
  "initializer_range": 0.02,
  "intermediate_size": 1536,
  "is_llama_config": true,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 8,
  "num_hidden_layers": 8,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-05,
  "rope_interleaved": false,
  "rope_scaling": null,
  "rope_theta": 100000,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "vocab_size": 49152
}
```

</details>

<details>
<summary><strong>Mixture of Experts</strong> — <code>Qwen3MoeForCausalLM</code> · Mixture of Experts · <a href="https://huggingface.co/docs/transformers/model_doc/qwen3_moe">HF Docs</a></summary>

```json
{
  "architectures": [
    "Qwen3MoeForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "decoder_sparse_step": 1,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 512,
  "initializer_range": 0.02,
  "intermediate_size": 1536,
  "max_position_embeddings": 4096,
  "max_window_layers": 8,
  "mlp_only_layers": [],
  "model_type": "qwen3_moe",
  "moe_intermediate_size": 384,
  "norm_topk_prob": true,
  "num_attention_heads": 8,
  "num_experts": 8,
  "num_experts_per_tok": 2,
  "num_hidden_layers": 8,
  "num_key_value_heads": 8,
  "output_router_logits": false,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 100000,
  "router_aux_loss_coef": 0.001,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 49152
}
```

</details>

<details>
<summary><strong>Qwen3.5 Dense (Full Attention)</strong> — <code>Qwen3_5ForCausalLM</code> · Dense transformer with full attention · <a href="https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5">HF Docs</a></summary>

```json
{
  "architectures": [
    "Qwen3_5ForCausalLM"
  ],
  "model_type": "qwen3_5_text",
  "vocab_size": 49152,
  "hidden_size": 512,
  "intermediate_size": 1536,
  "num_hidden_layers": 8,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "head_dim": 64,
  "hidden_act": "silu",
  "max_position_embeddings": 4096,
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-06,
  "use_cache": true,
  "tie_word_embeddings": true,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "partial_rotary_factor": 0.25,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "rope_parameters": null,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "pad_token_id": 0,
  "torch_dtype": "bfloat16"
}
```

</details>

<details>
<summary><strong>Qwen3.5 Hybrid (Linear + Full Attention)</strong> — <code>Qwen3_5ForCausalLM</code> · Dense hybrid · <a href="https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5">HF Docs</a></summary>

Every 4th layer is full attention, the rest use Gated-DeltaNet linear attention.

```json
{
  "architectures": [
    "Qwen3_5ForCausalLM"
  ],
  "model_type": "qwen3_5_text",
  "vocab_size": 49152,
  "hidden_size": 512,
  "intermediate_size": 1536,
  "num_hidden_layers": 8,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "head_dim": 64,
  "hidden_act": "silu",
  "max_position_embeddings": 4096,
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-06,
  "use_cache": true,
  "tie_word_embeddings": true,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "partial_rotary_factor": 0.25,
  "layer_types": [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention"
  ],
  "linear_num_key_heads": 8,
  "linear_num_value_heads": 16,
  "linear_key_head_dim": 64,
  "linear_value_head_dim": 64,
  "linear_conv_kernel_dim": 4,
  "rope_parameters": null,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "pad_token_id": 0,
  "torch_dtype": "bfloat16"
}
```

</details>

<details>
<summary><strong>Qwen3.5 MoE (Full Attention)</strong> — <code>Qwen3_5MoeForCausalLM</code> · MoE transformer with full attention · <a href="https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5_moe">HF Docs</a></summary>

```json
{
  "architectures": [
    "Qwen3_5MoeForCausalLM"
  ],
  "model_type": "qwen3_5_moe_text",
  "vocab_size": 49152,
  "hidden_size": 512,
  "intermediate_size": 1536,
  "num_hidden_layers": 8,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "head_dim": 64,
  "hidden_act": "silu",
  "max_position_embeddings": 4096,
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-06,
  "use_cache": true,
  "tie_word_embeddings": true,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "partial_rotary_factor": 0.25,
  "layer_types": [
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention",
    "full_attention"
  ],
  "num_experts": 8,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 384,
  "shared_expert_intermediate_size": 384,
  "norm_topk_prob": true,
  "output_router_logits": false,
  "router_aux_loss_coef": 0.001,
  "rope_parameters": null,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "pad_token_id": 0,
  "torch_dtype": "bfloat16"
}
```

</details>

<details>
<summary><strong>Qwen3.5 MoE Hybrid (Linear + Full Attention)</strong> — <code>Qwen3_5MoeForCausalLM</code> · MoE hybrid · <a href="https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5_moe">HF Docs</a></summary>

Every 4th layer is full attention, the rest use Gated-DeltaNet linear attention; MLPs are routed mixture-of-experts with an optional shared expert.

```json
{
  "architectures": [
    "Qwen3_5MoeForCausalLM"
  ],
  "model_type": "qwen3_5_moe_text",
  "vocab_size": 49152,
  "hidden_size": 512,
  "intermediate_size": 1536,
  "num_hidden_layers": 8,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "head_dim": 64,
  "hidden_act": "silu",
  "max_position_embeddings": 4096,
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-06,
  "use_cache": true,
  "tie_word_embeddings": true,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "partial_rotary_factor": 0.25,
  "layer_types": [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention"
  ],
  "linear_num_key_heads": 8,
  "linear_num_value_heads": 16,
  "linear_key_head_dim": 64,
  "linear_value_head_dim": 64,
  "linear_conv_kernel_dim": 4,
  "num_experts": 8,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 384,
  "shared_expert_intermediate_size": 384,
  "norm_topk_prob": true,
  "output_router_logits": false,
  "router_aux_loss_coef": 0.001,
  "rope_parameters": null,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "pad_token_id": 0,
  "torch_dtype": "bfloat16"
}
```

</details>

