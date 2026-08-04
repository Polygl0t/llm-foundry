# Distributed Training

This folder contains distributed training scripts for large language models using PyTorch's DDP (Distributed Data Parallel) and FSDP (Fully Sharded Data Parallel) strategies. Both are optimized for multi-GPU, multi-node SLURM clusters and support standard AdamW or hybrid Muon + Adam optimizers.

## Contents

- [`/slurm`](./slurm) — Folder containing SLURM job scripts for cluster-managed environments. Before submitting, update the scripts with your cluster-specific settings and correct paths for your artifacts/workspace. **These are templates, not ready-to-run scripts.**
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
sbatch train_ddp.sh
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
sbatch train_fsdp.sh
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
# - Marvin:
#SBATCH --account=your_account
#SBATCH --partition=sgpu_devel
#SBATCH --job-name=ddp-training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=480GB
#SBATCH --time=01:00:00
#
# - Bender:
#SBATCH --partition=A40short
#SBATCH --job-name=ddp-training
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --gpus=2
```

For Marvin update the following in each shell script before submission:
- `--account` — Your SLURM account
- `--partition` — Target partition/queue
- `--nodes` — Number of compute nodes
- `--ntasks-per-node` — Number of GPUs per node
- `username`, `workspace_name` — Paths to your working directory and model checkpoint locations

On Bender, set the `--partition`, `--gpus`, and `--cpus-per-task` directives according to your job requirements.

## Installation

Before running the training scripts, ensure that the required Python packages are installed in your environment. Use the provided [`create_venv_marvin.sh`](slurm/create_venv_marvin.sh) or [`create_venv_bender.sh`](slurm/create_venv_bender.sh) scripts to create a virtual environment and install dependencies.

```bash
# For Marvin:
bash distributed/slurm/create_venv_marvin.sh

# For Bender:
bash distributed/slurm/create_venv_bender.sh
```

Remember to adapt the scripts (especially the paths!) to your specific cluster environment.

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
  "output_router_logits": true,
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
  "output_router_logits": true,
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
  "output_router_logits": true,
  "router_aux_loss_coef": 0.001,
  "rope_parameters": null,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "pad_token_id": 0,
  "torch_dtype": "bfloat16"
}
```

</details>

## Benchmarks

Training throughput measured on 2-GPU nodes (seq len 4096, bfloat16). TPS and dt are per-GPU figures for a single optimization step.

| Model Class                              | GPU          | Context Length | Batch Size          | Total Params | Active Params | VRAM     | MFU     | TPS / GPU | Step dt  |
|------------------------------------------|--------------|----------------|---------------------|--------------|---------------|----------|---------|-----------|----------|
| `LlamaForCausalLM`                       | A100 2×80 GB | 4096           | 256 total (128/GPU) | 52.4 M       | 52.4 M        | 71.64 GB | 62.15 % | 361,389   | 1,451 ms |
| `LlamaForCausalLM`                       | A40 2×48 GB  | 4096           | 128 total (64/GPU)  | 52.4 M       | 52.4 M        | 36.05 GB | 56.21 % | 164,473   | 1,594 ms |
| `Qwen3_5ForCausalLM` (full attention)    | A100 2×80 GB | 4096           | 128 total (64/GPU)  | 52.4 M       | 52.4 M        | 52.16 GB | 45.30 % | 263,734   |   994 ms |
| `Qwen3_5ForCausalLM` (full attention)    | A40 2×48 GB  | 4096           | 64 total (32/GPU)   | 52.4 M       | 52.4 M        | 26.31 GB | 40.00 % | 116,379   | 1,123 ms |
| `Qwen3_5ForCausalLM` (hybrid 3:1)        | A100 2×80 GB | 4096           | 128 total (64/GPU)  | 58.8 M       | 58.8 M        | 61.69 GB | 42.19 % | 279,797   |   937 ms |
| `Qwen3_5ForCausalLM` (hybrid 3:1)        | A40 2×48 GB  | 4096           | 64 total (32/GPU)   | 58.8 M       | 58.8 M        | 31.10 GB | 36.25 % | 115,735   | 1,134 ms |
| `Qwen3_5ForCausalLM` (full linear)       | A100 2×80 GB | 4096           | 128 total (64/GPU)  | 61 M         | 61 M          | 65.88 GB | 42.20 % | 286,253   |   915 ms |
| `Qwen3_5ForCausalLM` (full linear)       | A40 2×48 GB  | 4096           | 64 total (32/GPU)   | 61 M         | 61 M          | 33.20 GB | 34.50 % | 112,514   | 1,164 ms |
| `Qwen3_5MoeForCausalLM` (full attention) | A100 2×80 GB | 4096           | 128 total (64/GPU)  | 76.1 M       | 47.8 M        | 63.05 GB | 34.52 % | 211,553   | 1,233 ms |
| `Qwen3_5MoeForCausalLM` (full attention) | A40 2×48 GB  | 4096           | 64 total (32/GPU)   | 76.1 M       | 47.8 M        | 24.26 GB | 33.75 % | 103,772   | 1,263 ms |
| `Qwen3_5MoeForCausalLM` (hybrid 3:1)     | A100 2×80 GB | 4096           | 128 total (64/GPU)  | 82.5 M       | 54.1 M        | 57.43 GB | 34.18 % | 241,045   | 1,087 ms |
| `Qwen3_5MoeForCausalLM` (hybrid 3:1)     | A40 2×48 GB  | 4096           | 64 total (32/GPU)   | 82.5 M       | 54.1 M        | 29.06 GB | 31.66 % | 107,326   | 1,221 ms |
| `Qwen3_5MoeForCausalLM` (full linear)    | A40 2×48 GB  | 4096           | 64 total (32/GPU)   | 84.6 M       | 56.3 M        | 61.40 GB | 35.28 % | 257,200   | 1,011 ms |
| `Qwen3_5MoeForCausalLM` (full linear)    | A40 2×48 GB  | 4096           | 64 total (32/GPU)   | 84.6 M       | 56.3 M        | 31.05 GB | 30.34 % | 105,426   | 1,243 ms |
| `Qwen3MoeForCausalLM`   (full attention) | A100 2×80 GB | 4096           | 128 total (64/GPU)  | 79.7 M       | 51.4 M        | 70.84 GB | 43.10 % | 189,089   | 1,386 ms |
| `Qwen3MoeForCausalLM`   (full attention) | A40 2×48 GB  | 4096           | 64 total (32/GPU)   | 79.7 M       | 51.4 M        | 35.75 GB | 39.54 % | 83,385    | 1,571 ms |


> - **Note:** The table values are rounded from the raw benchmark logs. `LlamaForCausalLM` was benchmarked at 2× the total batch size of the other models. Active params equal total params for dense models; MoE models activate 2 of 8 experts per token.


## Expected Bugs and Quirks

### MoE Error on Bender

If your training a MoE model on Bender and encounter this error:

```
ValueError: atomic_add does not support bf16
```

This happens because Liger-Kernel uses `tl.atomic_add` to accumulate gradients ([source](https://github.com/linkedin/Liger-Kernel/blob/v0.8.0/src/liger_kernel/ops/fused_moe.py)), but Triton's `atomic_add` does not support bfloat16 with the version of Triton we use in this stack (3.2.0). Since we train our models in bf16 precision, the training crashes on the first backward pass.

- **Note:** Support for bf16 in `atomic_add` is added in Triton 3.4.0 ([source](https://github.com/triton-lang/triton/releases/tag/v3.4.0)).

**Possible Solution 1:**

Upgrade Triton to 3.4.0 or later to get bf16 support in `atomic_add`. However, this may require upgrading PyTorch to a version that supports Triton 3.4.0, which may not be possible on Bender because of CUDA version constraints (max is 12.4 in Bender), the old GLIBC version that Bender is running (2.28), and flash attention's pickyness.

**Possible Solution 2:**

Disable swiglu when applying the liger kernel to MoE models.

in [`model_setup.py`](./model_setup.py):

```python
...

def _apply_liger_kernels(model, args):
    """
    Apply Liger kernels to the model for optimized performance.

    Liger's RoPE replacement is only valid for HF rotary embedding modules
    with the standard interface (Llama / Qwen3 / Qwen2.5 ...). Qwen3.5 uses
    a customized rotary embedding (partial rotation, per-layer shapes) that
    is not compatible with Liger's RoPE kernel, so we disable it there.
    """
    liger_transformers = importlib.import_module("liger_kernel.transformers")
    apply_liger_kernel = getattr(liger_transformers, "_apply_liger_kernel_to_instance")
    model_type = str(getattr(model.config, "model_type", "") or "")
    rope_compatible = not model_type.startswith("qwen3_5")
    liger_kwargs = {
        "rope": rope_compatible,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
        "rms_norm": True,
        "swiglu": False,  # Set to False to avoid atomic_add bf16 error in MoE models on Bender
    }
    apply_liger_kernel(model=model, **liger_kwargs)

...
```

### Qwen3.5 Linear Attention NaN Loss

For reasons of divine mystery, when we set `Qwen3_5ForCausalLM` or `Qwen3MoeForCausalLM` to use linear attention on ALL layers, on some seeds, the loss becomes `NaN` after the first step. This is very strange, since the same config with 7 linear layers + 1 full-attention layer works fine, and the linear attention implementation is identical in both cases. We are investigating this issue, but for now, if you want to train a Qwen3.5 model with linear attention, be prepared to join the lottery and try different seeds until you find one that doesn't produce NaN loss.

### Bender + Linear Attention Kernels

We currently cannot get the `flash-linear-attention` and `causal-conv1d` packages to install on Bender due to CUDA version constraints. Flash-linear-attention requires PyTorch >= 2.7.0, but the latest CUDA available on Bender is CUDA 12.4, which is not compatible with the release versions of PyTorch 2.7.x. As a result, we cannot use the optimized kernels for linear attention on Bender at this time.
