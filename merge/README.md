# Model Merging

Scripts and configuration utilities for merging multiple language models using MergeKit and token surgery techniques.

## Contents

- [`merge.sh`](merge.sh) — SLURM job submission script for merging models using MergeKit.
- [`tokensurgeon.sh`](tokensurgeon.sh) — SLURM job submission script for performing token surgery on model vocabularies.
- [`mergekit_config.yml`](mergekit_config.yml) — Example MergeKit configuration file defining merge strategy and model inputs.

## Usage Summary

### `merge.sh`

Submits a SLURM job to merge multiple language models using MergeKit. Supports various merging strategies (e.g., linear interpolation, SLERP, task arithmetic).

Example:
```bash
sbatch merge.sh
```

Configure the following in the script before submission:
- `--account` — Your SLURM account
- `--partition` — Target GPU partition
- `--nodes` — Number of compute nodes
- `--ntasks-per-node` — Number of GPUs per node
- `config` — Path to MergeKit configuration file (e.g., `mergekit_config.yml`)
- `output_dir` — Directory to save the merged model
- `model_dtype` — Data type for merged model (`float32`, `float16`, `bfloat16`)

The merge configuration in `mergekit_config.yml` defines:
- `merge_method` — Strategy to use: `linear`, `slerp`, `task_arithmetic`, `ties`, or `dare_ties`
- `models` — List of models to merge with their weights/paths
- `base_model` — Base model reference (if using task arithmetic or DARE)
- `parameters` — Method-specific parameters

### `tokensurgeon.sh`

Submits a SLURM job to perform token surgery on model vocabularies. This is useful for extending or adapting tokenizers across merged models.

Example:
```bash
sbatch tokensurgeon.sh
```

Configure the following in the script before submission:
- `--account` — Your SLURM account
- `--partition` — Target GPU partition
- `--nodes` — Number of compute nodes
- `--ntasks-per-node` — Number of GPUs per node
- `model_path` — Path or HuggingFace hub ID of the model to modify
- `output_dir` — Directory to save the modified model
- `operation` — Token surgery operation to perform (e.g., `extend`, `resize`, `merge_vocab`)
- `target_vocab_size` — Target vocabulary size after token surgery

## SLURM Cluster Jobs

The `.sh` scripts are configured for SLURM-based GPU clusters. Before submitting, update the following variables in each script:

- `--account` — Your SLURM account
- `--partition` — Your target partition
- `--nodes` — Number of nodes required
- `--ntasks-per-node` — Number of GPUs per node
- `--cpus-per-task` — CPU cores per GPU task
- `--mem` — Memory per node
- `--time` — Maximum job runtime

Submit a job with:
```bash
sbatch merge.sh
```

## MergeKit Configuration

The `mergekit_config.yml` file defines how models are merged. Example configuration:

```yaml
merge_method: linear
models:
  - model_name_or_path: model_1
    parameters:
      weight: 0.5
  - model_name_or_path: model_2
    parameters:
      weight: 0.5
output_dir: ./merged_model
```

Main supported merge methods:
- See https://github.com/arcee-ai/mergekit#merge-methods.
