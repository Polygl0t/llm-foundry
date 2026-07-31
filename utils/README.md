# Utilities

This folder contains miscellaneous utility scripts and helpers for working with models, datasets, and training pipelines.

## Contents

- [`baf/`](./baf/) — Documentation for how to work with the BAF (aka, Optimus Prime) Cluster.
- [`jupiter/`](./jupiter/) — Documentation for how to work with the JSC Jupiter booster.
- [`/slurm`](./slurm) — Folder containing SLURM job scripts for cluster-managed environments. Before submitting, update the scripts with your cluster-specific settings and correct paths for your artifacts/workspace. **These are templates, not ready-to-run scripts.**
- [`compute_hyperparams.py`](./compute_hyperparams.py) — Compute training hyperparameters like learning rate, batch size, and steps based on model size and training configuration (uses the heuristics from the [DeepSeek LLM scaling laws paper](https://arxiv.org/abs/2401.02954)).
- [`convert_dataset_to_hf.py`](./convert_dataset_to_hf.py) — Convert JSONL or Parquet dataset shards into a Hugging Face Dataset format and optionally upload it to the Hub.
- [`count_tokens.py`](./count_tokens.py) — Create token count reports for a pretraining corpus.
- [`distributed_test.py`](./distributed_test.py) — GPU Communication Benchmark / LLM Gradient All-Reduce Simulation.
- [`download.py`](./download.py) — Download and cache Hugging Face repositories using patterns and authentication.
- [`env_doctor.sh`](./slurm/env_doctor.sh) — SLURM batch script to diagnose GPU/CUDA/PyTorch environment issues using env-doctor.
- [`extract_reasoning_traces.py`](./extract_reasoning_traces.py) — Extract reasoning traces from a conversational dataset (i.e., has `"messages"` array) and save them in JSONL format.
- [`inference_test.py`](./inference_test.py) — Run inference on a model using a sample dataset and save outputs.
- [`inspect_model.py`](./inspect_model.py) — Analyze model configuration, parameter counts, and MoE routing statistics.
- [`launch_trackio.py`](./launch_trackio.py) — Launch a Trackio dashboard for logging and visualizing training metrics.
- [`marvin_create_workspace.sh`](./slurm/marvin_create_workspace.sh) — Allocate a workspace and clone the repo on the Marvin HPC cluster.
- [`parse_run.py`](./parse_run.py) — Parse and summarize distributed training logs and emissions files.
- [`pdf2markdown.sh`](./slurm/pdf2markdown.sh) — Example SLURM batch job for converting PDFs to Markdown using Marker.
- [`reset_weights.py`](./reset_weights.py) — Reset selected model weights while optionally preserving or saving the modified model.
- [`resize_embedding_layer.py`](./resize_embedding_layer.py) — Validate and resize a model embedding layer to match tokenizer vocabulary size.
- [`upload_ckpts_to_hf.py`](./upload_ckpts_to_hf.py) — Upload checkpoint directories to the Hugging Face Hub, one branch per training step.
- [`upload_quick.py`](./upload_quick.py) — Quickly upload files or folders to a Hugging Face repo with minimal arguments.
- [`upload.py`](./upload.py) — Upload a local directory to the Hugging Face Hub with optional repo creation.

## Usage Summary

### `compute_hyperparams.py`
Compute training hyperparameters based on model size and training configuration.

Example:
```bash
python utils/compute_hyperparams.py \
  --n-layer 28 \
  --d-model 1536 \
  --l-seq 4096 \
  --params 670e6 \
  --tokens 408e9
```

Main parameters:
- `--n-layer`: number of transformer layers.
- `--d-model`: model hidden dimension.
- `--l-seq`: sequence length.
- `--params`: total model parameters (can be computed from other args).
- `--tokens`: total training tokens.

### `convert_dataset_to_hf.py`
Convert a dataset directory to HF format.

Example:
```bash
python utils/convert_dataset_to_hf.py \
  --directory_path ./data/my_dataset \
  --dataset_type jsonl \
  --output_path ./data/hf_dataset \
  --cache_dir ./data/.cache \
  --max_chunk_size_gb 5.0 \
  --new_repo_id username/my-dataset \
  --private \
  --token $HF_TOKEN
```

Main parameters:
- `--directory_path`: source directory containing JSONL or Parquet data.
- `--dataset_type`: source file type, either `jsonl` or `parquet`.
- `--output_path`: where to save converted dataset files.
- `--cache_dir`: dataset cache directory.
- `--max_chunk_size_gb`: max output chunk size in GB.
- `--new_repo_id`: optional Hugging Face repo to upload the dataset.
- `--private`: mark the created repo as private.
- `--token`: Hugging Face token (defaults to `HF_TOKEN`).

### `count_tokens.py`
Generate token counts for a tokenized dataset.

Example:
```bash
python utils/count_tokens.py --main-dir ./data/tokenized --output-file token_report.txt
```

Main parameters:
- `--main-dir`: top-level directory containing tokenized datasets.
- `--output-file`: name of the output report file.

### `download.py`
Download a repository from Hugging Face Hub.

Example:
```bash
python utils/download.py \
  --repo_name Polygl0t/some-dataset \
  --output_dir ./downloads/some-dataset \
  --cache_dir ./cache \
  --token $HF_TOKEN \
  --repo_type dataset \
  --allow_patterns "data/train*.parquet"
```

Main parameters:
- `--repo_name`: repository identifier, e.g. `username/repo`.
- `--output_dir`: target directory for the downloaded repository.
- `--cache_dir`: local HF cache directory.
- `--token`: Hugging Face authentication token.
- `--repo_type`: repository type: `dataset`, `model`, or `space`.
- `--allow_patterns`: glob patterns to filter downloaded files.

### `distributed_test.py`
GPU communication benchmark that runs all_reduce (SUM) on gradient tensors for a Llama-style LLM. Also benchmarks activation/logits-scale tensors (hidden states, logits) whose size depends on `--batch-size` and `--seq-length`, so you can measure communication cost at realistic token counts.

Example:
```bash
# Via SLURM:
srun --cpu-bind=none python utils/distributed_test.py

# Via torchrun:
torchrun --nproc-per-node=<N> utils/distributed_test.py
```

Main parameters:
- `--n-warmup`: warmup iterations (default: 5).
- `--n-iter`: timed iterations (default: 20).
- `--batch-size`: micro batch size controlling activation/logits tensor sizes (default: 128).
- `--seq-length`: sequence length controlling activation/logits tensor sizes (default: 4096).
- `--dtype`: data type for gradients — `float32`, `bfloat16`, or `float16` (default: `bfloat16`).

### `slurm/env_doctor.sh`
Diagnose GPU/CUDA/PyTorch environment issues using env-doctor.

Example:
```bash
source "./.venv/bin/activate" # <-- Activate the same environment you want to diagnose with env-doctor
env-doctor check
# Or just run the full SLURM batch script to see environment details and run env-doctor:
# sbatch utils/slurm/env_doctor.sh
```

### `extract_reasoning_traces.py`
Extract reasoning traces from a conversational dataset and save them in JSONL format.

Main parameters:
- `input_file`: path to the input dataset file (JSON or JSONL).
- `--output`: path to the output JSONL file for reasoning traces (default: `<input>_reasoning_traces.jsonl`).
- `--id-column`: column name for trace IDs (default: `"id"`).
- `--encoding`: file encoding (default: `"utf-8"`).

Example:
```bash
python utils/extract_reasoning_traces.py "./data/conversational_dataset.jsonl" \
 --output "./data/reasoning_traces.jsonl" \
 --id-column "id" \
 --encoding "utf-8"
```

### `inference_test.py`
Run inference on a model using sample inputs.

Example:
```bash
python utils/inference_test.py \
  --model_path ./models/my-model \
  --samples_file ./samples/test_samples.json \
  --output_file ./outputs/inference_results.json \
  --max_new_tokens 512 \
  --temperature 0.2 \
  --enable_thinking \
  --mode chat
```

Main parameters:
- `--model_path`: model path or Hugging Face identifier.
- `--samples_file`: JSON file containing input samples.
- `--output_file`: JSON output path for model results.
- `--max_new_tokens`: max tokens generated per sample.
- `--temperature`: sampling temperature.
- `--enable_thinking`: enable chat template thinking mode.
- `--chat_template_path`: optional custom Jinja chat template.
- `--mode`: `chat` or `completion` mode.

### `inspect_model.py`
Inspect a model configuration and count parameters.

Example:
```bash
python utils/inspect_model.py \
  --config_path ./models/my-model/config.json \
  --base_model gpt2 \
  --precision bfloat16 \
  --device cpu
```

Main parameters:
- `--config_path`: model config path or Hugging Face model directory.
- `--base_model`: optional base model for tokenizer loading.
- `--precision`: model dtype, `bfloat16` or `float32`.
- `--device`: device to load the model onto.

## `launch_trackio.py`
Launch a Trackio dashboard for logging and visualizing training metrics.

Example:
```bash
python utils/launch_trackio.py \
  --project MyProjectName \
  --db-dir ./MyProjectName
```

Main parameters:
- `--project`: name of the Trackio project.
- `--db-dir`: directory to store the Trackio database.

You can also launch trackio dashboards as a Space on Hugging Face Hub. See the `launch_trackio.py` script for more details.

### `slurm/marvin_create_workspace.sh`
Create and prepare a Marvin HPC workspace.

Key variables to customize inside the script:
- `username`, `file_system`, `work_group`, `email`
- `remainder`, `num_days`, `workspace_name`
- `workdir`: resulting workspace path

This script allocates the workspace, clones the repo, and documents environment setup steps.

### `parse_run.py`
Parse training logs and optional emissions data.

Example:
```bash
python utils/parse_run.py \
  --log ./logs/training.log \
  --emissions ./logs/emissions.csv \
  --nodes 8 \
  --output-dir ./logs/parsed \
  --plot
```

Main parameters:
- `--log`: path to training log file.
- `--emissions`: path to emissions CSV file.
- `--nodes`: number of training nodes used.
- `--output-dir`: output directory for parsed files.
- `--plot`: generate PNG plots for statistics.

### `slurm/pdf2markdown.sh`
SLURM batch helper for PDF-to-Markdown conversion using Marker.

Key variables to customize inside the script:
- `PDF_DIR`: input PDF directory.
- `OUTPUT_DIR`: directory for Markdown output.
- `NUM_DEVICES` / `NUM_WORKERS`: GPUs and workers used by the job.
- `CLEAN_CACHE`: whether to clean the HF cache after completion.

### `reset_weights.py`
Reset model weights in place or save the modified model.

Example:
```bash
python utils/reset_weights.py \
  --model ./models/my-model \
  --output_dir ./models/my-model-reset \
  --device cuda:0 \
  --dtype float16 \
  --trust_remote_code \
  --dry_run
```

Main parameters:
- `--model`: model ID or local path.
- `--output_dir`: optional save directory for the reset model.
- `--device`: load device, e.g. `cpu` or `cuda:0`.
- `--dtype`: loading dtype, `auto`, `float32`, `float16`, or `bfloat16`.
- `--trust_remote_code`: allow remote code loading.
- `--dry_run`: show reset actions without modifying the model.

### `resize_embedding_layer.py`
Check and resize model embeddings to match the tokenizer.

Example:
```bash
python utils/resize_embedding_layer.py ./models/my-model \
  --output-dir ./models/my-model-resized \
  --resize \
  --pad-to-multiple-of 64 \
  --save-missing \
  --dtype bfloat16
```

Main parameters:
- `model_path`: path to the model directory.
- `--output-dir`: output path for the resized model.
- `--resize`: enable actual resizing instead of only checking.
- `--pad-to-multiple-of`: pad vocabulary size to a hardware-friendly multiple.
- `--save-missing`: save mismatched token/embedding info.
- `--dtype`: data type used when loading the model.

### `upload_ckpts_to_hf.py`
Upload checkpoint step folders to a Hugging Face repo.

Example:
```bash
python utils/upload_ckpts_to_hf.py \
  --repo_id username/my-checkpoint-repo \
  --root_dir ./checkpoints \
  --token $HF_TOKEN
```

Main parameters:
- `--token`: Hugging Face API token.
- `--repo_id`: destination repo ID.
- `--root_dir`: root directory containing checkpoint folders.

### `upload_quick.py`
Upload a file or folder quickly to a Hugging Face repo.

Example:
```bash
python utils/upload_quick.py \
  --repo username/my-repo \
  --repo-type model \
  --token $HF_TOKEN \
  --folder ./output \
  --repo-folder artifacts \
  --main-only
```

Main parameters:
- `--repo`: destination Hugging Face repo.
- `--repo-type`: `model`, `dataset`, or `space`.
- `--token`: HF token or `HF_TOKEN` env var.
- `--main-only`: upload only to `main` branch.
- `--files` / `--folder`: local files or directory to upload.
- `--repo-folder`: path inside the repo.

### `upload.py`
Upload a directory to the Hugging Face Hub.

Example:
```bash
python utils/upload.py \
  --main_dir ./output \
  --new_repo_id username/new-repo \
  --private \
  --token $HF_TOKEN \
  --num_workers 16 \
  --repo_type dataset
```

Main parameters:
- `--main_dir`: directory to upload.
- `--new_repo_id`: new repo ID to create.
- `--private`: create a private repo.
- `--token`: HF token.
- `--num_workers`: upload concurrency.
- `--repo_type`: repository type.
