#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_bit_flek              # <-- Change to your SLURM account
#SBATCH --partition=mlgpu_short            # <-- Change to your partition
#SBATCH --job-name=synthetic-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a40:8
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="scratch"                      # <-- Change to your filesystem
workspace_name="poly_datasets"             # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/synth/.logs"
cd "$workdir"
ulimit -c 0

out="$workdir/synth/.logs/out.$SLURM_JOB_ID"
err="$workdir/synth/.logs/err.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
source $workdir/.modules_amd.sh                      # <-- Load necessary modules
# python3 -m venv "$workdir/.venv_synth"             # <-- Create a clean virtual environment
source "$workdir/.venv_synth/bin/activate"           # <-- Activate virtual environment

# pip3 install --upgrade pip --no-cache-dir
# pip3 install \
#    "datatrove[io]" \
#    "aiofiles" \
#    "httpx" \
#    "aiosqlite" \
#    "vllm==0.19.0" \
#    "transformers>=4.56.0,<5" \
#    "huggingface-hub>=0.34.0,<1.0" \
#    "bitsandbytes" \
#    "numpy>=2.0.0,<2.3.0" \
#    "typer" \
#    "pyyaml" \
#    "pandas" \
#    --no-cache-dir
# pip3 check

export HF_TOKEN="<your-token-here>"                      # <-- Change to your Hugging Face token
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export PYTHONPYCACHEPREFIX="$HF_DATASETS_CACHE/.pycache"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache"
export CLEAN_CACHE="0"                                   # Set to "1" to clean cache after job completion
export DP=8                                              # <-- Data parallelism across GPUs
export TP=1                                              # <-- Tensor parallelism (for bigger models)
export PP=1                                              # <-- Pipeline parallelism
export MODEL_NAME_OR_PATH="Qwen/Qwen3-14B"               # <-- Change to your model name or path
export DATASET_PATH="$workdir/synth/data"                # <-- Change to your dataset path (directory with JSONL or Parquet files)
export TEXT_COLUMN="prompt"                              # <-- Change to your dataset text column name
export OUTPUT_DIR="$workdir/synth/output"                # <-- Change to your desired output directory
export SYSTEM_PROMPT=$(cat "$workdir/synth/SYSTEM.md")   # <-- Read system prompt from file
export PROMPT_TEMPLATE=$(cat "$workdir/synth/PROMPT.md") # <-- Read prompt template from file (must contain [[DOCUMENT]] placeholder)
export MAX_TOKENS=10000                     # <-- Max output tokens per generation
export MODEL_MAX_CONTEXT=16384              # <-- Maximum context length for the model
export TEMPERATURE=0.7
export TOP_K=20
export TOP_P=0.8
export ROLLOUTS_PER_DOCUMENT=1
export EXAMPLES_PER_CHUNK=17500              # <-- Documents per checkpoint chunk
#export VLLM_LOGGING_LEVEL="DEBUG"           # Useful for diagnosing vLLM distributed startup
#export VLLM_ENABLE_LOG_REQUESTS="1"         # Set to "1" for per-request traces (very verbose)
#export NCCL_DEBUG="INFO"                    # Useful for diagnosing multi-GPU communication

mkdir -p "$OUTPUT_DIR"

if [[ -n "$HF_TOKEN" ]]; then
    # Login to Hugging Face (if needed)
    hf auth login --token "$HF_TOKEN"
fi

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES node(s)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $DP GPU(s) via data parallelism" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $TP GPU(s) via tensor parallelism" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $PP GPU(s) via pipeline parallelism" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution
#############################################
# Build optional arguments
OPTIONAL_ARGS=""
if [[ -n "$SYSTEM_PROMPT" ]]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --system-prompt \"$SYSTEM_PROMPT\""
fi
if [[ -n "$PROMPT_TEMPLATE" ]]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --prompt-template \"$PROMPT_TEMPLATE\""
fi

eval python3 $workdir/synth/generate_datatrove.py \
    --input-path "$DATASET_PATH" \
    --prompt-column "$TEXT_COLUMN" \
    --output-path "$OUTPUT_DIR" \
    --model-name-or-path "$MODEL_NAME_OR_PATH" \
    --model-max-context "$MODEL_MAX_CONTEXT" \
    --dp "$DP" \
    --tp "$TP" \
    --pp "$PP" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-k "$TOP_K" \
    --top-p "$TOP_P" \
    --rollouts-per-document "$ROLLOUTS_PER_DOCUMENT" \
    --examples-per-chunk "$EXAMPLES_PER_CHUNK" \
    $OPTIONAL_ARGS \
    1>>"$out" 2>>"$err"

#############################################
# End of Script
#############################################
# Clean HF_DATASETS_CACHE folder if requested
if [ "$CLEAN_CACHE" = "1" ]; then
    echo "# [${SLURM_JOB_ID}] Cleaning HF_DATASETS_CACHE" >> "$out"
    if [ -d "$HF_DATASETS_CACHE" ]; then
        find "$HF_DATASETS_CACHE" -mindepth 1 -delete 2>/dev/null || true
    fi
else
    echo "# [${SLURM_JOB_ID}] Skipping cache cleanup (CLEAN_CACHE=$CLEAN_CACHE)" >> "$out"
fi

echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out"