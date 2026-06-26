#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn about SLURM sbatch options at:
# - https://slurm.schedmd.com/sbatch.html
#
# Learn about job submissions (Marvin|Bender) at:
# - https://wiki.hpc.uni-bonn.de/en/running_jobs
#
# Learn about Marvin|Bender dual software stacks at:
# - https://wiki.hpc.uni-bonn.de/en/dualstacks
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

# Set this to your workspace root (where you have the .venv and .modules.sh files).
workdir="/lustre/mlnvme/data/polyglot"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/synth/.logs/out.$SLURM_JOB_ID"
err="$workdir/synth/.logs/err.$SLURM_JOB_ID"

#############################################
# Modules & Libraries Setup
#############################################

source $workdir/.modules.sh > "$out" 2>&1
# python3 -m venv $workdir/.venv_synth
source $workdir/.venv_synth/bin/activate

# ===== Install for vLLM + Datatrove Pipeline =====
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

#############################################
# Environment Setup
#############################################

export HF_TOKEN=""                                        # <-- Change to your Hugging Face token
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK               # <-- Set OpenMP threads to match SLURM CPU allocation
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"  # <-- Path to Hugging Face datasets cache
export PYTHONPYCACHEPREFIX="$HF_DATASETS_CACHE/.pycache"  # <-- Path to Python bytecode cache
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"         # <-- Path to Hugging Face Hub cache (model weights, tokenizers, etc.)
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache" # <-- Path to Triton cache (for vLLM)
export CLEAN_CACHE="0"                                   # Set to "1" to clean cache after job completion
export DP=8                                              # <-- Data parallelism across GPUs
export TP=1                                              # <-- Tensor parallelism (for bigger models)
export PP=1                                              # <-- Pipeline parallelism
export MODEL_NAME_OR_PATH="Qwen/Qwen3-14B"               # <-- Change to your model name or path
export DATASET_PATH="$workdir/data"                      # <-- Change to your dataset path (directory with JSONL or Parquet files)
export TEXT_COLUMN="text"                                # <-- Change to your dataset text column name
export OUTPUT_DIR="$workdir/output"                      # <-- Change to your desired output directory
export SYSTEM_PROMPT_FILE="$workdir/SYSTEM.md"           # <-- Path to system prompt file
export PROMPT_TEMPLATE_FILE="$workdir/PROMPT.md"         # <-- Path to prompt template file (must contain [[DOCUMENT]] placeholder)
export MAX_CONCURRENT_GENERATIONS=100                    # <-- Max concurrent generations across all GPUs (tune based on model size and GPU memory)
export MAX_TOKENS=10000                                  # <-- Max output tokens per generation
export MODEL_MAX_CONTEXT=32768                           # <-- Maximum context length for the model
export TEMPERATURE=0.7                                   # <-- Sampling parameter
export TOP_K=20                                          # <-- Sampling parameter
export TOP_P=0.8                                         # <-- Sampling parameter
export ROLLOUTS_PER_DOCUMENT=1                           # <-- Number of generations to produce per input document
export EXAMPLES_PER_CHUNK=2000                           # <-- Documents per checkpoint chunk
#export VLLM_LOGGING_LEVEL="DEBUG"                       # <-- Useful for diagnosing vLLM distributed startup
#export VLLM_ENABLE_LOG_REQUESTS="1"                     # <-- Set to "1" for per-request traces (very verbose)
#export NCCL_DEBUG="INFO"                                # <-- Useful for diagnosing multi-GPU communication

mkdir -p "$OUTPUT_DIR"

if [[ -n "$HF_TOKEN" ]]; then
    # Login to Hugging Face (if needed)
    hf auth login --token "$HF_TOKEN"
fi

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES node(s)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $DP GPU(s) via data parallelism" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $TP GPU(s) via tensor parallelism" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $PP GPU(s) via pipeline parallelism" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# [${SLURM_JOB_ID}] GLIBC version: $(ldd --version | head -n1)" >> "$out"
echo "# [${SLURM_JOB_ID}] Working directory: $workdir" >> "$out"
echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version)" >> "$out"
echo "# [${SLURM_JOB_ID}] Model: $MODEL_NAME_OR_PATH" >> "$out"
echo "# [${SLURM_JOB_ID}] Dataset path: $DATASET_PATH" >> "$out"

#############################################
# Main Job Execution
#############################################
# Build optional arguments (pass file paths instead of inline content
# to avoid shell interpretation of backticks, $, (, ), etc.)
OPTIONAL_ARGS=""
if [[ -n "$SYSTEM_PROMPT_FILE" && -f "$SYSTEM_PROMPT_FILE" ]]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --system-prompt-file \"$SYSTEM_PROMPT_FILE\""
fi
if [[ -n "$PROMPT_TEMPLATE_FILE" && -f "$PROMPT_TEMPLATE_FILE" ]]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --prompt-template-file \"$PROMPT_TEMPLATE_FILE\""
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
    --max-concurrent-generations "$MAX_CONCURRENT_GENERATIONS" \
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
