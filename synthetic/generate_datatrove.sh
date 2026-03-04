#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=mlgpu_short            # <-- Change to your partition
#SBATCH --job-name=synthetic-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:a40:8
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="scratch"                      # <-- Change to your filesystem
workspace_name="polyglot_datasets"         # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/synth/synth_logs"
cd "$workdir"
ulimit -c 0

out="$workdir/synth/synth_logs/out.$SLURM_JOB_ID"
err="$workdir/synth/synth_logs/err.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
source $workdir/.modules_amd.sh                       # <-- Load necessary modules
# python3 -m venv $workdir/.venv_amd                  # <-- Create virtual environment
source $workdir/.venv_amd/bin/activate                # <-- Activate virtual environment

# pip3 install --upgrade pip --no-cache-dir
# pip3 install torch==2.8.0 --no-cache-dir
# pip3 install torchaudio==2.8.0 --no-cache-dir
# pip3 install torchvision==0.23.0 --no-cache-dir
# pip3 install transformers --no-cache-dir
# pip3 install datatrove[inference] --no-cache-dir
# pip3 install pyarrow --no-cache-dir
# pip3 install xxhash --no-cache-dir
# pip3 install vllm --no-cache-dir

export HF_TOKEN="your_hugging_face_token"               # <-- Change to your Hugging Face token
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache"
export CLEAN_CACHE="0"                                  # Set to "1" to clean cache after job completion
export DP=8                                             # <-- Data parallelism across GPUs
export TP=1                                             # <-- Tensor parallelism (for bigger models)
export MODEL_NAME_OR_PATH="Qwen/Qwen3-4B-Instruct-2507" # <-- Change to your model name or path
export DATASET_PATH="$workdir/portuguese/personas"      # <-- Change to your dataset path (directory with JSONL or Parquet files)
export TEXT_COLUMN="text"                   # <-- Change to your dataset text column name
export OUTPUT_DIR="$workdir/synth/personas_bio"    # <-- Change to your desired output directory
export SYSTEM_PROMPT="Você é um gerador de livros auto biográficos. Seu objetivo é criar livros biográficos detalhados e precisos."          # <-- Change to your system prompt (or leave empty)
export PROMPT_TEMPLATE="Cria um livro biográfico sobre está pessoa: [[DOCUMENT]]"   # <-- Optional: template with [[DOCUMENT]] placeholder (e.g. "Summarize: [[DOCUMENT]]")
export MAX_TOKENS=4096                     # <-- Max output tokens per generation
export MODEL_MAX_CONTEXT=8192              # <-- Maximum context length for the model
export TEMPERATURE=0.7
export TOP_K=20
export TOP_P=0.8
export ROLLOUTS_PER_DOCUMENT=1
export EXAMPLES_PER_CHUNK=500               # <-- Documents per checkpoint chunk
mkdir -p "$OUTPUT_DIR"

if [[ -n "$HF_TOKEN" ]]; then
    # Login to Hugging Face (if needed)
    hf auth login --token "$HF_TOKEN"
fi

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES node(s)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $DP GPU(s) via data parallelism" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $TP GPU(s) via tensor parallelism" >> "$out"
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

# Use eval to properly handle the optional arguments with spaces.
eval python3 $workdir/synth/generate_datatrove.py \
    --input-path "$DATASET_PATH" \
    --prompt-column "$TEXT_COLUMN" \
    --output-path "$OUTPUT_DIR" \
    --model-name-or-path "$MODEL_NAME_OR_PATH" \
    --model-max-context "$MODEL_MAX_CONTEXT" \
    --dp "$DP" \
    --tp "$TP" \
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