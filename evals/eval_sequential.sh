#!/bin/bash
#############################################
# LM Evaluation Harness - Simple Sequential Evaluator
#
# Loops over a list of local model folders, runs lm-evaluation-harness
# on each sequentially on a single GPU, and saves results as YAML
# in each model's folder.
#############################################

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
#SBATCH --partition=A40devel               # <-- Change to your partition
#SBATCH --job-name=eval
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --gpus=1

#############################################
# Configuration — edit these to match your setup
#############################################
set -euo pipefail
ulimit -c 0

# List of local model folders to evaluate
MODELS=(
    "/home/nklugeco/checkpoints/ckpt-0000"
    "/home/nklugeco/checkpoints/ckpt-0100"
    "/home/nklugeco/checkpoints/ckpt-0200"
    "/home/nklugeco/checkpoints/ckpt-0300"
)

# Comma-separated list of evaluation tasks
TASKS="lambada_openai,blimp,hellaswag,winogrande,arc_easy"

NUM_FEWSHOT=0                         # Number of few-shot examples (0 = zero-shot)
BATCH_SIZE="auto"                     # "auto" or an integer
MODE=hf                               # hf or vllm

# Path to the lm-evaluation-harness clone
WORKDIR="/home/nklugeco"
HARNESS_DIR="$WORKDIR/lm_evaluation_harness"
export HF_DATASETS_CACHE="$WORKDIR/.cache"                  # <-- Cache directory for datasets
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE/models"    # <-- Cache directory for models

# Temporary directory for raw JSON output (cleaned after each model)
TMP_OUTPUT_DIR="$WORKDIR/.eval_tmp_$SLURM_JOB_ID"

# Directory for per-model stdout/stderr logs
LOGS_DIR="$WORKDIR/.eval_logs_$SLURM_JOB_ID"

#############################################
# Installation instructions (uncomment on first run)
#############################################
source .modules.sh
# python3 -m venv $WORKDIR/.venv_eval
source $WORKDIR/.venv_eval/bin/activate

# git clone --branch main https://github.com/Polygl0t/lm-evaluation-harness.git
# mv $WORKDIR/lm-evaluation-harness $WORKDIR/lm_evaluation_harness
# pip3 install --upgrade pip --no-cache-dir
# pip3 install -e $WORKDIR/lm_evaluation_harness --no-cache-dir
# pip3 install "lm_eval[hf,vllm]" --no-cache-dir               # <-- Install lm-eval with HuggingFace and vLLM support
# pip3 install pyyaml --no-cache-dir                           # <-- Required for post-processing step

# --- Alternatively, install with uv ---
# pip3 install uv
# uv pip install -e $WORKDIR/lm_evaluation_harness --no-cache
# uv pip install "lm_eval[hf,vllm]" --no-cache
# uv pip install pyyaml --no-cache

#############################################
# Main loop
#############################################

mkdir -p "$TMP_OUTPUT_DIR"
mkdir -p "$LOGS_DIR"

TOTAL=${#MODELS[@]}
CURRENT=0

for MODEL_PATH in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    MODEL_NAME=$(basename "$MODEL_PATH")
    TIMESTAMP=$(date +%Y-%m-%d-%H)
    YAML_OUTPUT="$MODEL_PATH/results-$TIMESTAMP.yaml"

    echo ""
    echo "========================================="
    echo "[$CURRENT/$TOTAL] Evaluating: $MODEL_NAME"
    echo "========================================="

    if [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: Model folder not found: $MODEL_PATH"
        continue
    fi

    # Clean previous temp JSON so we only pick up this run's output
    rm -rf "$TMP_OUTPUT_DIR"
    mkdir -p "$TMP_OUTPUT_DIR"

    # Per-model log files
    OUT_LOG="$LOGS_DIR/out_${MODEL_NAME}.log"
    ERR_LOG="$LOGS_DIR/err_${MODEL_NAME}.log"

    # --- Run lm_eval ---
    echo "Running lm_eval..."
    python3 "$HARNESS_DIR/lm_eval" \
        --model "$MODE" \
        --model_args pretrained="$MODEL_PATH" \
        --tasks "$TASKS" \
        --batch_size "$BATCH_SIZE" \
        --num_fewshot "$NUM_FEWSHOT" \
        --device "cuda" \
        --output_path "$TMP_OUTPUT_DIR" \
        >"$OUT_LOG" 2>"$ERR_LOG"

    echo "Evaluation finished for $MODEL_NAME."

    # --- Post-process: JSON → YAML ---
    echo "Converting results to YAML..."

    python3 - "$TMP_OUTPUT_DIR" "$YAML_OUTPUT" "$MODEL_NAME" "$MODEL_PATH" << 'PYEOF'
import os, sys, json, yaml

logs_dir = sys.argv[1]
yaml_path = sys.argv[2]
model_name = sys.argv[3]
model_path = sys.argv[4]

# Find the JSON result file
json_files = []
for root, _, files in os.walk(logs_dir):
    for f in files:
        if f.endswith(".json"):
            json_files.append(os.path.join(root, f))

if not json_files:
    print(f"  WARNING: No JSON results found in {logs_dir} — skipping YAML generation")
    sys.exit(0)

# Use the first (and typically only) JSON file
filepath = json_files[0]
print(f"  Reading {os.path.basename(filepath)}...")

with open(filepath, "r") as f:
    data = json.load(f)

results = data.get("results", data)

# Flatten nested results
flat_results = {}
if isinstance(results, dict):
    for key, value in results.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                clean_subkey = subkey.replace(",none", "")
                flat_results[f"{key}_{clean_subkey}"] = subvalue
        else:
            flat_results[key] = value

out = {
    "model_name": model_name,
    "model_path": model_path,
    "results": flat_results,
}

with open(yaml_path, "w") as f:
    yaml.dump(out, f, default_flow_style=False)

print(f"  ✅ Saved {yaml_path}")
PYEOF

    # Clean up temp JSON
    rm -rf "$TMP_OUTPUT_DIR"

    echo "Done with $MODEL_NAME."
done

echo ""
echo "========================================="
echo "All $TOTAL model(s) evaluated."
echo "========================================="
