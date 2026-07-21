#!/bin/bash
#############################################
# LM Evaluation Harness - Parallel Evaluator
#
# Runs lm-evaluation-harness on multiple local models in parallel,
# one per GPU (up to MAX_GPUS_PER_NODE). Saves YAML results in
# each model's folder.
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
#SBATCH --gpus=4

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
MAX_GPUS_PER_NODE=4                   # Max concurrent GPU jobs
MODE=hf                               # hf or vllm

# Path to the lm-evaluation-harness clone
WORKDIR="/home/nklugeco"
HARNESS_DIR="$WORKDIR/lm_evaluation_harness"
export HF_DATASETS_CACHE="$WORKDIR/.cache"                  # <-- Cache directory for datasets
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE/models"    # <-- Cache directory for models

# Directory for per-model temp JSON and stdout/stderr logs
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
# pip3 install pyyaml --no-cache-dir                      # <-- Required for post-processing step

#############################################
# Pre-flight
#############################################

TOTAL=${#MODELS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "No models configured. Edit the MODELS array."
    exit 1
fi

# Cap concurrent jobs to the number of models (don't need more GPUs than models)
if [ "$TOTAL" -lt "$MAX_GPUS_PER_NODE" ]; then
    MAX_GPUS_PER_NODE=$TOTAL
fi

# Determine how many models we will actually launch this run
NUM_TO_LAUNCH=$TOTAL
if [ "$TOTAL" -gt "$MAX_GPUS_PER_NODE" ]; then
    NUM_TO_LAUNCH=$MAX_GPUS_PER_NODE
fi

TIMESTAMP=$(date +%Y-%m-%d-%H)
mkdir -p "$LOGS_DIR"

echo "========================================="
echo "Models configured:  $TOTAL"
echo "GPUs available:     $MAX_GPUS_PER_NODE"
echo "Launching:          $NUM_TO_LAUNCH"
echo "Tasks: $TASKS"
echo "========================================="

# Warn about models that won't be evaluated this run
if [ "$TOTAL" -gt "$MAX_GPUS_PER_NODE" ]; then
    echo "⚠️  WARNING: $TOTAL models but only $MAX_GPUS_PER_NODE GPU(s) available."
    echo "    The following models will be SKIPPED this run:"
    for i in $(seq $MAX_GPUS_PER_NODE $((TOTAL - 1))); do
        echo "     - $(basename "${MODELS[$i]}")"
    done
    echo ""
fi

#############################################
# Phase 1 — Launch all evaluation jobs in parallel
#############################################

declare -a PIDS=()
declare -a JOB_MODEL_PATHS=()
declare -a JOB_MODEL_NAMES=()
declare -a JOB_TMP_DIRS=()

for i in $(seq 0 $((NUM_TO_LAUNCH - 1))); do
    MODEL_PATH="${MODELS[$i]}"
    MODEL_NAME=$(basename "$MODEL_PATH")
    GPU_ID=$i
    TMP_DIR="$LOGS_DIR/.tmp_${i}_${MODEL_NAME}"
    OUT_LOG="$LOGS_DIR/out_${MODEL_NAME}.log"
    ERR_LOG="$LOGS_DIR/err_${MODEL_NAME}.log"

    # Validate model folder
    if [ ! -d "$MODEL_PATH" ]; then
        echo "[$((i + 1))/$TOTAL] SKIP — folder not found: $MODEL_PATH"
        continue
    fi

    # Skip if YAML already exists
    YAML_OUTPUT="$MODEL_PATH/results-$TIMESTAMP.yaml"
    if [ -f "$YAML_OUTPUT" ]; then
        echo "[$((i + 1))/$TOTAL] SKIP — already evaluated: $MODEL_NAME"
        continue
    fi

    # Prepare temp dir for this job
    rm -rf "$TMP_DIR"
    mkdir -p "$TMP_DIR"

    echo "[$((i + 1))/$TOTAL] Launching $MODEL_NAME on GPU $GPU_ID"

    # Run in background, pinned to a specific GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python3 "$HARNESS_DIR/lm_eval" \
        --model "$MODE" \
        --model_args pretrained="$MODEL_PATH" \
        --tasks "$TASKS" \
        --batch_size "$BATCH_SIZE" \
        --num_fewshot "$NUM_FEWSHOT" \
        --device "cuda" \
        --output_path "$TMP_DIR" \
        >"$OUT_LOG" 2>"$ERR_LOG" &

    PIDS+=($!)
    JOB_MODEL_PATHS+=("$MODEL_PATH")
    JOB_MODEL_NAMES+=("$MODEL_NAME")
    JOB_TMP_DIRS+=("$TMP_DIR")

    # Small stagger to avoid thundering-herd on launch
    sleep 1
done

NUM_JOBS=${#PIDS[@]}
echo ""
echo "Launched $NUM_JOBS job(s). Waiting for completion..."

#############################################
# Phase 2 — Wait for all jobs
#############################################

FAILED=0
for i in $(seq 0 $((NUM_JOBS - 1))); do
    wait "${PIDS[$i]}"
    EXIT_CODE=$?
    MODEL_NAME="${JOB_MODEL_NAMES[$i]}"

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "✅ $MODEL_NAME finished successfully"
    else
        echo "❌ $MODEL_NAME failed (exit code: $EXIT_CODE) — see $LOGS_DIR/err_${MODEL_NAME}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "All jobs finished ($FAILED failure(s))."

#############################################
# Phase 3 — Post-process JSON → YAML
#############################################

echo ""
echo "========================================="
echo "Post-processing results..."
echo "========================================="

for i in $(seq 0 $((NUM_JOBS - 1))); do
    MODEL_PATH="${JOB_MODEL_PATHS[$i]}"
    MODEL_NAME="${JOB_MODEL_NAMES[$i]}"
    TMP_DIR="${JOB_TMP_DIRS[$i]}"
    YAML_OUTPUT="$MODEL_PATH/results-$TIMESTAMP.yaml"

    python3 - "$TMP_DIR" "$YAML_OUTPUT" "$MODEL_NAME" "$MODEL_PATH" << 'PYEOF'
import os, sys, json, yaml

logs_dir = sys.argv[1]
yaml_path = sys.argv[2]
model_name = sys.argv[3]
model_path = sys.argv[4]

json_files = []
for root, _, files in os.walk(logs_dir):
    for f in files:
        if f.endswith(".json"):
            json_files.append(os.path.join(root, f))

if not json_files:
    print(f"  ⚠ {model_name}: No JSON results — skipping YAML")
    sys.exit(0)

filepath = json_files[0]

with open(filepath, "r") as f:
    data = json.load(f)

results = data.get("results", data)

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

print(f"  ✅ {model_name} → {os.path.basename(yaml_path)}")
PYEOF

    # Clean up this job's temp dir
    rm -rf "$TMP_DIR"
done

echo ""
echo "========================================="
echo "Done. Evaluated $NUM_JOBS model(s)."
echo "========================================="
