#!/bin/bash
#############################################
# LM Evaluation Harness - Multi-GPU Evaluator
#
# Evaluates models using ALL available GPUs together
# (data-parallel, tensor-parallel, or model-sharding).
# Supports both HuggingFace (accelerate) and VLLM backends.
# Models are processed sequentially — each gets the full node.
#############################################

#############################################
# SLURM Job Configuration
#############################################
#SBATCH --partition=A40devel               # <-- Change to your partition
#SBATCH --job-name=eval-mgpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --gpus=4                           # All GPUs used together per model

#############################################
# Configuration — edit these to match your setup
#############################################
set -euo pipefail
ulimit -c 0

# List of local model folders to evaluate (processed one at a time)
MODELS=(
    "/home/nklugeco/checkpoints/ckpt-0000"
    "/home/nklugeco/checkpoints/ckpt-0100"
    "/home/nklugeco/checkpoints/ckpt-0200"
    "/home/nklugeco/checkpoints/ckpt-0300"
)

# Comma-separated list of evaluation tasks
TASKS="bluex_cloze,oab_exams_cloze,enem_cloze"

NUM_FEWSHOT=3                         # Number of few-shot examples (0 = zero-shot)
BATCH_SIZE="auto"                     # "auto" (recommended for vLLM) or an integer

# --- Backend selection ---
#   hf   = HuggingFace transformers (via accelerate for multi-GPU)
#   vllm = vLLM (tensor-parallel / data-parallel)
MODE="vllm"

# --- Multi-GPU strategy (only relevant when >1 GPU available) ---
#   HF strategies:
#     data_parallel  -> accelerate launch --multi_gpu (each GPU holds a full model copy)
#     model_shard    -> --model_args parallelize=True (model split across GPUs via device_map)
#     combined       -> both data_parallel + model_shard (for very large models)
#     tp_native      -> torchrun + tp_plan=auto (PyTorch native tensor parallelism, PT >= 2.4)
#   VLLM strategies:
#     tensor_parallel -> --model_args tensor_parallel_size=N (default, recommended)
#     data_parallel   -> --model_args data_parallel_size=N (requires pip install ray)
#     combined        -> both tensor + data parallel
HF_STRATEGY="model_shard"
VLLM_STRATEGY="tensor_parallel"

# Number of GPUs to use per model (default: all GPUs requested via SLURM)
# Set to a lower number to leave GPUs free, or "auto" to use $SLURM_GPUS_ON_NODE
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}}

# --- VLLM-specific tuning ---
VLLM_GPU_MEMORY_UTILIZATION=0.9       # Fraction of GPU memory for KV cache (0.0-1.0)
VLLM_MAX_MODEL_LEN="4096"             # Max sequence length (e.g. 4096). Empty = model default.
VLLM_DTYPE="auto"                     # auto, float16, bfloat16

# --- HF-specific tuning ---
HF_DEVICE_MAP_OPTION="auto"           # auto, sequential, balanced, etc.
HF_MAX_MEMORY_PER_GPU=""              # e.g. "40GiB" — max memory per GPU for model sharding

#############################################
# Paths & environment
#############################################
WORKDIR="/home/nklugeco"
HARNESS_DIR="$WORKDIR/lm_evaluation_harness"
export HF_DATASETS_CACHE="$WORKDIR/.cache"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE/models"

LOGS_DIR="$WORKDIR/.eval_logs_$SLURM_JOB_ID"

#############################################
# Installation instructions (uncomment on first run)
#############################################
source .modules.sh
# python3 -m venv $WORKDIR/.venv_eval
source $WORKDIR/.venv_eval/bin/activate

# --- Step 1: Install the base harness ---
# git clone --branch polyglot_harness_portuguese https://github.com/Polygl0t/lm-evaluation-harness.git
# mv $WORKDIR/lm-evaluation-harness $WORKDIR/lm_evaluation_harness
# pip3 install --upgrade pip --no-cache-dir
# pip3 install -e $WORKDIR/lm_evaluation_harness --no-cache-dir

# --- Step 2: Install the HuggingFace backend ---
# pip3 install "lm_eval[hf]" --no-cache-dir

# --- Step 3: Install vLLM SEPARATELY (only if using MODE=vllm) ---
# pip3 install "lm_eval[vllm]" --no-cache-dir
# If using VLLM data-parallel mode, also install ray:
# pip3 install ray --no-cache-dir

# --- Step 4: Misc dependencies ---
# pip3 install pyyaml --no-cache-dir

#############################################
# Pre-flight
#############################################

TOTAL=${#MODELS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "No models configured. Edit the MODELS array."
    exit 1
fi

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: NUM_GPUS must be >= 1 (detected: $NUM_GPUS)"
    exit 1
fi

TIMESTAMP=$(date +%Y-%m-%d-%H)
mkdir -p "$LOGS_DIR"

# Validate mode
case "$MODE" in
    hf|vllm) ;;
    *)
        echo "ERROR: MODE must be 'hf' or 'vllm' (got: '$MODE')"
        exit 1
        ;;
esac

# Determine strategy based on mode
if [ "$MODE" = "hf" ]; then
    STRATEGY="$HF_STRATEGY"
    case "$STRATEGY" in
        data_parallel|model_shard|combined|tp_native) ;;
        *)
            echo "ERROR: HF_STRATEGY must be one of: data_parallel, model_shard, combined, tp_native"
            exit 1
            ;;
    esac
else
    STRATEGY="$VLLM_STRATEGY"
    case "$STRATEGY" in
        tensor_parallel|data_parallel|combined) ;;
        *)
            echo "ERROR: VLLM_STRATEGY must be one of: tensor_parallel, data_parallel, combined"
            exit 1
            ;;
    esac
fi

echo "========================================="
echo " Multi-GPU LM Evaluation Harness"
echo "========================================="
echo "Mode:               $MODE"
echo "Strategy:           $STRATEGY"
echo "GPUs per model:     $NUM_GPUS"
echo "Models to evaluate: $TOTAL"
echo "Tasks:              $TASKS"
echo "Batch size:         $BATCH_SIZE"
echo "Few-shot:           $NUM_FEWSHOT"
echo "========================================="
echo ""

# Print models
for i in $(seq 0 $((TOTAL - 1))); do
    echo "  [$((i + 1))] $(basename "${MODELS[$i]}")"
done
echo ""

#############################################
# Build model_args string based on mode/strategy
#############################################
build_model_args() {
    local model_path="$1"
    local args="pretrained=$model_path"

    if [ "$MODE" = "hf" ]; then
        case "$STRATEGY" in
            model_shard|combined)
                args="$args,parallelize=True"
                [ -n "$HF_DEVICE_MAP_OPTION" ] && args="$args,device_map_option=$HF_DEVICE_MAP_OPTION"
                [ -n "$HF_MAX_MEMORY_PER_GPU" ] && args="$args,max_memory_per_gpu=$HF_MAX_MEMORY_PER_GPU"
                ;;
        esac
    else  # vllm
        args="$args,dtype=$VLLM_DTYPE,gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION"
        [ -n "$VLLM_MAX_MODEL_LEN" ] && args="$args,max_model_len=$VLLM_MAX_MODEL_LEN"

        case "$STRATEGY" in
            tensor_parallel)
                args="$args,tensor_parallel_size=$NUM_GPUS"
                ;;
            data_parallel)
                args="$args,data_parallel_size=$NUM_GPUS"
                ;;
            combined)
                # Split GPUs: e.g. 8 GPUs, tp=2, dp=4
                local tp_size=2
                if [ "$NUM_GPUS" -le 2 ]; then
                    tp_size=1
                elif [ "$NUM_GPUS" -le 4 ]; then
                    tp_size=2
                else
                    tp_size=$((NUM_GPUS / 4 > 0 ? NUM_GPUS / 4 : 2))
                fi
                local dp_size=$((NUM_GPUS / tp_size))
                args="$args,tensor_parallel_size=$tp_size,data_parallel_size=$dp_size"
                echo "  -> VLLM combined: tp_size=$tp_size, dp_size=$dp_size" >&2
                ;;
        esac
    fi

    echo "$args"
}

#############################################
# Build the launch command for a model
#############################################
build_cmd() {
    local model_path="$1"
    local tmp_dir="$2"
    local model_args
    model_args=$(build_model_args "$model_path")

    if [ "$MODE" = "hf" ]; then
        case "$STRATEGY" in
            data_parallel)
                echo "accelerate launch --multi_gpu --num_processes $NUM_GPUS \
    -m lm_eval \
    --model hf \
    --model_args \"$model_args\" \
    --tasks \"$TASKS\" \
    --batch_size \"$BATCH_SIZE\" \
    --num_fewshot \"$NUM_FEWSHOT\" \
    --output_path \"$tmp_dir\""
                ;;
            model_shard)
                echo "python3 -m lm_eval \
    --model hf \
    --model_args \"$model_args\" \
    --tasks \"$TASKS\" \
    --batch_size \"$BATCH_SIZE\" \
    --num_fewshot \"$NUM_FEWSHOT\" \
    --device cuda \
    --output_path \"$tmp_dir\""
                ;;
            combined)
                echo "accelerate launch --multi_gpu --num_processes $NUM_GPUS \
    -m lm_eval \
    --model hf \
    --model_args \"$model_args\" \
    --tasks \"$TASKS\" \
    --batch_size \"$BATCH_SIZE\" \
    --num_fewshot \"$NUM_FEWSHOT\" \
    --output_path \"$tmp_dir\""
                ;;
            tp_native)
                echo "torchrun --nproc-per-node=$NUM_GPUS -m lm_eval \
    --model hf \
    --model_args \"$model_args,tp_plan=auto\" \
    --tasks \"$TASKS\" \
    --batch_size \"$BATCH_SIZE\" \
    --num_fewshot \"$NUM_FEWSHOT\" \
    --output_path \"$tmp_dir\""
                ;;
        esac
    else  # vllm
        echo "python3 -m lm_eval \
    --model vllm \
    --model_args \"$model_args\" \
    --tasks \"$TASKS\" \
    --batch_size \"$BATCH_SIZE\" \
    --num_fewshot \"$NUM_FEWSHOT\" \
    --output_path \"$tmp_dir\""
    fi
}

#############################################
# Main loop — evaluate models sequentially
#############################################

EVALUATED=0
SKIPPED=0
FAILED=0

for i in $(seq 0 $((TOTAL - 1))); do
    MODEL_PATH="${MODELS[$i]}"
    MODEL_NAME=$(basename "$MODEL_PATH")
    TMP_DIR="$LOGS_DIR/.tmp_${MODEL_NAME}"
    OUT_LOG="$LOGS_DIR/out_${MODEL_NAME}.log"
    ERR_LOG="$LOGS_DIR/err_${MODEL_NAME}.log"

    echo "========================================="
    echo "[$((i + 1))/$TOTAL] $MODEL_NAME"
    echo "========================================="

    # Validate model folder
    if [ ! -d "$MODEL_PATH" ]; then
        echo "  ⚠ SKIP — folder not found: $MODEL_PATH"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Skip if YAML already exists
    YAML_OUTPUT="$MODEL_PATH/results-$TIMESTAMP.yaml"
    if [ -f "$YAML_OUTPUT" ]; then
        echo "  ⚠ SKIP — already evaluated (results-$TIMESTAMP.yaml exists)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Prepare temp dir
    rm -rf "$TMP_DIR"
    mkdir -p "$TMP_DIR"

    # Build and print the command
    CMD=$(build_cmd "$MODEL_PATH" "$TMP_DIR")
    echo "  Command:"
    echo "$CMD" | sed 's/^/    /'
    echo ""

    # Run the evaluation
    echo "  Launching evaluation on $NUM_GPUS GPU(s)..."
    set +e
    eval "$CMD" >"$OUT_LOG" 2>"$ERR_LOG"
    EXIT_CODE=$?
    set -e

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "  ✅ Evaluation completed"

        # Post-process JSON -> YAML
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

print(f"  ✅ YAML saved -> {os.path.basename(yaml_path)}")
PYEOF

        EVALUATED=$((EVALUATED + 1))
    else
        echo "  ❌ Evaluation FAILED (exit code: $EXIT_CODE)"
        echo "     stdout: $OUT_LOG"
        echo "     stderr: $ERR_LOG"
        FAILED=$((FAILED + 1))
    fi

    # Clean up temp dir
    rm -rf "$TMP_DIR"
    echo ""
done

#############################################
# Summary
#############################################
echo "========================================="
echo " Evaluation Summary"
echo "========================================="
echo "Total models:    $TOTAL"
echo "Evaluated:       $EVALUATED"
echo "Skipped:         $SKIPPED"
echo "Failed:          $FAILED"
echo "Logs:            $LOGS_DIR"
echo "========================================="
