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
#SBATCH --account=ag_bit_flek               # <-- Change to your SLURM account
#SBATCH --partition=mlgpu_short             # <-- Change to your partition
#SBATCH --job-name=benchmark-lm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --oversubscribe

#############################################
# Working Directory Setup
#############################################

# Set this to your workspace root (where you have the .venv-bench and .modules.sh files).
workdir="/lustre/mlnvme/data/polyglot"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-benchmark-lm.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-benchmark-lm.$SLURM_JOB_ID"

#############################################
# Modules & Libraries Setup
#############################################

source "$workdir/.modules.sh" > "$out" 2>&1
# python3 -m venv "$workdir/.venv_bench"
source "$workdir/.venv_bench/bin/activate"


# ===== Installation (only required once) =====
# pip3 install --upgrade pip
# pip3 install vllm==0.28.0 transformers --no-cache-dir

# ===== Alternatively, install with uv =====
# pip3 install --upgrade pip --no-cache-dir
# pip3 install uv
# uv pip install vllm==0.28.0 transformers --no-cache

#############################################
# Environment Setup
#############################################

# ─── Cache Directories ────────────────────────────────────────── #
export CLEAN_CACHE="1"                                             # <-- Set to "1" to clean cache after job completion
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache"

# ─── Runtime ──────────────────────────────────────────────────── #
export CUDA_VISIBLE_DEVICES=0                                      # <-- GPU(s) visible to vLLM; keep "0" with --gres=gpu:...:1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export HF_TOKEN="hf_..."                                         # <-- For private/gated models, set your Hugging Face token.
# export VLLM_WORKER_MULTIPROC_METHOD=spawn                        # <-- Required for tensor parallelism > 1 on some clusters.

# ─── Model ────────────────────────────────────────────────────── #
export MODEL_ID="$workdir/qwen3-1.7b-base"                         # <-- JSONL file or directory of prompts (--prompts)
export TARGET_LANGUAGE="portuguese"                                # <-- Target language for drift detection (--target-language)
export BATCH_SIZES="1,8,16,32,64"                                  # <-- Comma-separated batch sizes; 1 = sequential, >1 = batches (--batch-sizes)
export TENSOR_PARALLEL_SIZE=1                                      # <-- Number of GPUs for tensor parallelism (--tensor-parallel-size)
export DTYPE="auto"                                                # <-- Model weight dtype: auto, float16, bfloat16, ... (--dtype)
export GPU_MEMORY_UTILIZATION=0.9                                  # <-- Fraction of GPU memory for weights + KV cache, 0.0-1.0 (--gpu-memory-utilization)
export MAX_TOKENS=2048                                             # <-- Max new output tokens per generation; empty = model config or 256 (--max-tokens)
export MAX_MODEL_LEN=""                                            # <-- Max total sequence length (prompt + output); empty = model config (--max-model-len)
export TRUST_REMOTE_CODE="0"                                       # <-- Set to "1" to allow custom model/tokenizer code (--trust-remote-code)
export SEED=""                                                     # <-- Random seed for sampling; empty = random (--seed)

# ─── Benchmark / Generation ───────────────────────────────────── #
export PROMPTS_PATH="$workdir/llm-foundry/tools/assets/benchmark_prompts.jsonl" # <-- JSONL file or directory of prompts (--prompts)
export MODE="completion"                                           # <-- completion (raw generation) or chat (chat template) (--mode)
export LOGPROBS="1"                                                # <-- Set to "1" to enable per-token logprobs for drift likelihood (--logprobs)
export N_WINDOWS=5                                                 # <-- Number of drift windows per response (--n-windows)

# ─── Sampling overrides (empty = model's generation_config) ───── #
export TEMPERATURE=""                                              # <-- Sampling temperature (--temperature)
export TOP_P=""                                                    # <-- Nucleus sampling top-p (--top-p)
export TOP_K=""                                                    # <-- Top-k sampling cutoff (--top-k)
export REPETITION_PENALTY=""                                       # <-- Repetition penalty (--repetition-penalty)
export PRESENCE_PENALTY=""                                         # <-- Presence penalty (--presence-penalty)
export FREQUENCY_PENALTY=""                                        # <-- Frequency penalty (--frequency-penalty)

# ─── Output ───────────────────────────────────────────────────── #
export OUTPUT_DIR="$workdir/results"                               # <-- Directory for results and summary (--output-dir)
export APPEND_CSV="$workdir/results/summary.csv"                   # <-- Optional CSV for cross-model comparison (--append-csv)

mkdir -p "$OUTPUT_DIR"

echo "# [${SLURM_JOB_ID}] Job started on $SLURM_JOB_NODELIST at: $(date)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# [${SLURM_JOB_ID}] Working directory: $workdir" >> "$out"
echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version)" >> "$out"

#############################################
# Main Job Execution
#############################################

# Build optional flags (only passed when explicitly set)
MAX_TOKENS_FLAG=""
if [[ -n "$MAX_TOKENS" ]]; then
    MAX_TOKENS_FLAG="--max-tokens $MAX_TOKENS"
fi

MAX_MODEL_LEN_FLAG=""
if [[ -n "$MAX_MODEL_LEN" ]]; then
    MAX_MODEL_LEN_FLAG="--max-model-len $MAX_MODEL_LEN"
fi

SEED_FLAG=""
if [[ -n "$SEED" ]]; then
    SEED_FLAG="--seed $SEED"
fi

APPEND_CSV_FLAG=""
if [[ -n "$APPEND_CSV" ]]; then
    APPEND_CSV_FLAG="--append-csv $APPEND_CSV"
fi

TEMPERATURE_FLAG=""
if [[ -n "$TEMPERATURE" ]]; then
    TEMPERATURE_FLAG="--temperature $TEMPERATURE"
fi

TOP_P_FLAG=""
if [[ -n "$TOP_P" ]]; then
    TOP_P_FLAG="--top-p $TOP_P"
fi

TOP_K_FLAG=""
if [[ -n "$TOP_K" ]]; then
    TOP_K_FLAG="--top-k $TOP_K"
fi

REPETITION_PENALTY_FLAG=""
if [[ -n "$REPETITION_PENALTY" ]]; then
    REPETITION_PENALTY_FLAG="--repetition-penalty $REPETITION_PENALTY"
fi

PRESENCE_PENALTY_FLAG=""
if [[ -n "$PRESENCE_PENALTY" ]]; then
    PRESENCE_PENALTY_FLAG="--presence-penalty $PRESENCE_PENALTY"
fi

FREQUENCY_PENALTY_FLAG=""
if [[ -n "$FREQUENCY_PENALTY" ]]; then
    FREQUENCY_PENALTY_FLAG="--frequency-penalty $FREQUENCY_PENALTY"
fi

LOGPROBS_FLAG=""
if [ "$LOGPROBS" = "1" ]; then
    LOGPROBS_FLAG="--logprobs"
fi

TRUST_REMOTE_CODE_FLAG=""
if [ "$TRUST_REMOTE_CODE" = "1" ]; then
    TRUST_REMOTE_CODE_FLAG="--trust-remote-code"
fi

python3 "$workdir/llm-foundry/tools/benchmark_lm.py" \
    --model "$MODEL_ID" \
    --prompts "$PROMPTS_PATH" \
    --target-language "$TARGET_LANGUAGE" \
    --batch-sizes "$BATCH_SIZES" \
    --mode "$MODE" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --n-windows "$N_WINDOWS" \
    --output-dir "$OUTPUT_DIR" \
    $MAX_TOKENS_FLAG \
    $MAX_MODEL_LEN_FLAG \
    $SEED_FLAG \
    $APPEND_CSV_FLAG \
    $TEMPERATURE_FLAG \
    $TOP_P_FLAG \
    $TOP_K_FLAG \
    $REPETITION_PENALTY_FLAG \
    $PRESENCE_PENALTY_FLAG \
    $FREQUENCY_PENALTY_FLAG \
    $LOGPROBS_FLAG \
    $TRUST_REMOTE_CODE_FLAG \
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
