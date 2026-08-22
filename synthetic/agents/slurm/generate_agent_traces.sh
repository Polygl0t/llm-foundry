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
#SBATCH --job-name=agent-traces
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a40:8
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################

# Set this to your traces project root.
workdir="/lustre/scratch/data/polyglot"
mkdir -p "$workdir/logs"
cd "$workdir"
ulimit -c 0

out="$workdir/logs/out-agent-traces.$SLURM_JOB_ID"
err="$workdir/logs/err-agent-traces.$SLURM_JOB_ID"

#############################################
# Modules & Libraries Setup
#############################################

source $workdir/.modules.sh > "$out" 2>&1
source $workdir/.venv_agents/bin/activate

# ===== Installation =====
# See synthetic/agents/slurm/generate_agent_traces.sh for the installation of the venv and packages.

#############################################
# Environment Setup
#############################################

# ─── Cache Directories ──────────────────────────────────────────── #
export CLEAN_CACHE="1"                                               # <-- Set to "1" to clean cache after job completion
export HF_DATASETS_CACHE="$workdir/.cache"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export PYTHONPYCACHEPREFIX="$HF_DATASETS_CACHE/.pycache"
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache/${SLURM_JOB_ID}"
export FLASHINFER_WORKSPACE_DIR="$workdir/.cache/flashinfer/${SLURM_JOB_ID}"

# ─── vLLM / FlashInfer / GDN ───────────────────────────────────── #
export VLLM_LOGGING_LEVEL=ERROR                                     # <-- Silence vLLM worker/engine INFO+WARNING logs (incl. Triton bundler spam)
export VLLM_USE_FLASHINFER_SAMPLER=0                                # <-- 0 to skip flashinfer sampler JIT entirely
export VLLM_USE_DEEP_GEMM=0                                         # <-- only needed for the FP8 model
export GDN_PREFILL_BACKEND="triton"                                 # <-- avoid FlashInfer SM90 GDN JIT on the BAF CUDA 12 toolkit

# ─── Model ─────────────────────────────────────────────────────── #
# Model type: "litellm" (remote API), "transformers" (local HF), or "vllm" (local vLLM server)
export MODEL_TYPE="vllm"                                            # <-- Change to "transformers" or "vllm" for local models
export MODEL_ID="$workdir/Qwen3.5-9B"                               # <-- Model identifier (HF name or LiteLLM id)
export API_KEY=""                                                   # <-- API key for LiteLLM models (also set in .env)
export API_BASE=""                                                  # <-- API base URL for LiteLLM (also set in .env)
export MAX_NEW_TOKENS=32000                                         # <-- Max tokens
export MODEL_MAX_LEN=""                                             # <-- Max model sequence length (empty = model's own max)
export TEMPERATURE=0.7                                              # <-- Sampling temperature (empty = model default)
export TOP_P=0.80                                                   # <-- Nucleus sampling top-p (empty = model default)
export TOP_K=20                                                     # <-- Top-k sampling cutoff (empty = model default)

# ─── Dataset ───────────────────────────────────────────────────── #
export DATASET_PATH="$workdir/data/samples.jsonl"                   # <-- Path to dataset (.json, .jsonl, .parquet)
export PROMPT_COLUMN="prompt"                                       # <-- Column name for the prompt
export ID_COLUMN="id"                                               # <-- Column name for trace IDs (empty = use prompt hashes)
export GROUND_TRUTH_COLUMN="ground_truth"                           # <-- Column name for ground-truth answers (empty = none)
export LANGUAGE="pt"                                                # <-- Language: system prompt + Wikipedia + DuckDuckGo region

# ─── Agent ─────────────────────────────────────────────────────── #
export MAX_STEPS=20                                                 # <-- Max agent steps per example
export EXECUTOR_TIMEOUT=120                                         # <-- Max seconds per tool execution step
export SYSTEM_PROMPT_FILE=""                                        # <-- Optional override: custom YAML prompt file (empty = use prompts/<language>.yaml)

# ─── Thinking & Planning ───────────────────────────────────────── #
export ENABLE_THINKING="0"                                          # <-- Set to "1" to enable thinking/reasoning mode (Transformers/vLLM)
export ENABLE_PLANNING="0"                                          # <-- Set to "1" to run a planning step before agent execution

# ─── Code-block delimiters ─────────────────────────────────────── #
export CODE_BLOCK_OPENING_TAG="<code>"                              # <-- Opening tag for code blocks
export CODE_BLOCK_CLOSING_TAG="</code>"                             # <-- Closing tag for code blocks

# ─── Output ────────────────────────────────────────────────────── #
export OUTPUT_DIR="$workdir/traces"                                 # <-- Base directory for saving traces
export SAVE_RAW_TRACES="0"                                          # <-- Set to "1" to also save raw traces (formatted traces are always saved)
export MAX_ENTRIES_PER_FILE="50000"                                 # <-- Max JSON objects per consolidated output file before rotation
export NO_RESUME="0"                                                # <-- Set to "1" to disable auto-resume (always start fresh)

mkdir -p "$OUTPUT_DIR"

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES node(s)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# [${SLURM_JOB_ID}] GLIBC version: $(ldd --version | head -n1)" >> "$out"
echo "# [${SLURM_JOB_ID}] Working directory: $workdir" >> "$out"
echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version)" >> "$out"

#############################################
# Main Job Execution
#############################################

# Build API flags
API_KEY_FLAG=""
if [[ -n "$API_KEY" ]]; then
    API_KEY_FLAG="--api-key $API_KEY"
fi

API_BASE_FLAG=""
if [[ -n "$API_BASE" ]]; then
    API_BASE_FLAG="--api-base $API_BASE"
fi

# Build optional flags
GROUND_TRUTH_FLAG=""
if [[ -n "$GROUND_TRUTH_COLUMN" ]]; then
    GROUND_TRUTH_FLAG="--ground-truth-column $GROUND_TRUTH_COLUMN"
fi

SYSTEM_PROMPT_FLAG=""
if [[ -n "$SYSTEM_PROMPT_FILE" ]]; then
    SYSTEM_PROMPT_FLAG="--system-prompt-file $SYSTEM_PROMPT_FILE"
fi

ID_COLUMN_FLAG=""
if [[ -n "$ID_COLUMN" ]]; then
    ID_COLUMN_FLAG="--id-column $ID_COLUMN"
fi

SAVE_RAW_TRACES_FLAG=""
if [ "$SAVE_RAW_TRACES" = "1" ]; then
    SAVE_RAW_TRACES_FLAG="--save-raw-traces"
fi

ENABLE_THINKING_FLAG=""
if [ "$ENABLE_THINKING" = "1" ]; then
    ENABLE_THINKING_FLAG="--enable-thinking"
fi

ENABLE_PLANNING_FLAG=""
if [ "$ENABLE_PLANNING" = "1" ]; then
    ENABLE_PLANNING_FLAG="--enable-planning"
fi

NO_RESUME_FLAG=""
if [ "$NO_RESUME" = "1" ]; then
    NO_RESUME_FLAG="--no-resume"
fi

# Build code-block delimiter flags (only passed when explicitly set)
CODE_BLOCK_OPENING_TAG_FLAG=""
if [[ -n "$CODE_BLOCK_OPENING_TAG" ]]; then
    CODE_BLOCK_OPENING_TAG_FLAG="--code-block-opening-tag $CODE_BLOCK_OPENING_TAG"
fi

CODE_BLOCK_CLOSING_TAG_FLAG=""
if [[ -n "$CODE_BLOCK_CLOSING_TAG" ]]; then
    CODE_BLOCK_CLOSING_TAG_FLAG="--code-block-closing-tag $CODE_BLOCK_CLOSING_TAG"
fi

MAX_ENTRIES_PER_FILE_FLAG=""
if [[ -n "$MAX_ENTRIES_PER_FILE" ]]; then
    MAX_ENTRIES_PER_FILE_FLAG="--max-entries-per-file $MAX_ENTRIES_PER_FILE"
fi

# Build sampling flags (only passed when explicitly set)
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

MODEL_MAX_LEN_FLAG=""
if [[ -n "$MODEL_MAX_LEN" ]]; then
    MODEL_MAX_LEN_FLAG="--model-max-len $MODEL_MAX_LEN"
fi

python3 "$workdir/synthetic/agents/generate_agent_traces.py" \
    --model-type "$MODEL_TYPE" \
    --model-id "$MODEL_ID" \
    --max-new-tokens $MAX_NEW_TOKENS \
    --dataset "$DATASET_PATH" \
    --prompt-column "$PROMPT_COLUMN" \
    --max-steps $MAX_STEPS \
    --executor-timeout $EXECUTOR_TIMEOUT \
    --output-dir "$OUTPUT_DIR" \
    --language "$LANGUAGE" \
    $TEMPERATURE_FLAG \
    $TOP_P_FLAG \
    $TOP_K_FLAG \
    $MODEL_MAX_LEN_FLAG \
    $API_KEY_FLAG \
    $API_BASE_FLAG \
    $GROUND_TRUTH_FLAG \
    $SYSTEM_PROMPT_FLAG \
    $ID_COLUMN_FLAG \
    $SAVE_RAW_TRACES_FLAG \
    $ENABLE_THINKING_FLAG \
    $ENABLE_PLANNING_FLAG \
    $NO_RESUME_FLAG \
    $CODE_BLOCK_OPENING_TAG_FLAG \
    $CODE_BLOCK_CLOSING_TAG_FLAG \
    $MAX_ENTRIES_PER_FILE_FLAG \
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
