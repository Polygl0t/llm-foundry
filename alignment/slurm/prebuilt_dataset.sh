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
#SBATCH --partition=lm_short               # <-- Change to your partition
#SBATCH --job-name=prebuilt-ft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=08:00:00
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################

# Set this to your workspace root (where you have the .venv and .modules.sh files).
workdir="/lustre/mlnvme/data/polyglot"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-prebuilt-ft.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-prebuilt-ft.$SLURM_JOB_ID"

#############################################
# Modules & Libraries Setup
#############################################

source $workdir/.modules.sh > "$out" 2>&1
source $workdir/.venv_trl/bin/activate

# ===== Installation =====
# See alignment/slurm/create_venv_marvin.sh for the installation of the venv and packages.

#############################################
# Environment Setup
#############################################

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Keep the intermediate Arrow files out of the way of the training cache: the build is a
# one-off, and its cache is safe to delete afterwards.
export HF_DATASETS_CACHE="$workdir/.cache/prebuilt-ft/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # <-- Set to "1" to clean cache after job completion

# The builder only ever reads LOCAL shard paths (`load_training_dataset` globs the
# filesystem), so no hub/wandb login is needed.

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES node(s)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_CPUS_PER_TASK CPUs per task" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# [${SLURM_JOB_ID}] GLIBC version: $(ldd --version | head -n1)" >> "$out"
echo "# [${SLURM_JOB_ID}] Working directory: $workdir" >> "$out"
echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version)" >> "$out"

#############################################
# Dataset Configuration
#############################################

DATASET_TYPE="jsonl"

TRAIN_DIRS=(
    "/data/general"
    "/data/code"
    "/data/function_call"
    "/data/math"
    "/data/retrieval_500m"
    "/data/rewriting"
    "/data/structured"
    "/data/summarization"
    "/data/system_prompts"
    "/data/translation"
)

OUTPUT_DIR="$workdir/data/alignment/prebuilt/sft-mix"

# The validation split is carved out of the training data with Dataset.train_test_split.
# A value in (0, 1) is a fraction, a value >= 1 a row count. Leave this empty and set
# VAL_DIRS to use an explicit held-out set instead (the two are mutually exclusive).
TEST_SIZE="0.02"
SEED=42
VAL_DIRS=()  # <-- e.g. ( "$workdir/data/alignment/validation" )

# How TEST_SIZE is spread over TRAIN_DIRS:
#   uniform -> every folder gives the SAME number of validation rows, so a huge folder
#              cannot crowd a small one out of the test set. A folder too small for its
#              share is clamped to half of itself and reported by the job log.
#   global  -> one random sample over all folders concatenated (folder share ~ size).
VAL_SPLIT_MODE="uniform"

# Optional sanity check on the raw schema: "messages" for SFT, "chosen" for DPO.
# Leave empty to skip.
VALIDATE_COLUMN="messages"

# Drop over-long samples BEFORE the train/validation split is built, so they reach
# neither split. The count is read from TOKEN_COUNT_COLUMN. Leave MAX_TOKEN_COUNT empty
# (or 0) to keep every row. NOTE: changing this does not invalidate an existing build --
# set OVERWRITE=1 to rebuild with the new limit.
MAX_TOKEN_COUNT="32768"
TOKEN_COUNT_COLUMN="token_count"

NUM_PROC=$SLURM_CPUS_PER_TASK
MAX_NUM_PROC=$SLURM_CPUS_PER_TASK
OVERWRITE="0"  # <-- Set to "1" to rebuild over an existing prebuilt dataset.

#############################################
# Main Job Execution
#############################################

if [ -f "$OUTPUT_DIR/.metadata" ] && [ "$OVERWRITE" != "1" ]; then
    echo "# [${SLURM_JOB_ID}] Skipping: $OUTPUT_DIR/.metadata already exists (set OVERWRITE=1 to rebuild)" >> "$out"
else
    # A prebuilt dataset always holds both splits, so exactly one source must be given.
    VAL_ARGS=()
    if [ "${#VAL_DIRS[@]}" -gt 0 ]; then
        VAL_ARGS+=(--val_dataset_dir "${VAL_DIRS[@]}")
    elif [ -n "$TEST_SIZE" ]; then
        VAL_ARGS+=(--test_size "$TEST_SIZE" --seed "$SEED" --val_split_mode "$VAL_SPLIT_MODE")
    else
        echo "# [${SLURM_JOB_ID}] ERROR: set either TEST_SIZE or VAL_DIRS." >> "$err"
        exit 1
    fi

    EXTRA_ARGS=()
    if [ -n "$VALIDATE_COLUMN" ]; then
        EXTRA_ARGS+=(--validate_column "$VALIDATE_COLUMN")
    fi
    if [ -n "$MAX_TOKEN_COUNT" ] && [ "$MAX_TOKEN_COUNT" != "0" ]; then
        EXTRA_ARGS+=(--max_token_count "$MAX_TOKEN_COUNT" --token_count_column "$TOKEN_COUNT_COLUMN")
    fi
    if [ "$OVERWRITE" = "1" ]; then
        EXTRA_ARGS+=(--overwrite)
    fi

    echo "# [${SLURM_JOB_ID}] Building prebuilt dataset in $OUTPUT_DIR" >> "$out"

    python3 "$workdir/llm-foundry/alignment/prebuilt_dataset.py" \
        --train_dataset_dir "${TRAIN_DIRS[@]}" \
        --output_dir "$OUTPUT_DIR" \
        --dataset_type "$DATASET_TYPE" \
        --num_proc "$NUM_PROC" \
        --max_num_proc "$MAX_NUM_PROC" \
        --cache_dir "$HF_DATASETS_CACHE" \
        "${VAL_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" 1>>"$out" 2>>"$err"
    STATUS=$?

    if [ "$STATUS" -ne 0 ]; then
        echo "# [${SLURM_JOB_ID}] Build FAILED with exit code $STATUS (see $err)" >> "$out"
        exit "$STATUS"
    fi
fi

#############################################
# Cleanup
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
