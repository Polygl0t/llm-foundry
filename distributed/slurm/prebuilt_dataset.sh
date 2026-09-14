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
# Learn about JUPITER jobs at:
# - https://apps.fz-juelich.de/jsc/hps/jupiter/index.html
#############################################
#SBATCH --account=ag_bit_flek              # <-- Change to your SLURM account
#SBATCH --partition=lm_short               # <-- Change to your partition
#SBATCH --job-name=prebuilt-dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --mem=1900G
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################

# Set this to your workspace root (where you have the .venv and .modules.sh files).
workdir="/lustre/mlnvme/data/polyglot"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-prebuilt.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-prebuilt.$SLURM_JOB_ID"

#############################################
# Modules & Libraries Setup
#############################################

source $workdir/.modules.sh > "$out" 2>&1
source $workdir/.venv_distributed/bin/activate

# ===== Installation =====
# See distributed/slurm/create_venv_marvin.sh for the installation of the venv and packages

#############################################
# Environment Setup
#############################################

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Keep the intermediate Arrow files out of the way of the training cache: the build is a
# one-off, and its cache is safe to delete afterwards.
export HF_DATASETS_CACHE="$workdir/.cache/prebuilt-build/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # <-- Set to "1" to clean cache after job completion

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

BLOCK_SIZE=4096
PACKED_DIR="$workdir/data/packed_$BLOCK_SIZE"

TRAIN_DIRS=(
    "$PACKED_DIR/gigaverbo_v2_4plus"
    "$PACKED_DIR/gigaverbo_v2_3"
    "$PACKED_DIR/fineweb_edu"
    "$PACKED_DIR/codeparrot"
    "$PACKED_DIR/finemath-4plus"
)

VAL_DIR="$PACKED_DIR/validation"
OUTPUT_DIR="$PACKED_DIR/prebuilt"
OVERWRITE="0"  # <-- Set to "1" to rebuild over an existing prebuilt dataset.

#############################################
# Main Job Execution
#############################################

if [ -f "$OUTPUT_DIR/.metadata" ] && [ "$OVERWRITE" != "1" ]; then
    echo "# [${SLURM_JOB_ID}] Skipping: $OUTPUT_DIR/.metadata already exists (set OVERWRITE=1 to rebuild)" >> "$out"
else
    OVERWRITE_ARGS=()
    if [ "$OVERWRITE" = "1" ]; then
        OVERWRITE_ARGS+=(--overwrite)
    fi

    echo "# [${SLURM_JOB_ID}] Building prebuilt dataset in $OUTPUT_DIR" >> "$out"

    python3 "$workdir/llm-foundry/distributed/prebuilt_dataset.py" \
        --train_dataset_dir "${TRAIN_DIRS[@]}" \
        --val_dataset_dir "$VAL_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --dataset_type parquet \
        --num_proc "$SLURM_CPUS_PER_TASK" \
        --max_num_proc "$SLURM_CPUS_PER_TASK" \
        --cache_dir "$HF_DATASETS_CACHE" \
        --shuffle --seed 1337 \
        "${OVERWRITE_ARGS[@]}" 1>>"$out" 2>>"$err"
fi

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
