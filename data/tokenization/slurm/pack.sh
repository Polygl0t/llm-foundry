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
#SBATCH --job-name=pack
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

out="$workdir/run_outputs/out-pack.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-pack.$SLURM_JOB_ID"

#############################################
# Modules & Libraries Setup
#############################################

source $workdir/.modules.sh > "$out" 2>&1
# python3 -m venv $workdir/.venv_intel
source $workdir/.venv_intel/bin/activate

# ===== LLM Foundry Install =====
# pip3 install --upgrade pip --no-cache-dir
# git clone --depth 1 --branch main https://github.com/Polygl0t/llm-foundry.git
# pip3 install -e "$workdir/llm-foundry/.[data]" --no-cache-dir

# ===== Alternatively, install with uv =====
# pip3 install --upgrade pip --no-cache-dir
# pip3 install uv
# uv pip install -e "$workdir/llm-foundry/.[data]" --no-cache

#############################################
# Environment Setup
#############################################

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # <-- Set to "1" to clean cache after job completion

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_CPUS_PER_TASK CPUs per task" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# [${SLURM_JOB_ID}] GLIBC version: $(ldd --version | head -n1)" >> "$out"
echo "# [${SLURM_JOB_ID}] Working directory: $workdir" >> "$out"
echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version)" >> "$out"

#############################################
# Main Job Execution
#############################################

BLOCK_SIZE=8192
TOKENIZED_DIR="$workdir/data/portuguese/tokenized"
PACKED_DIR="$workdir/data/portuguese/packed_$BLOCK_SIZE"

for folder in "$TOKENIZED_DIR"/*/; do
    name=$(basename "$folder")

    # Skip hidden folders (those starting with ".")
    case "$name" in
        .*) continue ;;
    esac

    # Skip if the output folder already has a .metadata file
    # (a previous job finished packing it completely).
    if [ -f "$PACKED_DIR/$name/.metadata" ]; then
        echo "# [${SLURM_JOB_ID}] Skipping $name: already packed (.metadata found)" >> "$out"
        continue
    fi

    # If the output folder exists but has no .metadata, the previous
    # job was interrupted part-way. Remove it and start fresh.
    if [ -d "$PACKED_DIR/$name" ]; then
        echo "# [${SLURM_JOB_ID}] Cleaning incomplete output for $name (no .metadata)" >> "$out"
        rm -rf "$PACKED_DIR/$name"
    fi

    echo "# [${SLURM_JOB_ID}] Packing $name" >> "$out"

    python3 $workdir/llm-foundry/data/tokenization/pack.py \
        --input_path "$TOKENIZED_DIR/$name" \
        --output_dir "$PACKED_DIR/$name" \
        --strategy concatenate \
        --block_size $BLOCK_SIZE \
        --cache_dir "$HF_DATASETS_CACHE" \
        --num_proc $SLURM_CPUS_PER_TASK 1>>"$out" 2>>"$err"
done

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
