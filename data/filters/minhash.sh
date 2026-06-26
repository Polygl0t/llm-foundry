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
#SBATCH --partition=vlm_long               # <-- Change to your partition
#SBATCH --job-name=dedup-minhash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=7-00:00:00
#SBATCH --mem=3900G
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################

# Set this to your workspace root (where you have the .venv and .modules.sh files).
workdir="/lustre/mlnvme/data/polyglot"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-dedup-minhash.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-dedup-minhash.$SLURM_JOB_ID"

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

# ===== Optional: Install Indic NLP Library for Indic languages =====
# pip3 install indic-nlp-library --no-cache-dir

#############################################
# Environment Setup
#############################################

export DATA_FOLDER="$workdir/hindi/dataset"
export LOGS_FOLDER="$workdir/hindi/logs"
export OUTPUT_MINHASH_SIGNATURES="$workdir/hindi/output_minhash_signatures"
export OUTPUT_MINHASH_BUCKET="$workdir/hindi/output_minhash_bucket"
export OUTPUT_REMOVED_IDS="$workdir/hindi/output_removed_ids"
export OUTPUT_DUPLICATED_SAMPLES="$workdir/hindi/output_duplicated_samples"
export OUTPUT_DEDUPLICATION_FINAL="$workdir/hindi/dataset_dedup"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export TOKENIZER_NAME_OR_PATH="Qwen/Qwen3-0.6B"
export LANGUAGE="hi"
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
# Before starting the loop, clean the folders in case they contain old data
mkdir -p "$LOGS_FOLDER" "$OUTPUT_MINHASH_SIGNATURES" "$OUTPUT_MINHASH_BUCKET" "$OUTPUT_REMOVED_IDS" "$OUTPUT_DUPLICATED_SAMPLES" "$OUTPUT_DEDUPLICATION_FINAL"
find "$LOGS_FOLDER" -mindepth 1 -delete 2>/dev/null || true
find "$OUTPUT_MINHASH_SIGNATURES" -mindepth 1 -delete 2>/dev/null || true
find "$OUTPUT_MINHASH_BUCKET" -mindepth 1 -delete 2>/dev/null || true
find "$OUTPUT_REMOVED_IDS" -mindepth 1 -delete 2>/dev/null || true
find "$OUTPUT_DUPLICATED_SAMPLES" -mindepth 1 -delete 2>/dev/null || true

python3 -u "$workdir/llm-foundry/data/filters/minhash.py" \
    --tasks $SLURM_CPUS_PER_TASK \
    --workers $SLURM_CPUS_PER_TASK \
    --cache_dir "$HF_DATASETS_CACHE" \
    --data_folder "$DATA_FOLDER" \
    --logs_folder "$LOGS_FOLDER" \
    --expand_metadata \
    --output_minhash_signatures "$OUTPUT_MINHASH_SIGNATURES" \
    --output_minhash_bucket "$OUTPUT_MINHASH_BUCKET" \
    --output_removed_ids "$OUTPUT_REMOVED_IDS" \
    --output_duplicated_samples "$OUTPUT_DUPLICATED_SAMPLES" \
    --output_deduplication_final "$OUTPUT_DEDUPLICATION_FINAL" \
    --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH" \
    --language "$LANGUAGE" 1>>"$out" 2>>"$err"

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
