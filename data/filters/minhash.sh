#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
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
username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="scratch"                      # <-- Change to your filesystem
workspace_name="polyglot_datasets"         # <-- Change to your workspace/project nameme

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-dedup-minhash.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-dedup-minhash.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
source $workdir/.modules_intel.sh
source $workdir/.venv_intel/bin/activate
# pip3 install datatrove[io,processing] --no-cache-dir
# pip3 install indic-nlp-library --no-cache-dir
# pip3 install stanza --no-cache-dir
# pip3 install spacy --no-cache-dir
# pip3 install pyyaml==6.0.2 --no-cache-dir
# pip3 install indic-nlp-library --no-cache-dir

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
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion
export TOKENIZER_NAME_OR_PATH="Qwen/Qwen3-0.6B" # Qwen3 has a good multilingual tokenizer
export LANGUAGE="hi"  # Hindi language code

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_CPUS_PER_TASK CPUs per task" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

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

python3 -u "$workdir/minhash.py" \
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
