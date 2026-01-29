#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=vlm_long               # <-- Change to your partition
#SBATCH --job-name=quality-filters
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=7-00:00:00
#SBATCH --mem=1800G
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="scratch"                      # <-- Change to your filesystem
workspace_name="polyglot_datasets"         # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-quality-filters.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-quality-filters.$SLURM_JOB_ID"

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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion
export CONFIG_FOLDER="$workdir/.configs"
export DATA_FOLDER="$workdir/bengali/bengali_text"
export LOGS_FOLDER="$workdir/bengali/logs"
export FINAL_OUTPUT_FOLDER="$workdir/bengali/final_output"
export TOKENIZER_NAME_OR_PATH="Qwen/Qwen3-0.6B" # Qwen3 has a good multilingual tokenizer
export LANGUAGE="bn"  # Bengali language code

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_CPUS_PER_TASK CPUs per task" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution
#############################################
# Before starting the loop, clean the folders in case they contain old data
mkdir -p "$LOGS_FOLDER" "$FINAL_OUTPUT_FOLDER"
find "$LOGS_FOLDER" -mindepth 1 -delete 2>/dev/null || true

python3 -u "$workdir/quality_filters.py" \
    --tasks $SLURM_CPUS_PER_TASK \
    --workers $SLURM_CPUS_PER_TASK \
    --cache_dir "$HF_DATASETS_CACHE" \
    --config_folder "$CONFIG_FOLDER" \
    --data_folder "$DATA_FOLDER" \
    --logs_folder "$LOGS_FOLDER" \
    --expand_metadata \
    --final_output_folder "$FINAL_OUTPUT_FOLDER" \
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
