#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=lm_medium              # <-- Change to your partition
#SBATCH --job-name=shuffle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=1-00:00:00
#SBATCH --mem=1900G
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="scratch"                      # <-- Change to your filesystem
workspace_name="polyglot_datasets"        # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-shuffle.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-shuffle.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
source $workdir/.modules_intel.sh
source $workdir/.venv_intel/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_CPUS_PER_TASK CPUs per task" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution
#############################################
python3 "$workdir/shuffle.py" \
    --dataset_dir "$workdir/portuguese/tokenized/starcoder/c" \
    "$workdir/portuguese/tokenized/starcoder/css" \
    "$workdir/portuguese/tokenized/starcoder/git_commits_cleaned" \
    "$workdir/portuguese/tokenized/starcoder/github_issues_filtered_structured" \
    "$workdir/portuguese/tokenized/starcoder/javascript" \
    "$workdir/portuguese/tokenized/starcoder/jupyter_scripts_dedup_filtered" \
    "$workdir/portuguese/tokenized/starcoder/jupyter_structured_clean_dedup" \
    "$workdir/portuguese/tokenized/starcoder/markdown" \
    "$workdir/portuguese/tokenized/starcoder/python" \
    "$workdir/portuguese/tokenized/starcoder/sql" \
    --dataset_type "parquet" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --output_dir "$workdir/portuguese/tokenized/starcoder/all" 1>>"$out" 2>>"$err"

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
