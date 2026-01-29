#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=lm_long             # <-- Change to your partition
#SBATCH --job-name=decontaminate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=7-00:00:00
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################
export username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
export file_system="scratch"                      # <-- Change to your filesystem
export workspace_name="polyglot_datasets"         # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-decontaminate.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-decontaminate.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
source "$workdir/.modules_intel.sh"
source "$workdir/.venv_intel/bin/activate"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_CPUS_PER_TASK CPUs per task" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution
#############################################
python3 $workdir/decontaminate.py \
    --input_pattern "$workdir/portuguese/gigaverbo-v2-sft/contaminated/*.jsonl" \
    --reference_files "$workdir/portuguese/gigaverbo-v2-sft/references.jsonl" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --num_proc $SLURM_CPUS_PER_TASK \
    --output_dir "$workdir/portuguese/gigaverbo-v2-sft/decontaminated" \
    --min_k 8 \
    --max_k 32 \
    --allow_one_token_mismatch \
    --approx_max_k 10 1>>"$out" 2>>"$err"

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
