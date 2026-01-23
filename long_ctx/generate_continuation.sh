#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel            # <-- Change to your SLURM account
#SBATCH --partition=sgpu_short              # <-- Change to your partition
#SBATCH --job-name=synthetic-gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive
#SBATCH --dependency=afterany:22556319      # <-- Change dependency as needed

#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="mlnvme"                       # <-- Change to your filesystem
workspace_name="polyglot"               # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-gen.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-gen.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
source "$workdir/.modules_amd.sh"
source "$workdir/.venv_amd/bin/activate"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution
#############################################
export CUDA_VISIBLE_DEVICES=0
python3 $workdir/long_ctx/generate_continuation.py \
    --model_name "$workdir/checkpoints/models/Tucano2-qwen-0.6B-Base-CPT" \
    --output_file "$workdir/generated_output_8100_tokens.txt" \
    --max_context_tokens 7800 \
    --max_tokens 300 \
    --temperature 0.5 \
    --top_k 50 \
    --top_p 0.95 \
    --repetition_penalty 1.5 \
    --cache_dir "$workdir/.cache" 1>$out 2>$err

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
