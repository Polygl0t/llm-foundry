#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_bit_flek              # <-- Change to your SLURM account
#SBATCH --partition=lm_medium              # <-- Change to your partition
#SBATCH --job-name=sft-filter
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
workspace_name="polyglot_datasets"         # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0
echo "Job started at: $(date)"

out="$workdir/run_outputs/out-sft-filter.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-sft-filter.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
source "$workdir/.modules.sh"
# python3 -m venv "$workdir/.venv_amd"
source "$workdir/.venv_amd/bin/activate"

# pip3 install --upgrade pip
# git clone --depth 1 --branch main https://github.com/Polygl0t/llm-foundry.git
# pip3 install -e "$workdir/llm-foundry/.[data]" --no-cache-dir

# ==== For this script, you will need matplotlib for visualizations ====
# pip3 install matplotlib --no-cache-dir

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export INPUT_DIR="$workdir/data/sft"
export INPUT_TYPE="jsonl"
export OUTPUT_DIR="$workdir/data/sft_filtered"
export OUTPUT_TYPE="jsonl"
export MESSAGES_COLUMN="messages"
export QUALITY_SCORE_COLUMN="instruct_score"
export NUM_PROC=$SLURM_CPUS_PER_TASK
export MIN_TOKEN_COUNT=120
export MAX_TOKEN_COUNT=1000
export MIN_QUALITY_SCORE=4.5
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
python3 $workdir/llm-foundry/data/filters/sft_filters.py \
    --input_dir "$INPUT_DIR" \
    --input_type "$INPUT_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    --output_type "$OUTPUT_TYPE" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --messages_column "$MESSAGES_COLUMN" \
    --min_token_count $MIN_TOKEN_COUNT \
    --max_token_count $MAX_TOKEN_COUNT \
    --filter_invalid_markers \
    --filter_repetition_loops \
    --filter_undecoded_sequences \
    --filter_incomplete_sentences \
    --filter_malformed_code_blocks \
    --filter_corrupted_code \
    --quality_score_column "$QUALITY_SCORE_COLUMN" \
    --min_quality_score $MIN_QUALITY_SCORE \
    --num_proc $NUM_PROC 1>>"$out" 2>>"$err"

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
