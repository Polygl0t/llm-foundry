#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=sgpu_medium              # <-- Change to your partition
#SBATCH --job-name=synthetic-cai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                          # <-- Change to the corresponding username that created the workspace
file_system="mlnvme"                             # <-- Change to your filesystem
workspace_name="nanotronics"                     # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/synth/synth_logs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-synthetic-cai.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-synthetic-cai.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
source $workdir/.modules_amd.sh
source $workdir/.venv_amd/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion
export HF_TOKEN="<your-token-here>" # <-- Change to your HF token
export PROMPT_PREFIX=""
export PROMPT_SUFFIX=""
export CONSTITUTION_FILE="$workdir/synth/REASONING_CONSTITUTION.md"
export MODEL_NAME="Qwen/Qwen2.5-32B-Instruct" #  Qwen/Qwen2.5-32B-Instruct [Used for Harmless+Harmfull | chosen] / huihui-ai/Qwen2.5-32B-Instruct-abliterated [Used for Harmfull | rejected] / Qwen/Qwen2.5-7B-Instruct [Used for Harmless | rejected]
export COLUMN_NAME="instruction"
export COLUMN_NAME2="rejected_response"
export OUTPUT_DIR="$workdir/portuguese/gigaverbo-v2-dpo"
export MAX_LENGTH=4096
export MAX_CHUNK_SIZE=2048
export TEMPERATURE=0.7
export TOP_K=20
export TOP_P=0.8
export REPETITION_PENALTY=1.2
export NUM_RETURN_SEQUENCES=1
export ENABLE_CRITIQUE="0"  # Set to "1" to enable constitutional critique and revision loop
export MAX_REVISIONS=1      # Number of critique/revision iterations

hf auth login --token "$HF_TOKEN"

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES node(s)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution
#############################################
# Build critique flag
CRITIQUE_FLAG=""
if [ "$ENABLE_CRITIQUE" = "1" ]; then
    CRITIQUE_FLAG="--enable_critique --max_revisions $MAX_REVISIONS"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 "$workdir/generate_cai.py" \
    --model_name "$MODEL_NAME" \
    --tensor_parallel_size 4 \
    --dataset_path "$OUTPUT_DIR/harmless_prompts.jsonl" \
    --column_name "$COLUMN_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --output_file "harmless_with_chosen.jsonl" \
    --max_length $MAX_LENGTH \
    --max_chunk_size $MAX_CHUNK_SIZE \
    --temperature $TEMPERATURE \
    --top_k $TOP_K \
    --top_p $TOP_P \
    --repetition_penalty $REPETITION_PENALTY \
    --num_return_sequences $NUM_RETURN_SEQUENCES \
    --cache_dir "$HF_DATASETS_CACHE" \
    --constitution_file "$CONSTITUTION_FILE" \
    --prompt_prefix "$PROMPT_PREFIX" \
    --prompt_suffix "$PROMPT_SUFFIX" \
    $CRITIQUE_FLAG 1>>"$out" 2>>"$err"

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
