#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=sgpu_long              # <-- Change to your partition
#SBATCH --job-name=train-classifier
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --exclusive
#SBATCH --dependency=afterany:22556203

#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="mlnvme"                       # <-- Change to your filesystem
workspace_name="nanotronics"               # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_training_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_training_outputs/out-train-classifier.$SLURM_JOB_ID"
err="$workdir/run_training_outputs/err-train-classifier.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
# References:
# - PyTorch NCCL environment variables:
# https://github.com/pytorch/pytorch/blob/main/docs/source/cuda_environment_variables.rst
# - PyTorch Distributed Documentation:
# https://github.com/pytorch/pytorch/blob/main/docs/source/distributed.md
# - NCCL Documentation:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
#############################################
source "$workdir/.modules_amd.sh"
source "$workdir/.venv_amd/bin/activate"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion
export HF_TOKEN="<your-token-here>" # <-- Change to your HF token
export WANDB_TOKEN="<your-token-here>" # <-- Change to your wandb token

hf auth login --token "$HF_TOKEN"
wandb login "$WANDB_TOKEN"

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution (Distributed Training)
#############################################
# References:
# - Accelerate Documentation 
# https://huggingface.co/docs/accelerate/package_reference/cli
#############################################
export CUDA_VISIBLE_DEVICES=0,1,2,3

export LAUNCHER="accelerate launch --config_file $workdir/.ddp_config.yaml"

export PYTHON_FILE="$workdir/train_classifier.py"

export ARGS="--train_dataset_dir ./portuguese/portuguese-instruct-quality-qwen-annotations/data \
--dataset_type jsonl \
--shuffle_dataset \
--cache_dir $HF_DATASETS_CACHE \
--num_proc $SLURM_CPUS_PER_TASK \
--model_name Qwen/Qwen3-4B \
--checkpoint_dir ./checkpoints/models/portuguese-qwen3-4b-instruct-quality-classifier \
--hub_token $HF_TOKEN \
--freeze \
--test_size 10000 \
--max_length 6032 \
--eval_steps 3000 \
--save_steps 3000 \
--logging_steps 1 \
--learning_rate 0.00005 \
--weight_decay 0.1 \
--lr_scheduler_type cosine \
--warmup_ratio 0.1 \
--num_train_epochs 2 \
--attn_implementation flash_attention_2 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing \
--bf16 \
--tf32 \
--id_label INS-Score \
"

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $ARGS"
$CMD 1>>"$out" 2>>"$err"

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
