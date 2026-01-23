#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=sgpu_medium             # <-- Change to your partition
#SBATCH --job-name=sft-context-extension
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --exclusive
#SBATCH --exclude=sgpu020
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

source "$workdir/.modules_amd.sh"
source "$workdir/.venv_amd/bin/activate"
#pip3 install trl --no-cache-dir
#pip3 install vllm==0.11.2 --no-cache-dir
#pip3 install flash_attn==2.8.2 --no-build-isolation --no-cache-dir

out="$workdir/run_outputs/out-sft-context-extension.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-sft-context-extension.$SLURM_JOB_ID"

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
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.tmp/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export HF_TOKEN="<your-token-here>" # <-- Change to your HF token
export WANDB_TOKEN="<your-token-here>" # <-- Change to your wandb token
export WANDB_DIR="$HF_DATASETS_CACHE/wandb"
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache/$SLURM_JOB_ID"
export NCCL_TIMEOUT=3600
export TORCH_FR_BUFFER_SIZE=1000
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_DIST_INIT_BARRIER=0  # Disable initial barrier that can cause timeouts
export TORCH_CPP_LOG_LEVEL=INFO
#export NCCL_DEBUG=INFO # Uncomment for NCCL debugging
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4
export NUM_PROCESSES=$SLURM_NTASKS
export NUM_MACHINES=$SLURM_NNODES
export head_node_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Where to save model checkpoints and send the log files
export CHECKPOINT_DIR="/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/checkpoints/models/Tucano2-qwen-0.5B-Base-32k"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion

hf auth login --token "$HF_TOKEN"
wandb login "$WANDB_TOKEN"

echo "# [${SLURM_JOB_ID}] Job started on $SLURM_JOB_NODELIST at: $(date)" > "$out"
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
export LAUNCHER="accelerate launch --config_file $workdir/long_ctx/.fsdp_ctx_parallel.yaml"

export PYTHON_FILE="$workdir/long_ctx/train_hf.py"

export ARGS="--dataset_type parquet \
--train_dataset_dir /lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/tokenized/gigaverbo-v2-long-32k/4 \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/tokenized/gigaverbo-v2-long-32k/5 \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/tokenized/gigaverbo-v2-long-32k/reasoning \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/tokenized/gigaverbo-v2-long-32k/retrieval \
--val_dataset_dir /lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/tokenized/gigaverbo-v2-long-32k/val \
--shuffle_dataset \
--cache_dir $HF_DATASETS_CACHE \
--num_proc 32 \
--dataloader_num_workers 16 \
--dataloader_prefetch_factor 4 \
--model_name_or_path /lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/checkpoints/models/Tucano2-qwen-0.5B-Base \
--checkpoint_dir $CHECKPOINT_DIR \
--hub_token $HF_TOKEN \
--max_length 32768 \
--new_max_position_embeddings 32768 \
--new_rope_theta 2000000 \
--save_steps 500 \
--eval_steps 500 \
--logging_steps 1 \
--use_liger_kernel \
--learning_rate 0.0000001 \
--weight_decay 0.1 \
--lr_scheduler_type cosine \
--warmup_steps 100 \
--max_steps 5000 \
--attn_implementation flash_attention_2 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--bf16 \
--tf32 \
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
cp "$out" "$CHECKPOINT_DIR/logs.txt"
cp "${BASH_SOURCE[0]}" "$CHECKPOINT_DIR/job.sh"
