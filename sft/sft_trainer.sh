#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=sgpu_short             # <-- Change to your partition
#SBATCH --job-name=sft-tucano-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="mlnvme"                       # <-- Change to your filesystem
workspace_name="polyglot"                  # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-sft-trainer.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-sft-trainer.$SLURM_JOB_ID"

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
# python3 -m venv $workdir/.venv_trl
source "$workdir/.venv_trl/bin/activate"

# pip3 install --upgrade pip setuptools wheel --no-cache-dir
# pip3 install trl --no-cache-dir
# pip3 install vllm==0.11.2 --no-cache-dir
# pip3 install flash_attn==2.8.2 --no-build-isolation --no-cache-dir
# pip3 install codecarbon==3.0.6 wandb==0.21.0 --no-cache-dir
# pip3 install liger-kernel --no-cache-dir

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.tmp/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export HF_TOKEN="<your-token-here>" # <-- Change to your HF token
export WANDB_TOKEN="<your-token-here>" # <-- Change to your wandb token
export WANDB_DIR="$HF_DATASETS_CACHE/wandb"
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache/$SLURM_JOB_ID"
export NCCL_TIMEOUT=3600 # <-- Avoid Timeout errors due to tokenization taking too long
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 # <-- Avoid Timeout errors due to tokenization taking too long
export NCCL_IB_TIMEOUT=24 # <-- InfiniBand Verbs Timeout
export NCCL_IB_RETRY_CNT=7 # <-- InfiniBand retry count.
export TORCH_FR_BUFFER_SIZE=1000
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_P2P_DISABLE=0 # Enable P2P communication
export NCCL_SHM_DISABLE=0 # Enable Shared Memory communication
#export NCCL_DEBUG=INFO # Uncomment for NCCL debugging
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=$SLURM_NTASKS_PER_NODE
export NUM_PROCESSES=$SLURM_NTASKS
export NUM_MACHINES=$SLURM_NNODES
export head_node_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Where to save model checkpoints and send the log files
export CHECKPOINT_DIR="/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/checkpoints/models/Tucano2-qwen-0.6B-Base-CPT-Instruct"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion

hf auth login --token "$HF_TOKEN"
wandb login "$WANDB_TOKEN"

echo "# [${SLURM_JOB_ID}] Job started on $SLURM_JOB_NODELIST at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution (Distributed Training)
#############################################
# References:
# - Accelerate Documentation 
# https://huggingface.co/docs/accelerate/package_reference/cli
#############################################
export LAUNCHER="accelerate launch --config_file $workdir/sft/.ddp_config.yaml"

export PYTHON_FILE="$workdir/sft/sft_trainer.py"

export ARGS="--dataset_type jsonl \
--train_dataset_dir /lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/general \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/code \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/function_call \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/function_call \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/function_call \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/function_call \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/math \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/retrieval_500m \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/rewriting \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/rewriting \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/rewriting \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/rewriting \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/structured \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/summarization \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/system_prompts \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/system_prompts \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/system_prompts \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/system_prompts \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/translation \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/translation \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/translation \
/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft/translation \
--shuffle_dataset \
--cache_dir $HF_DATASETS_CACHE \
--num_proc 32 \
--model_name_or_path /lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/checkpoints/models/Tucano2-qwen-0.6B-Base-CPT \
--chat_template_path /lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/checkpoints/tokenizers/chat/jinja_templates/chat_template_pt_non_reasoning.jinja \
--checkpoint_dir $CHECKPOINT_DIR \
--hub_token $HF_TOKEN \
--save_test_set \
--max_length 4096 \
--save_steps 2000 \
--logging_steps 1 \
--packing \
--assistant_only_loss \
--use_liger_kernel \
--learning_rate 0.00005 \
--weight_decay 0.1 \
--lr_scheduler_type cosine \
--warmup_ratio 0.1 \
--num_train_epochs 1 \
--attn_implementation flash_attention_2 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--bf16 \
--tf32 \
--gradient_checkpointing \
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
