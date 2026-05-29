#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn about SLURM sbatch options at:
# - https://slurm.schedmd.com/sbatch.html
#
# Learn about job submissions (Marvin|Bender) at:
# - https://wiki.hpc.uni-bonn.de/en/running_jobs
#
# Learn about Marvin|Bender dual software stacks at:
# - https://wiki.hpc.uni-bonn.de/en/dualstacks
#############################################
#SBATCH --account=ag_bit_flek              # <-- Change to your SLURM account
#SBATCH --partition=sgpu_short             # <-- Change to your partition
#SBATCH --job-name=sft
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

# Set this to your workspace root (where you have the .venv and .modules.sh files).
workdir="/lustre/mlnvme/data/polyglot"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-sft-trainer.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-sft-trainer.$SLURM_JOB_ID"

#############################################
# Modules & Libraries Setup
#############################################

source $workdir/.modules.sh > "$out" 2>&1
# python3 -m venv $workdir/.venv_trl
source "$workdir/.venv_trl/bin/activate"

# ===== Upgrade PIP =====
# pip3 install --upgrade pip

# ===== LLM Foundry Install =====
# git clone --depth 1 --branch main https://github.com/Polygl0t/llm-foundry.git
# pip3 install -e "$workdir/llm-foundry/.[trl]" --no-cache-dir

# ===== ALL HAIL FLASH-ATTN! =====
# Option A – Use a prebuilt wheel, no nvcc or compilation needed.
#
#   Step 1: Find the right wheel for your environment using the search tool:
#           https://mjunya.com/flash-attention-prebuild-wheels/
#           Filter by: flash_attn version, Python version, PyTorch version, CUDA version.
#           The community repo (https://github.com/mjun0812/flash-attention-prebuild-wheels)
#           covers many more CUDAxtorch combinations than the official releases.
#
#   Step 2: Copy the direct-install URL and replace the one below.
#           Wheel name format: flash_attn-<FA>+cu<CUDA>torch<torch>-cp<py>-cp<py>-linux_x86_64.whl
#
#   FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE  fail fast if no matching wheel is found
#                                         instead of silently falling back to a source build
# Example:
# FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip3 install \
#    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu126torch2.8-cp312-cp312-linux_x86_64.whl \
#    --no-cache-dir
#
# Option B – Build from source.
#            This takes time ... However, it can be the only option if no
#            compatible wheel exists for your environment.    
#            The build process requires a working nvcc setup and a compatible PyTorch installation.
#            The following environment variables and pip options can help ensure a smooth build:
#      
#   FLASH_ATTENTION_FORCE_BUILD=TRUE  flash-attn's setup.py skips its wheel search
#   --no-binary :flash-attn:          pip-level guard: never use a prebuilt wheel
#   --no-build-isolation              keep the current venv active (avoids reinstalling torch)
#   MAX_JOBS                          cap parallel C++ compilation to avoid OOM
#   FLASH_ATTN_CUDA_ARCHS              specify your GPU architectures to speed up the build
#
# Example:
# FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 FLASH_ATTN_CUDA_ARCHS="80;90" \
#   pip3 install flash-attn==2.8.3 --no-binary :flash-attn: --no-build-isolation --no-cache-dir


# ===== OPTIONAL: Specialized Attention Packages =====
# These packages provide optimized CUDA kernels for specific attention mechanisms.
# Uncomment only if your model uses the corresponding attention type.

# Flash Linear Attention (for fast linear attention implementations)
# Causal Conv1D (for models using causal convolutional layers instead of standard attention)
# - Note: flash-linear-attention requires PyTorch >= 2.7.0. However, on Bender, the latest CUDA
#         available is CUDA 12.4, which is not compatible with the release versions of PyTorch 2.7.x.
# pip3 install flash-linear-attention --no-cache-dir
# pip3 install causal-conv1d --no-build-isolation --no-cache-dir

#############################################
# Environment Setup
#############################################
# PyTorch NCCL environment variables:
# - https://github.com/pytorch/pytorch/blob/main/docs/source/cuda_environment_variables.rst
#
# PyTorch Distributed Documentation:
# - https://github.com/pytorch/pytorch/blob/main/docs/source/distributed.md
# 
# NCCL Documentation:
# - https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
#############################################

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.tmp/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export HF_TOKEN="<your-token-here>"
export WANDB_TOKEN="<your-token-here>"
export WANDB_DIR="$HF_DATASETS_CACHE/wandb"
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache/$SLURM_JOB_ID"
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_IB_TIMEOUT=24
export NCCL_IB_RETRY_CNT=7
export TORCH_FR_BUFFER_SIZE=1000
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
# export NCCL_DEBUG=INFO # Uncomment for NCCL debugging
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=$SLURM_NTASKS_PER_NODE
export NUM_PROCESSES=$SLURM_NTASKS
export NUM_MACHINES=$SLURM_NNODES
export head_node_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export CHECKPOINT_DIR="./checkpoints/MyModel-DPO-$SLURM_JOB_ID"
export CLEAN_CACHE="1"  # <-- Set to "1" to clean cache after job completion

hf auth login --token "$HF_TOKEN"
wandb login "$WANDB_TOKEN"

echo "# [${SLURM_JOB_ID}] Job started on $SLURM_JOB_NODELIST at: $(date)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# [${SLURM_JOB_ID}] GLIBC version: $(ldd --version | head -n1)" >> "$out"
echo "# [${SLURM_JOB_ID}] Working directory: $workdir" >> "$out"
echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version)" >> "$out"

#############################################
# Main Job Execution (Distributed Training)
#############################################
# Accelerate Documentation
# - https://huggingface.co/docs/accelerate/package_reference/cli
#############################################

export LAUNCHER="accelerate launch --config_file $workdir/llm-foundry/alignment/.ddp_config.yaml"

export PYTHON_FILE="$workdir/llm-foundry/alignment/sft_trainer.py"

export ARGS="--dataset_type jsonl \
--train_dataset_dir /data/general \
/data/code \
/data/function_call \
/data/math \
/data/retrieval_500m \
/data/rewriting \
/data/structured \
/data/summarization \
/data/system_prompts \
/data/translation \
--shuffle_dataset \
--cache_dir $HF_DATASETS_CACHE \
--num_proc 32 \
--model_name_or_path Polygl0t/Tucano2-qwen-0.5B-Base \
--chat_template_path /assets/chat_template.jinja \
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
