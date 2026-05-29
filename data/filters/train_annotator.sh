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
#SBATCH --partition=sgpu_long              # <-- Change to your partition
#SBATCH --job-name=train-annotator
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00
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

out="$workdir/run_outputs/out-train-annotator.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-train-annotator.$SLURM_JOB_ID"

#############################################
# Modules & Libraries Setup
#############################################

source $workdir/.modules.sh > "$out" 2>&1
# python3 -m venv $workdir/.venv_distributed
source $workdir/.venv_distributed/bin/activate

# ===== Upgrade PIP =====
# pip3 install --upgrade pip

# ===== LLM Foundry Install =====
# git clone --depth 1 --branch main https://github.com/Polygl0t/llm-foundry.git
# pip3 install -e "$workdir/llm-foundry/.[distributed]" --no-cache-dir

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
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export HF_TOKEN="<your-token-here>"
export WANDB_TOKEN="<your-token-here>"
export CLEAN_CACHE="1"  # <-- Set to "1" to clean cache after job completion

hf auth login --token "$HF_TOKEN"
wandb login "$WANDB_TOKEN"

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" >> "$out"
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

export CUDA_VISIBLE_DEVICES=0,1,2,3

export LAUNCHER="accelerate launch --config_file $workdir/llm-foundry/data/filters/.ddp_config.yaml/"

export PYTHON_FILE="$workdir/llm-foundry/data/filters/train_annotator.py"

export ARGS="--train_dataset_dir ./data \
--dataset_type jsonl \
--shuffle_dataset \
--cache_dir $HF_DATASETS_CACHE \
--num_proc $SLURM_CPUS_PER_TASK \
--model_name Qwen/Qwen3-0.6B \
--checkpoint_dir ./checkpoints \
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
