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
#SBATCH --job-name=train-annotators
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

for i in $(seq 0 $((SLURM_NTASKS_PER_NODE - 1))); do
    eval "out$i=\"\$workdir/run_outputs/out$i.\$SLURM_JOB_ID\""
    eval "err$i=\"\$workdir/run_outputs/err$i.\$SLURM_JOB_ID\""
done

#############################################
# Modules & Libraries Setup
#############################################

source "$workdir/.modules.sh"
# python3 -m venv "$workdir/.venv_amd"
source "$workdir/.venv_amd/bin/activate"

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

export HF_TOKEN="<your-token-here>"
export WANDB_TOKEN="<your-token-here>"
export ID_LABEL="Edu-Score"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion

hf auth login --token "$HF_TOKEN"
wandb login "$WANDB_TOKEN"

for i in $(seq 0 $((SLURM_NTASKS_PER_NODE - 1))); do
    eval "out_var=\"\$out$i\""
    eval "err_var=\"\$err$i\""
    echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out_var"
    echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out_var"
    echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out_var"
    echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out_var"
    echo "# [${SLURM_JOB_ID}] GLIBC version: $(ldd --version | head -n1)" >> "$out_var"
    echo "# [${SLURM_JOB_ID}] Working directory: $workdir" >> "$out_var"
    echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version)" >> "$out_var"
done

#############################################
# Main Job Execution (Parallel Training)
#############################################

export CUDA_VISIBLE_DEVICES=0
export UCX_NET_DEVICES=mlx5_0:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/llm-foundry/data/filters/train_annotator.py \
    --dataset_path "$workdir/data" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --model_name "PORTULAN/albertina-100m-portuguese-ptbr-encoder" \
    --checkpoint_dir "$workdir/checkpoints/albertina-100m" \
    --freeze \
    --hub_token "$HF_TOKEN" \
    --bf16 \
    --tf32 \
    --id_label $ID_LABEL 1>$out0 2>$err0 &

export CUDA_VISIBLE_DEVICES=1
export UCX_NET_DEVICES=mlx5_1:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/llm-foundry/data/filters/train_annotator.py \
    --dataset_path "$workdir/data" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --model_name "sagui-nlp/debertinha-ptbr-xsmall" \
    --checkpoint_dir "$workdir/checkpoints/debertinha" \
    --freeze \
    --hub_token "$HF_TOKEN" \
    --bf16 \
    --tf32 \
    --id_label $ID_LABEL 1>$out1 2>$err1 &

export CUDA_VISIBLE_DEVICES=2
export UCX_NET_DEVICES=mlx5_3:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/llm-foundry/data/filters/train_annotator.py \
    --dataset_path "$workdir/data" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --model_name "pablocosta/bertabaporu-large-uncased" \
    --checkpoint_dir "$workdir/checkpoints/bertabaporu-large" \
    --freeze \
    --hub_token "$HF_TOKEN" \
    --bf16 \
    --tf32 \
    --id_label $ID_LABEL 1>$out2 2>$err2 &

export CUDA_VISIBLE_DEVICES=3
export UCX_NET_DEVICES=mlx5_2:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/llm-foundry/data/filters/train_annotator.py \
    --dataset_path "$workdir/data" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --model_name "eduagarcia/RoBERTaCrawlPT-base" \
    --checkpoint_dir "$workdir/checkpoints/roberta-crawlpt" \
    --freeze \
    --hub_token "$HF_TOKEN" \
    --bf16 \
    --tf32 \
    --id_label $ID_LABEL 1>$out3 2>$err3 &

wait

#############################################
# End of Script
#############################################
# Clean HF_DATASETS_CACHE folder if requested
if [ "$CLEAN_CACHE" = "1" ]; then
    echo "# [${SLURM_JOB_ID}] Cleaning HF_DATASETS_CACHE" >> "$out0"
    if [ -d "$HF_DATASETS_CACHE" ]; then
        find "$HF_DATASETS_CACHE" -mindepth 1 -delete 2>/dev/null || true
    fi
else
    echo "# [${SLURM_JOB_ID}] Skipping cache cleanup (CLEAN_CACHE=$CLEAN_CACHE)" >> "$out0"
fi

for i in $(seq 0 $((SLURM_NTASKS_PER_NODE - 1))); do
    eval "out_var=\"\$out$i\""
    eval "err_var=\"\$err$i\""
    echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out_var"
done