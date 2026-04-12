#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=sgpu_long              # <-- Change to your partition
#SBATCH --job-name=fsdp-training
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

username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="mlnvme"                       # <-- Change to your filesystem
workspace_name="polyglot"                  # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out.$SLURM_JOB_ID"
err="$workdir/run_outputs/err.$SLURM_JOB_ID"

#############################################
# Working Build : )
# Python 3.12, CUDA 12.6, PyTorch 2.8, and CXX11 ABI set to TRUE.
#############################################

source $workdir/.modules_amd.sh
# python3 -m venv $workdir/.venv_fsdp
source $workdir/.venv_fsdp/bin/activate
# pip3 install --upgrade pip
# pip3 install -e "$workdir/llm-foundry/.[distributed]" --no-cache-dir
# pip3 install flash_attn==2.8.2 --no-build-isolation --no-cache-dir

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

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache"
export PYTHONPYCACHEPREFIX="$HF_DATASETS_CACHE/.pycache"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export WANDB_DIR="$HF_DATASETS_CACHE/wandb"
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache/$SLURM_JOB_ID"
export NCCL_TIMEOUT=300
export TORCH_FR_BUFFER_SIZE=1000
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_IB_TIMEOUT=20
export NCCL_IB_RETRY_CNT=7
# export NCCL_DEBUG=INFO # Uncomment for NCCL debugging
MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"        # <-- Get the master node address
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')" # <-- Resolve to IP address
export MASTER_PORT=12340                                                      # <-- Ensure this port is open in your SLURM cluster
export SPECS_FILE="$workdir/fsdp/train_config.yml"                            # <-- Change to your specs file path

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES node(s)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/srun.html
#############################################

srun --cpu-bind=none python3 "$workdir/fsdp/train_fsdp.py" \
    --specs "$SPECS_FILE" \
    --slurm-job-id "$SLURM_JOB_ID" \
    --hardware "a100" 1>>"$out" 2>>"$err"

#############################################
# Cleanup
#############################################

# Remove the triton cache folder at the end.
rm -rf "$TRITON_CACHE_DIR"