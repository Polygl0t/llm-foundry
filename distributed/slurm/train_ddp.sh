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
#SBATCH --job-name=ddp-training
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

out="$workdir/run_outputs/ddp-out.$SLURM_JOB_ID"
err="$workdir/run_outputs/ddp-err.$SLURM_JOB_ID"

#############################################
# Modules & Libraries Setup
#############################################

source $workdir/.modules.sh > "$out" 2>&1
source $workdir/.venv_distributed/bin/activate

# ===== Installation =====
# See distributed/slurm/create_venv_marvin.sh for the installation of the venv and packages.

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

export SPECS_FILE="$workdir/distributed/specifications.yaml"                  # <-- Change to your specs file path
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache"
export PYTHONPYCACHEPREFIX="$HF_DATASETS_CACHE/.pycache"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export WANDB_DIR="$HF_DATASETS_CACHE/wandb"
export TRACKIO_STORAGE_MODE=sqlite
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache/$SLURM_JOB_ID"
export NCCL_TIMEOUT=300
export TORCH_FR_BUFFER_SIZE=1000
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_IB_TIMEOUT=20
export NCCL_IB_RETRY_CNT=7
# export NCCL_DEBUG=INFO # Uncomment for NCCL debugging

# Slurm gives us the first allocated node name. On Marvin this is usually already
# resolvable as returned, while on Bender Slurm may return a short hostname such
# as "node-03". If the short name does not resolve, append the local DNS domain
# so torch.distributed gets a valid MASTER_ADDR on both clusters.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"        # <-- Get the master node hostname
if ! getent hosts "$MASTER_ADDR" >/dev/null 2>&1 && [[ "$MASTER_ADDR" != *.* ]]; then
    MASTER_ADDR="${MASTER_ADDR}.$(hostname -d)"
fi
export MASTER_ADDR
export MASTER_PORT=12340                                                      # <-- Ensure this port is open in your SLURM cluster

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES node(s)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# [${SLURM_JOB_ID}] GLIBC version: $(ldd --version | head -n1)" >> "$out"
echo "# [${SLURM_JOB_ID}] MASTER_ADDR: $MASTER_ADDR ($MASTER_PORT)" >> "$out"
echo "# [${SLURM_JOB_ID}] Working directory: $workdir" >> "$out"
echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version)" >> "$out"

#############################################
# Main Job Execution
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/srun.html
#############################################

srun --cpu-bind=none python3 "$workdir/llm-foundry/distributed/train_ddp.py" \
    --specs "$SPECS_FILE" \
    --slurm-job-id "$SLURM_JOB_ID" \
    --hardware "a100" 1>>"$out" 2>>"$err"

#############################################
# Cleanup
#############################################

# Remove the triton cache folder at the end.
rm -rf "$TRITON_CACHE_DIR"
