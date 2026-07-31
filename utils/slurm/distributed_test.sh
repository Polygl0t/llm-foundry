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
#SBATCH --account=polyglot                 # <-- Change to your SLURM account
#SBATCH --partition=booster                # <-- Change to your partition
#SBATCH --job-name=dist-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=72
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:4
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################

# Set this to your workspace root (where you have the .venv and .modules.sh files).
workdir="/e/project1/polyglot/COMMON"
mkdir -p "$workdir/logs"
cd "$workdir"
ulimit -c 0

out="$workdir/logs/dist-test-out.$SLURM_JOB_ID"
err="$workdir/logs/dist-test-err.$SLURM_JOB_ID"

#############################################
# Modules & Libraries Setup
#############################################

source $workdir/jupiter_modules_2026.sh > "$out" 2>&1
source $workdir/.venv_core_2026/bin/activate

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
#
# ── GPU Communication Cheat Sheet ─────────────────────────────────────
# GPUs within a node can communicate in three ways (fastest wins by default):
#
#   1. CPU / shared memory (slowest)
#      > NCCL_P2P_DISABLE=1   +   FI_PROVIDER=tcp
#      > NCCL INFO ... : 1[1] -> 0[0] via SHM/direct/direct
#
#   2. GPUDirect RDMA over EFA NICs (meh)
#      > NCCL_P2P_DISABLE=1   +   FI_PROVIDER=efa
#      > NCCL INFO ... : 1[1] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA/Shared
#
#   3. GPUDirect RDMA via NVLink (brrrrrrrr!)
#      > NCCL auto-prioritises NVLink — no explicit env vars needed
#      > NCCL INFO ... : 0[0] -> 1[1] via P2P/CUMEM
#
#   To identify which route NCCL selected, run with NCCL_DEBUG=INFO and
#   look for "via ..." in the log output.
# ──────────────────────────────────────────────────────────────────────
#############################################

export CUDA_VISIBLE_DEVICES=0,1,2,3                   # <-- Specify visible GPUs per node
export TORCH_HOME="$workdir/.cache/torch"             # <-- Set the cache directory for PyTorch
export NCCL_TIMEOUT=60                                # <-- How long (in seconds) to wait for an NCCL operation to complete before timing out
export TORCH_FR_BUFFER_SIZE=1000                      # <-- Set the torch c10d backend buffer size in MB
export CUDA_LAUNCH_BLOCKING=0                         # <-- Set to 1 for CUDA synchronous behavior (useful for debugging)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1              # <-- Enable asynchronous error handling for NCCL operations
export TORCH_DISTRIBUTED_DEBUG=OFF                    # <-- Use DETAIL to enable detailed debugging for PyTorch distributed operations
export NCCL_IB_TIMEOUT=20                             # <-- Timeout for InfiniBand operations in seconds
export NCCL_IB_RETRY_CNT=7                            # <-- Number of retries for InfiniBand operations
# export NCCL_DEBUG=INFO                              # <-- Enable NCCL debugging
# export NCCL_P2P_DISABLE=1                           # <-- Disable peer-to-peer (NVLink) communication (uncomment if needed)
# export FI_PROVIDER=tcp                              # <-- Uncomment to use TCP for libfabric. "efa" for GPUDirect RDMA via EFA

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
srun --cpu-bind=none python3 "$workdir/llm-foundry/utils/distributed_test.py" \
    --n-warmup 5 --n-iter 20 --dtype bfloat16 1>>"$out" 2>>"$err"

#############################################
# End of Script
#############################################
echo "# [${SLURM_JOB_ID}] Job completed successfully" >> "$out"
echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out"
