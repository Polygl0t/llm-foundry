#!/bin/bash
#
# To lunch this bash script, you need to submit a HTCondor job with the following JDL:
#   >> condor_submit job.jdl
#
# Where in job.jdl, you set the executable to this script.
#
# Workflow:
#   1. Load CUDA/driver modules via .modules.sh
#   2. Extract the pre-built Python venv from CephFS -> /jwd
#   3. Set environment variables (CUDA, NCCL, HuggingFace cache, etc.)
#   4. Launch training script
#   5. Clean up /jwd after training finishes
#
# All persistent outputs (logs, checkpoints) are written directly to CephFS, the
# filesystem is shared between container and login node.

#############################################
# Working Directory Setup
#############################################

# ${BUDDY} is set by HTCondor to your CephFS home (e.g., /cephfs/user/sfatimah).
workdir="${BUDDY}"
venv_name=".venv"
venv_path="$workdir/$venv_name.tar.gz"
mkdir -p "$workdir/logs"
cd "$workdir"

# CLUSTER_ID: unique job identifier passed from the JDL via arguments = $(ClusterId).
# Used to name log files so multiple runs don't overwrite each other.
CLUSTER_ID="${1:-$$}"
out="$workdir/logs/out.${CLUSTER_ID}"
err="$workdir/logs/err.${CLUSTER_ID}"

#############################################
# Modules & Libraries Setup
#############################################

source "$workdir/llm-foundry/.modules.sh" > "$out" 2>&1

# --- Venv Extraction ---
# The venv tarball lives on CephFS and is extracted to /jwd
# The tarball is created once with:  bash create_venv_training.sh.
# First login on an interactive node and run create_venv_training.sh to create the tarball,
# then all subsequent training jobs can use it.
cd /jwd
tar xf "$venv_path" 2>/dev/null || {
    echo "# ERROR: venv tarball not found" >> "$out"
    echo "# Run the create_venv.sh script first (see the llm-foundry)" >> "$out"
    exit 1
}
source /jwd/$venv_name/bin/activate

#############################################
# 3. Environment Setup
#############################################

export SPECS_FILE="$workdir/specifications.yaml"
export OMP_NUM_THREADS=24
export HF_DATASETS_CACHE="$workdir/.cache"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export WANDB_DIR="$HF_DATASETS_CACHE/wandb"
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache/${CLUSTER_ID}"
export NCCL_TIMEOUT=300
export TORCH_FR_BUFFER_SIZE=1000
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_IB_TIMEOUT=20
export NCCL_IB_RETRY_CNT=7

mkdir -p "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$WANDB_DIR"

# Single-node — MASTER_ADDR is always localhost
export MASTER_ADDR="localhost"
export MASTER_PORT=12340 # Ensure this port is open in your cluster. If you ever hit a port conflict, just change it to another value

echo "# [${CLUSTER_ID}] Job started at: $(date)" >> "$out"
echo "# [${CLUSTER_ID}] Hostname: $(hostname)" >> "$out"
echo "# [${CLUSTER_ID}] GLIBC version: $(ldd --version | head -n1)" >> "$out"
echo "# [${CLUSTER_ID}] MASTER_ADDR: $MASTER_ADDR ($MASTER_PORT)" >> "$out"
echo "# [${CLUSTER_ID}] Working directory: $workdir" >> "$out"
echo "# [${CLUSTER_ID}] Python executable: $(which python3) — $(python3 --version)" >> "$out"
echo "# [${CLUSTER_ID}] CUDA_HOME: ${CUDA_HOME:-not set}" >> "$out"

#############################################
# Main Training Execution
#############################################

torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "$workdir/llm-foundry/distributed/train_ddp.py" \
    --specs "$SPECS_FILE" \
    --slurm-job-id "${CLUSTER_ID}" \
    --hardware h200 \
    1>>"$out" 2>>"$err"

# $? captures the exit code of torchrun (the last command run).
# 0 = success, non-zero = failure (OOM, NCCL error, assertion, etc.).
# This is later returned to HTCondor so it knows if the job succeeded.
TRAIN_EXIT_CODE=$?

echo "# [${CLUSTER_ID}] Training finished at $(date) with exit code: ${TRAIN_EXIT_CODE}" >> "$out"

#############################################
# Cleanup
#############################################

# Remove the triton cache folder and venv at the end.
rm -rf "$TRITON_CACHE_DIR" "/jwd/venv_ddp" "$HF_DATASETS_CACHE"

deactivate 2>/dev/null || true

exit ${TRAIN_EXIT_CODE}
