#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn about SLURM sbatch options at:
# - https://slurm.schedmd.com/sbatch.html
#
# Learn about JSC JUPITER at:
# - https://www.fz-juelich.de/en/ias/jsc/systems/supercomputers/jupiter
#############################################
# MULTI-NODE: set `--nodes=N` below and submit the SAME script -- nothing else needs
# editing. `srun` starts ONE `accelerate launch` per node, each with its own
# --machine_rank, and accelerate forks the per-GPU workers on every node
# (--num_processes = N x 4). Requirements: all nodes must share the filesystem that
# holds HF_DATASETS_CACHE, the checkpoint dir, and any other shared resources.
# MASTER_ADDR/MASTER_PORT are derived below from SLURM_NODELIST / SLURM_JOB_ID.
#############################################
#SBATCH --account=ag_bit_flek              # <-- Change to your SLURM account
#SBATCH --partition=sgpu_medium            # <-- Change to your partition
#SBATCH --job-name=sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=1-00:00:00
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
source $workdir/.venv_trl/bin/activate

# ===== Installation =====
# See alignment/slurm/create_venv_marvin.sh for the installation of the venv and packages.

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

# OMP_NUM_THREADS is derived in the distributed-topology block below: it must be the
# per-node CPU count divided by the per-node GPU count, because ONE Slurm task runs all
# the GPU workers.
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export HF_TOKEN="<your-token-here>"
export WANDB_TOKEN="<your-token-here>"
export WANDB_DIR="$HF_DATASETS_CACHE/wandb"
export TRACKIO_STORAGE_MODE=sqlite
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

# ---- Distributed topology (single- AND multi-node) ---------------------------------#
# accelerate needs the topology EXPLICITLY: the .yaml config holds single-node
# defaults (`num_machines: 1`, `machine_rank: 0`), so a bare `accelerate launch` runs
# a single-node job on the FIRST node even when --nodes > 1. These values are handed
# to `accelerate launch` in the job-execution block below.
#
# `gpu_ids: all` in the .yaml means "every visible GPU", so the per-node count comes
# from CUDA_VISIBLE_DEVICES -- Slurm sets it from --gres, and it is what accelerate
# will use. It is deliberately NOT --ntasks-per-node: the launcher is ONE task per
# node, and accelerate forks the per-GPU workers itself.
# -----------------------------------------------------------------------------------#

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
GPUS_PER_NODE="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
NUM_MACHINES="${SLURM_NNODES:-1}"
NUM_PROCESSES=$(( NUM_MACHINES * GPUS_PER_NODE ))
MACHINE_RANK="${SLURM_NODEID:-0}"
export GPUS_PER_NODE NUM_MACHINES NUM_PROCESSES MACHINE_RANK

# MASTER_ADDR = the first allocated node. Slurm sometimes returns a short hostname
# that the compute nodes cannot resolve; append the DNS domain when it does not.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_ADDR="${MASTER_ADDR:-$(hostname -f)}"   # scontrol unavailable / returned nothing
if ! getent hosts "$MASTER_ADDR" >/dev/null 2>&1 && [[ "$MASTER_ADDR" != *.* ]]; then
    DOMAIN="$(hostname -d)"
    [[ -n "$DOMAIN" ]] && MASTER_ADDR="${MASTER_ADDR}.${DOMAIN}"
fi
export MASTER_ADDR

# Derive MASTER_PORT from SLURM_JOB_ID so concurrent jobs cannot collide on the
# default 29500. Kept inside the dynamic/private range (49152-65535).
MASTER_PORT=$(( 49152 + (SLURM_JOB_ID % 16384) ))
export MASTER_PORT

# Threads per GPU worker: this node's CPU allocation divided by its GPUs (288/4 = 72),
# because that single Slurm task runs all four accelerate workers.
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))
export CHECKPOINT_DIR="./checkpoints/MyModel-SFT-$SLURM_JOB_ID"
export CLEAN_CACHE="1"  # <-- Set to "1" to clean cache after job completion

# In places like JUPITER compute nodes (jpbo-*), we have NO internet access. Log in only when the
# Hub is actually reachable; on an offline node every artifact must be a local
# path. Note the trainer only pushes to the Hub when BOTH --hub_token and
# --hub_model_id are set, so with --hub_model_id unset this job never needs the
# network at all.
if curl -sSf --max-time 10 https://huggingface.co >/dev/null 2>&1; then
    hf auth login --token "$HF_TOKEN"
    wandb login "$WANDB_TOKEN"
else
    echo "# [${SLURM_JOB_ID}] No internet on this node: skipping hub/wandb login." >> "$out"
fi

echo "# [${SLURM_JOB_ID}] Job started on $SLURM_JOB_NODELIST at: $(date)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $NUM_PROCESSES processes in total ($GPUS_PER_NODE per node on $NUM_MACHINES node(s))" >> "$out"
echo "# [${SLURM_JOB_ID}] MASTER_ADDR: $MASTER_ADDR:$MASTER_PORT ; machine_rank: $MACHINE_RANK" >> "$out"
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

# Multi-node: change `--nodes=1` in the SBATCH header to the number of nodes you
# want. `srun` then starts ONE `accelerate launch` per node, each with its own
# --machine_rank, and accelerate forks the per-GPU workers on every node.
#
#   --num_machines / --machine_rank  the shape of the job, and this node's rank
#   --num_processes                  TOTAL processes (machines x GPUs per machine)
#   --main_process_ip/port           node 0's address, the rendezvous point
#   --rdzv_backend static            matches `rdzv_backend: static` in the config
#
# Set ACCELERATE_CONFIG=.fsdp_config.yaml to switch to FSDP.
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$workdir/llm-foundry/alignment/configs/.ddp_config.yaml}"

export LAUNCHER="accelerate launch \
--config_file $ACCELERATE_CONFIG \
--num_machines $NUM_MACHINES \
--num_processes $NUM_PROCESSES \
--machine_rank $MACHINE_RANK \
--main_process_ip $MASTER_ADDR \
--main_process_port $MASTER_PORT \
--rdzv_backend static"

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
--chat_template_path checkpoints/portuguese/chat_template.jinja \
--checkpoint_dir $CHECKPOINT_DIR \
--hub_token $HF_TOKEN \
--save_test_set \
--max_length 4096 \
--save_steps 2000 \
--save_total_limit 4 \
--logging_steps 1 \
--packing \
--assistant_only_loss \
--use_liger_kernel \
--learning_rate 0.00005 \
--weight_decay 0.1 \
--lr_scheduler_type cosine \
--warmup_steps 0.1 \
--num_train_epochs 1 \
--attn_implementation flash_attention_4 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 2 \
--bf16 \
--tf32 \
--gradient_checkpointing \
"

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $ARGS"

# ONE launcher per node (the job allocates exactly one Slurm task per node), each with
# its own --machine_rank; accelerate forks the per-GPU workers inside it.
srun --nodes="$NUM_MACHINES" --ntasks="$NUM_MACHINES" --cpu-bind=none \
    bash -c "$CMD" 1>>"$out" 2>>"$err"

#############################################
# Cleanup
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
