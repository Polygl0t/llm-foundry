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
#SBATCH --job-name=dpo
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

out="$workdir/run_outputs/out-dpo-trainer.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-dpo-trainer.$SLURM_JOB_ID"

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
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$HF_DATASETS_CACHE/inductor_cache"
export TRACKIO_STORAGE_MODE=sqlite
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

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
GPUS_PER_NODE="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
NUM_MACHINES="${SLURM_NNODES:-1}"
NUM_PROCESSES=$(( NUM_MACHINES * GPUS_PER_NODE ))
export GPUS_PER_NODE NUM_MACHINES NUM_PROCESSES

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

# Threads per GPU worker: this node's CPU allocation divided by its GPUs,
# because that single Slurm task runs all four accelerate workers.
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE ))

# ---- Distributed backend -----------------------------------------------------
# "ddp"  = one full model replica per GPU (accelerate .ddp_config.yaml)
# "fsdp" = FSDP2: parameters, gradients and optimizer state are sharded across every
#          rank of the job (accelerate .fsdp_config.yaml). The policy AND the reference
#          model are both sharded.
#
#   BACKEND=fsdp sbatch alignment/slurm/dpo_trainer.sh
export BACKEND="${BACKEND:-ddp}"
case "$BACKEND" in
    ddp)
        BACKEND_SUFFIX=""
        _default_accelerate_config="$workdir/llm-foundry/alignment/configs/.ddp_config.yaml"
        ;;
    fsdp)
        BACKEND_SUFFIX="-fsdp"
        _default_accelerate_config="$workdir/llm-foundry/alignment/configs/.fsdp_config.yaml"
        ;;
    *)
        echo "FATAL: BACKEND must be 'ddp' or 'fsdp' (got '$BACKEND')" >&2
        exit 1
        ;;
esac
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$_default_accelerate_config}"

# ---- Paths -------------------------------------------------------------------
export TRAIN_DATASET_DIR="/data/harmfull-no-reasoning"
export MODEL_PATH="Polygl0t/Tucano2-qwen-0.5B-Instruct"
export REF_MODEL_PATH="$MODEL_PATH"
export CHAT_TEMPLATE="checkpoints/portuguese/tokenizer/chat_template.jinja"
export CHECKPOINT_DIR="./checkpoints/MyModel-DPO-$SLURM_JOB_ID${BACKEND_SUFFIX}"
export RESUME_FROM=""   # Path to resume checkpoint from, leave empty to start fresh
export CLEAN_CACHE="1"  # <-- Set to "1" to clean cache after job completion

# ---- DPO hyper-parameters ----------------------------------------------------
export DATASET_TYPE="jsonl"             # "jsonl" or "parquet"
export MAX_LENGTH="4096"                # prompt+chosen/rejected budget; the collator pads to the longest sequence IN THE BATCH, so this bounds peak VRAM
export NUM_PROC="${NUM_PROC:-$SLURM_CPUS_PER_TASK}"
export SHUFFLE_DATASET="1"
export TRUNCATION_MODE="keep_start"     # TRL's forward-compatible value; "keep_end" is deprecated
export BETA="0.1"                       # KL-style deviation from the reference model
export LOSS_TYPE="apo_zero"             # "apo_zero", "sigmoid", "ipo", ... (trainer asserts)
export PRECOMPUTE_REF_LOG_PROBS="1"     # "0" = re-run the reference model every step. Dropped automatically when the Liger kernel is selected.
export USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-1}"   # fused DPO loss (DDP only; forced to "0" under BACKEND=fsdp, or when the router aux loss is enabled)
export GRADIENT_CHECKPOINTING="0"       # trade compute for VRAM; the pre-flight below reads this
export ATTN_IMPLEMENTATION="flash_attention_4"
export PER_DEVICE_TRAIN_BATCH_SIZE="8"
export GRADIENT_ACCUMULATION_STEPS="4"
export LEARNING_RATE="0.000005"
export WEIGHT_DECAY="0.0"
export LR_SCHEDULER_TYPE="cosine"
export WARMUP_STEPS="0.1"               # float in [0,1) = RATIO of total steps; int = exact count
export NUM_TRAIN_EPOCHS="5"
export MAX_STEPS=""                     # non-empty overrides NUM_TRAIN_EPOCHS
export SAVE_STEPS="1000"
export LOGGING_STEPS="1"
export BF16="1"
export TF32="1"

# ---- MoE router policy -------------------------------------------------------
# Load-balancing auxiliary loss for Mixture-of-Experts models (no effect on dense
# models).
export ROUTER_AUX_LOSS_COEF="${ROUTER_AUX_LOSS_COEF:-0.0}"
export FREEZE_ROUTER="${FREEZE_ROUTER:-0}"

_router_aux_enabled() {
    awk -v v="${1:-0}" 'BEGIN { exit !(v + 0 != 0) }'
}

# Liger + FSDP2 does not work for the DPO fused loss.
if [ "$BACKEND" = "fsdp" ] && [ "$USE_LIGER_KERNEL" = "1" ]; then
    echo "# [${SLURM_JOB_ID}] FSDP: forcing USE_LIGER_KERNEL=0 (the Liger DPO loss is not DTensor-safe)." >> "$out"
    USE_LIGER_KERNEL="0"
fi
# TRL raises when the MoE aux loss is enabled with the fused Liger DPO loss.
if _router_aux_enabled "$ROUTER_AUX_LOSS_COEF" && [ "$USE_LIGER_KERNEL" = "1" ]; then
    echo "# [${SLURM_JOB_ID}] ROUTER_AUX_LOSS_COEF=$ROUTER_AUX_LOSS_COEF: forcing USE_LIGER_KERNEL=0 (TRL refuses the MoE aux loss with the fused DPO loss)." >> "$out"
    USE_LIGER_KERNEL="0"
fi
# TRL also refuses precomputed reference log-probs with the fused Liger loss.
if [ "$USE_LIGER_KERNEL" = "1" ] && [ "$PRECOMPUTE_REF_LOG_PROBS" = "1" ]; then
    echo "# [${SLURM_JOB_ID}] Liger: dropping --precompute_ref_log_probs (the fused DPO loss never materialises the router logits)." >> "$out"
    PRECOMPUTE_REF_LOG_PROBS="0"
fi

# ---- FSDP2 pre-flight --------------------------------------------------------
# Rejects accelerate configs whose export would be empty or whose knobs collide with
# what the trainer passes, BEFORE the multi-node rendezvous is set up.
if [ "$BACKEND" = "fsdp" ]; then
    echo "# [${SLURM_JOB_ID}] FSDP pre-flight: $ACCELERATE_CONFIG" >> "$out"
    # `0` = this template does NOT pass --gradient_checkpointing (see ARGS below), so
    # fsdp_activation_checkpointing would not collide with anything.
    python3 - "$ACCELERATE_CONFIG" "$GRADIENT_CHECKPOINTING" dpo >> "$out" 2>&1 <<'PYEOF'
import sys

try:
    import yaml
except ModuleNotFoundError:
    sys.exit(
        f"ERROR: PyYAML is not importable, cannot validate '{sys.argv[1]}'. It ships with "
        "accelerate, so this is probably not the venv that holds the trainers."
    )

config_path, gradient_checkpointing, trainer = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
config = yaml.safe_load(open(config_path)) or {}
fsdp = config.get("fsdp_config") or {}
problems, warnings = [], []

if str(config.get("distributed_type", "")).upper() != "FSDP":
    problems.append(
        f"distributed_type is '{config.get('distributed_type')}', not 'FSDP': "
        "ACCELERATE_CONFIG does not point at an FSDP config."
    )

version = fsdp.get("fsdp_version")
if version != 2:
    warnings.append(
        f"fsdp_version={version} (not 2): FSDP1 shards differently from what this script assumes."
    )

state_dict_type = str(fsdp.get("fsdp_state_dict_type", "SHARDED_STATE_DICT")).upper()
if state_dict_type != "FULL_STATE_DICT":
    problems.append(
        f"fsdp_state_dict_type='{state_dict_type}': Trainer.save_model() only writes weights under "
        "FULL_STATE_DICT, so <checkpoint_dir>/final would end up empty."
    )

if gradient_checkpointing and fsdp.get("fsdp_activation_checkpointing"):
    problems.append(
        "fsdp_activation_checkpointing=true while the trainer passes --gradient_checkpointing: "
        "the HF Trainer refuses that pair. Keep fsdp_activation_checkpointing=false."
    )

if trainer == "dpo" and fsdp.get("fsdp_cpu_ram_efficient_loading"):
    problems.append(
        "fsdp_cpu_ram_efficient_loading=true with the DPO trainer: the reference model is sharded "
        "by TRL's prepare_fsdp, which never broadcasts rank 0's weights, so every other rank would "
        "compute the reference log-probs from uninitialised parameters."
    )

if not fsdp.get("fsdp_auto_wrap_policy"):
    warnings.append(
        "fsdp_auto_wrap_policy is unset: only the root module is sharded, i.e. one all-gather of "
        "the whole model per step instead of one per decoder layer."
    )

for warning in warnings:
    print(f"WARNING: {warning}")
for problem in problems:
    print(f"ERROR: {problem}")
if problems:
    sys.exit(1)
print(f"OK: FSDP{version} with state_dict_type={state_dict_type}, "
      f"auto_wrap_policy={fsdp.get('fsdp_auto_wrap_policy')}, "
      f"cpu_ram_efficient_loading={fsdp.get('fsdp_cpu_ram_efficient_loading')}")
PYEOF
    if [ "$?" -ne 0 ]; then
        echo "# [${SLURM_JOB_ID}] FSDP pre-flight FAILED -- not launching (see above)." >> "$out"
        exit 1
    fi
fi

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
echo "# [${SLURM_JOB_ID}] Backend: $BACKEND (accelerate config: $ACCELERATE_CONFIG)" >> "$out"
echo "# [${SLURM_JOB_ID}] DPO: loss=$LOSS_TYPE beta=$BETA lr=$LEARNING_RATE epochs=$NUM_TRAIN_EPOCHS${MAX_STEPS:+ (max_steps=$MAX_STEPS)}" >> "$out"
echo "# [${SLURM_JOB_ID}] Liger kernel: $USE_LIGER_KERNEL (precompute_ref_log_probs=$PRECOMPUTE_REF_LOG_PROBS)" >> "$out"
echo "# [${SLURM_JOB_ID}] Router policy: aux_coef=$ROUTER_AUX_LOSS_COEF freeze_router=$FREEZE_ROUTER" >> "$out"
echo "# [${SLURM_JOB_ID}] Checkpoints: $CHECKPOINT_DIR" >> "$out"

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
# ACCELERATE_CONFIG (DDP or FSDP2) is resolved in the backend block above from $BACKEND.

# NOTE: `--machine_rank` MUST be expanded by each task. `\$SLURM_NODEID` stays
# literal in this string and is resolved inside the `srun ... bash -c "$CMD"` below, so
# every node gets its own rank.
export LAUNCHER="accelerate launch \
--config_file $ACCELERATE_CONFIG \
--num_machines $NUM_MACHINES \
--num_processes $NUM_PROCESSES \
--machine_rank \$SLURM_NODEID \
--main_process_ip $MASTER_ADDR \
--main_process_port $MASTER_PORT \
--rdzv_backend static \
--rdzv_conf timeout=300"

export PYTHON_FILE="$workdir/llm-foundry/alignment/dpo_trainer.py"

export ARGS="--dataset_type $DATASET_TYPE \
--train_dataset_dir $TRAIN_DATASET_DIR \
--cache_dir $HF_DATASETS_CACHE \
--num_proc $NUM_PROC \
--model_name_or_path $MODEL_PATH \
--ref_model_name_or_path $REF_MODEL_PATH \
--chat_template_path $CHAT_TEMPLATE \
--checkpoint_dir $CHECKPOINT_DIR \
--hub_token $HF_TOKEN \
--max_length $MAX_LENGTH \
--truncation_mode $TRUNCATION_MODE \
--beta $BETA \
--loss_type $LOSS_TYPE \
--learning_rate $LEARNING_RATE \
--weight_decay $WEIGHT_DECAY \
--lr_scheduler_type $LR_SCHEDULER_TYPE \
--warmup_steps $WARMUP_STEPS \
--num_train_epochs $NUM_TRAIN_EPOCHS \
--save_steps $SAVE_STEPS \
--logging_steps $LOGGING_STEPS \
--attn_implementation $ATTN_IMPLEMENTATION \
--router_aux_loss_coef $ROUTER_AUX_LOSS_COEF \
--per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
"

if [ -n "$MAX_STEPS" ]; then
    ARGS="$ARGS --max_steps $MAX_STEPS"
fi
if [ "$SHUFFLE_DATASET" = "1" ]; then
    ARGS="$ARGS --shuffle_dataset"
fi
if [ "$GRADIENT_CHECKPOINTING" = "1" ]; then
    ARGS="$ARGS --gradient_checkpointing"
fi
if [ "$PRECOMPUTE_REF_LOG_PROBS" = "1" ]; then
    ARGS="$ARGS --precompute_ref_log_probs"
fi
if [ "$USE_LIGER_KERNEL" = "1" ]; then
    ARGS="$ARGS --use_liger_kernel"
fi
if [ "$FREEZE_ROUTER" = "1" ]; then
    ARGS="$ARGS --freeze_router"
fi
if [ "$BF16" = "1" ]; then
    ARGS="$ARGS --bf16"
fi
if [ "$TF32" = "1" ]; then
    ARGS="$ARGS --tf32"
fi

if [ -n "$RESUME_FROM" ] && [ ! -e "$RESUME_FROM" ]; then
    echo "# [${SLURM_JOB_ID}] RESUME_FROM=$RESUME_FROM does not exist yet -- training from MODEL_PATH" >> "$out"
elif [ -n "$RESUME_FROM" ]; then
    ARGS="$ARGS --resume_from_checkpoint $RESUME_FROM"
fi

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $ARGS"

# ONE launcher per node (the job allocates exactly one Slurm task per node), each with
# its own --machine_rank; accelerate forks the per-GPU workers inside it.
srun --nodes="$NUM_MACHINES" --ntasks="$NUM_MACHINES" --cpu-bind=none \
    bash -c "$CMD" 1>>"$out" 2>>"$err"

#############################################
# Post-processing
#############################################

if [ -f "$out" ]; then
    echo "# [${SLURM_JOB_ID}] Splitting trainer metrics into $CHECKPOINT_DIR/{training,validation}.jsonl" >> "$out"
    python3 - "$out" "$CHECKPOINT_DIR" >> "$out" 2>&1 <<'PYEOF'
import ast
import json
import os
import re
import sys

NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")

def metric_rows(path):
    for line in open(path, errors="replace"):
        start, end = line.find("{"), line.rfind("}")
        if start < 0 or end < start:
            continue
        try:
            row = ast.literal_eval(line[start : end + 1])
        except (ValueError, SyntaxError):
            continue
        if isinstance(row, dict) and row:
            yield row

def coerce(value):
    if isinstance(value, str) and NUMBER.match(value):
        try:
            return int(value)
        except ValueError:
            return float(value)
    return value

def write_split(data, name, outdir):
    path = os.path.join(outdir, name)
    seen = set()
    if os.path.exists(path):
        seen = {line.strip() for line in open(path, errors="replace") if line.strip()}
    rows = [json.dumps(row, ensure_ascii=False) for row in data]
    new = [line for line in rows if line not in seen]
    if new:
        os.makedirs(outdir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(new) + "\n")
    print(f"  {name:<18} {len(rows):>6} rows in this log | {len(new):>6} new | "
          f"{len(rows) - len(new):>6} already present | {len(seen) + len(new):>6} total")


log, outdir = sys.argv[1], sys.argv[2]
buckets = {"training": [], "validation": []}
for metric_row in metric_rows(log):
    kind = "validation" if any(str(key).startswith("eval_") for key in metric_row) else "training"
    buckets[kind].append({key: coerce(value) for key, value in metric_row.items()})

print(f"parsed {log}: {len(buckets['training'])} training rows, "
      f"{len(buckets['validation'])} validation rows -> {outdir}")
for kind, filename in (("training", "training.jsonl"), ("validation", "validation.jsonl")):
    write_split(buckets[kind], filename, outdir)
PYEOF
    if [ "$?" -ne 0 ]; then
        echo "# [${SLURM_JOB_ID}] WARNING: log split failed, see above" >> "$out"
    fi
fi

#############################################
# Cleanup
#############################################

if [ -d "$CHECKPOINT_DIR" ]; then
    cp "$out" "$CHECKPOINT_DIR/logs.txt" 2>/dev/null || true
    cp "${BASH_SOURCE[0]}" "$CHECKPOINT_DIR/job.sh" 2>/dev/null || true
fi

clean_dir() {
    [ -d "$1" ] || return 0
    find "$1" -mindepth 1 -delete 2>/dev/null || true
}

if [ "$CLEAN_CACHE" = "1" ]; then
    if [ -n "$SLURM_JOB_ID" ]; then
        echo "# [${SLURM_JOB_ID}] Cleaning caches: $HF_DATASETS_CACHE" >> "$out"
        clean_dir "$HF_DATASETS_CACHE"
    else
        # Without a job id the path collapses to the shared $workdir/.cache.
        echo "# [${SLURM_JOB_ID}] WARNING: CLEAN_CACHE=1 but SLURM_JOB_ID is unset -- skipping" >> "$out"
    fi
fi
clean_dir "$TRITON_CACHE_DIR"
clean_dir "$TORCHINDUCTOR_CACHE_DIR"

for split in train validation; do
    [ -d "$PREBUILT_DIR/$split" ] && \
        find "$PREBUILT_DIR/$split" -maxdepth 1 -name 'cache*' -delete 2>/dev/null
done

echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out"
echo "# [${SLURM_JOB_ID}] Logs: $out (stdout) / $err (stderr)" >> "$out"
