#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_bit_flek              # <-- Change to your SLURM account
#SBATCH --partition=sgpu_medium            # <-- Change to your partition
#SBATCH --job-name=grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
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

out="$workdir/run_outputs/out-grpo-trainer.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-grpo-trainer.$SLURM_JOB_ID"

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
source "$workdir/.modules.sh"
# python3 -m venv $workdir/.venv_trl
source "$workdir/.venv_trl/bin/activate"

# pip3 install --upgrade pip
# git clone --depth 1 --branch main https://github.com/Polygl0t/llm-foundry.git
# pip3 install -e "$workdir/llm-foundry/.[trl]" --no-cache-dir

# ===== ALL HAIL FLASH-ATTN! =====
# Using the pre-built flash-attn wheel for CUDA 12.6 and PyTorch 2.8 with CXX11 ABI set to TRUE, which is compatible with our environment.
# If you have a different setup, please build flash-attn from source or find the appropriate wheel for your configuration.
# pip3 install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl --no-cache-dir

# ===== OPTIONAL: vLLM-Powered Generation =====
# Generation is often the main bottleneck for online methods like GRPO.
# Install TRL with vLLM support to speed up rollouts with --use_vllm.
# pip3 install 'trl[vllm]' --no-cache-dir

# ===== OPTIONAL: Specialized Attention Packages =====
# Uncomment only if your model uses the corresponding attention type.
# pip3 install mamba-ssm[causal-conv1d] --no-build-isolation --no-cache-dir
# pip3 install flash-linear-attention --no-cache-dir

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
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
#export NCCL_DEBUG=INFO # Uncomment for NCCL debugging
export GPUS_PER_NODE=$SLURM_NTASKS_PER_NODE
export NUM_MACHINES=$SLURM_NNODES
export head_node_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export CHECKPOINT_DIR="./checkpoints/MyModel-GRPO-$SLURM_JOB_ID"
export CLEAN_CACHE="1"  # <-- Set to "1" to clean cache after job completion
export VLLM_PORT="8000"
export VLLM_BIND_HOST="0.0.0.0"
export VLLM_STARTUP_WAIT="60"  # Seconds to wait for vLLM to load the model before training starts

hf auth login --token "$HF_TOKEN"
wandb login "$WANDB_TOKEN"

echo "# [${SLURM_JOB_ID}] Job started on $SLURM_JOB_NODELIST at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution (Distributed Training + vLLM)
#############################################
# Single-node default: GPU 0 hosts vLLM, GPUs 1..N train GRPO.
# Multi-node: the last allocated node hosts vLLM, previous nodes train GRPO.
#############################################
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
last_node_index=$((${#NODELIST[@]} - 1))
vllm_node="${NODELIST[$last_node_index]}"
train_nodes=("${NODELIST[@]:0:$last_node_index}")

if [ "$SLURM_NNODES" -eq 1 ]; then
    train_nodes=("${NODELIST[0]}")
    vllm_node="${NODELIST[0]}"
    vllm_cuda_visible_devices=0
    vllm_tensor_parallel_size=1
    trainer_gpu_ids=$(seq -s, 1 $(($SLURM_NTASKS_PER_NODE - 1)))
    train_num_machines=1
    train_processes_per_node=$(($SLURM_NTASKS_PER_NODE - 1))
else
    vllm_cuda_visible_devices=$(seq -s, 0 $(($SLURM_NTASKS_PER_NODE - 1)))
    vllm_tensor_parallel_size=$SLURM_NTASKS_PER_NODE
    vllm_gpu_memory_utilization=0.9
    trainer_gpu_ids="all"
    train_num_machines=$(($SLURM_NNODES - 1))
    train_processes_per_node=$SLURM_NTASKS_PER_NODE
fi

export VLLM_HOST="$vllm_node"

if [ "$train_processes_per_node" -lt 1 ]; then
    echo "# [${SLURM_JOB_ID}] GRPO requires at least 2 GPUs for single-node vLLM server mode." >> "$err"
    exit 1
fi

train_num_processes=$(($train_num_machines * $train_processes_per_node))
train_node_list=$(IFS=, ; echo "${train_nodes[*]}")
export TRAIN_MAIN_PROCESS_IP="${train_nodes[0]}"
export TRAIN_NUM_PROCESSES="$train_num_processes"
export TRAIN_NUM_MACHINES="$train_num_machines"
export TRAINER_GPU_IDS="$trainer_gpu_ids"
export out
export err
export workdir

export PYTHON_FILE="$workdir/llm-foundry/alignment/grpo_trainer.py"

export ARGS="--dataset_type jsonl \
--train_dataset_dir $workdir/llm-foundry/alignment/0.jsonl \
--shuffle_dataset \
--cache_dir $HF_DATASETS_CACHE \
--num_proc $SLURM_CPUS_PER_TASK \
--model_name_or_path Polygl0t/Tucano2-qwen-0.5B-Instruct \
--chat_template_path /assets/chat_template.jinja \
--checkpoint_dir $CHECKPOINT_DIR \
--hub_token $HF_TOKEN \
--max_prompt_length 2048 \
--max_completion_length 1024 \
--num_generations 8 \
--num_iterations 1 \
--beta 0.0 \
--loss_type dapo \
--scale_rewards true \
--save_steps 1000 \
--logging_steps 1 \
--learning_rate 0.000001 \
--weight_decay 0.0 \
--lr_scheduler_type cosine \
--warmup_ratio 0.1 \
--num_train_epochs 1 \
--attn_implementation flash_attention_2 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--use_vllm \
--vllm_mode server \
--vllm_server_host $VLLM_HOST \
--vllm_server_port $VLLM_PORT \
--bf16 \
--tf32 \
--gradient_checkpointing \
"

srun --nodes=1 --ntasks=1 --nodelist="$vllm_node" \
    env CUDA_VISIBLE_DEVICES="$vllm_cuda_visible_devices" \
    trl vllm-serve \
    --model Polygl0t/Tucano2-qwen-0.5B-Instruct \
    --host "$VLLM_BIND_HOST" \
    --port "$VLLM_PORT" \
    --tensor_parallel_size "$vllm_tensor_parallel_size" \
    --vllm_gpu_memory_utilization "$vllm_gpu_memory_utilization" \  
    1>>"$out" 2>>"$err" &
vllm_pid=$!

echo "# [${SLURM_JOB_ID}] Started vLLM server on $vllm_node using $vllm_tensor_parallel_size GPU(s)" >> "$out"
echo "# [${SLURM_JOB_ID}] Waiting ${VLLM_STARTUP_WAIT}s for vLLM startup" >> "$out"
sleep "$VLLM_STARTUP_WAIT"

srun --nodes="$train_num_machines" --ntasks="$train_num_machines" --nodelist="$train_node_list" \
    bash -lc 'accelerate launch \
    --config_file "$workdir/llm-foundry/alignment/.ddp_config.yaml" \
    --num_processes "$TRAIN_NUM_PROCESSES" \
    --num_machines "$TRAIN_NUM_MACHINES" \
    --machine_rank "$SLURM_PROCID" \
    --main_process_ip "$TRAIN_MAIN_PROCESS_IP" \
    --gpu_ids "$TRAINER_GPU_IDS" \
    "$PYTHON_FILE" $ARGS \
    1>>"$out" 2>>"$err"'
train_exit=$?

kill "$vllm_pid" 2>/dev/null || true
wait "$vllm_pid" 2>/dev/null || true

if [ "$train_exit" -ne 0 ]; then
    echo "# [${SLURM_JOB_ID}] GRPO training failed with exit code $train_exit" >> "$err"
    exit "$train_exit"
fi

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