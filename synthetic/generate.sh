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
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=mlgpu_long             # <-- Change to your partition
#SBATCH --job-name=synthetic-gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a40:8
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
    eval "out$i=\"\$workdir/synth/logs/out$i.\$SLURM_JOB_ID\""
    eval "err$i=\"\$workdir/synth/logs/err$i.\$SLURM_JOB_ID\""
done

#############################################
# Modules & Libraries Setup
#############################################

source $workdir/.modules.sh
# python3 -m venv $workdir/.venv_synth  
source $workdir/.venv_synth/bin/activate

# ===== LLM Foundry Install =====
# pip3 install --upgrade pip
# git clone --depth 1 --branch main https://github.com/Polygl0t/llm-foundry.git
# pip3 install -e "$workdir/llm-foundry/.[synth]" --no-cache-dir

#############################################
# Environment Setup
#############################################

export HF_TOKEN=""                                            # <-- Change to your Hugging Face token       
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"      # <-- Use a unique cache directory for this job
export PYTHONPYCACHEPREFIX="$HF_DATASETS_CACHE/.pycache"      # <-- Use the same cache directory for Python bytecode cache
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"             # <-- Use the same cache directory for Hugging Face Hub
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache"     # <-- Use the same cache directory for Triton cache
export CLEAN_CACHE="1"                                        # <-- Set to "1" to clean cache after job completion
export MODEL_NAME_OR_PATH="Qwen/Qwen3-8B"                     # <-- Change to your model name or path
export DATASET_PATH="PATH/TO/YOUR/DATASET"                    # <-- Change to your dataset path
export TEXT_COLUMN="text"                                     # <-- Change to your dataset text column name
export OUTPUT_DIR="$workdir/outputs"                          # <-- Change to your desired output directory
export SYSTEM="Your system prompt here"                       # <-- Change to your system prompt if needed
export PROMPT_PREFIX="Your prompt prefix here"                # <-- Change to your prompt prefix if needed
export PROMPT_SUFFIX="Your prompt suffix here"                # <-- Change to your prompt suffix if needed
export MAX_LENGTH=4096                                        # <-- Change to your desired maximum generation length
export MAX_CHUNK_SIZE=8192                                    # <-- Change to your desired maximum chunk size for the model
export TEMPERATURE=0.5                                        # <-- Change to your desired sampling temperature
export TOP_K=20                                               # <-- Change to your desired top-k sampling value
export TOP_P=0.8                                              # <-- Change to your desired top-p sampling value
export REPETITION_PENALTY=1.2                                 # <-- Change to your desired repetition penalty
export NUM_RETURN_SEQUENCES=1                                 # <-- Change to your desired number of return sequences
export ENABLE_THINKING="1"                                    # <-- Set to "0" to disable thinking mode

mkdir -p "$OUTPUT_DIR"

if [[ -n "$HF_TOKEN" ]]; then
    # Login to Hugging Face (if needed)
    hf auth login --token "$HF_TOKEN"
fi

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
# Main Job Execution (Parallel Generation)
#############################################

# Build thinking flag
THINKING_FLAG=""
if [ "$ENABLE_THINKING" = "1" ]; then
    THINKING_FLAG="--enable_thinking"
fi

export CUDA_VISIBLE_DEVICES=0
export UCX_NET_DEVICES=mlx5_0:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/00000.jsonl" \
    --text_column "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "00000.jsonl" \
    --max_length "$MAX_LENGTH" \
    --max_chunk_size "$MAX_CHUNK_SIZE" \
    --chunk_once \
    --temperature "$TEMPERATURE" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --system "$SYSTEM" \
    --prompt_prefix "$PROMPT_PREFIX" \
    --prompt_suffix "$PROMPT_SUFFIX" \
    $THINKING_FLAG 1>$out0 2>$err0 &

export CUDA_VISIBLE_DEVICES=1
export UCX_NET_DEVICES=mlx5_1:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/00001.jsonl" \
    --text_column "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "00001.jsonl" \
    --max_length "$MAX_LENGTH" \
    --max_chunk_size "$MAX_CHUNK_SIZE" \
    --chunk_once \
    --temperature "$TEMPERATURE" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --system "$SYSTEM" \
    --prompt_prefix "$PROMPT_PREFIX" \
    --prompt_suffix "$PROMPT_SUFFIX" \
    $THINKING_FLAG 1>$out1 2>$err1 &

export CUDA_VISIBLE_DEVICES=2
export UCX_NET_DEVICES=mlx5_2:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/00002.jsonl" \
    --text_column "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "00002.jsonl" \
    --max_length "$MAX_LENGTH" \
    --max_chunk_size "$MAX_CHUNK_SIZE" \
    --chunk_once \
    --temperature "$TEMPERATURE" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --system "$SYSTEM" \
    --prompt_prefix "$PROMPT_PREFIX" \
    --prompt_suffix "$PROMPT_SUFFIX" \
    $THINKING_FLAG 1>$out2 2>$err2 &

export CUDA_VISIBLE_DEVICES=3
export UCX_NET_DEVICES=mlx5_3:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/00003.jsonl" \
    --text_column "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "00003.jsonl" \
    --max_length "$MAX_LENGTH" \
    --max_chunk_size "$MAX_CHUNK_SIZE" \
    --chunk_once \
    --temperature "$TEMPERATURE" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --system "$SYSTEM" \
    --prompt_prefix "$PROMPT_PREFIX" \
    --prompt_suffix "$PROMPT_SUFFIX" \
    $THINKING_FLAG 1>$out3 2>$err3 &

export CUDA_VISIBLE_DEVICES=4
export UCX_NET_DEVICES=mlx5_4:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/00004.jsonl" \
    --text_column "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "00004.jsonl" \
    --max_length "$MAX_LENGTH" \
    --max_chunk_size "$MAX_CHUNK_SIZE" \
    --chunk_once \
    --temperature "$TEMPERATURE" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --system "$SYSTEM" \
    --prompt_prefix "$PROMPT_PREFIX" \
    --prompt_suffix "$PROMPT_SUFFIX" \
    $THINKING_FLAG 1>$out4 2>$err4 &

export CUDA_VISIBLE_DEVICES=5
export UCX_NET_DEVICES=mlx5_5:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/00005.jsonl" \
    --text_column "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "00005.jsonl" \
    --max_length "$MAX_LENGTH" \
    --max_chunk_size "$MAX_CHUNK_SIZE" \
    --chunk_once \
    --temperature "$TEMPERATURE" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --system "$SYSTEM" \
    --prompt_prefix "$PROMPT_PREFIX" \
    --prompt_suffix "$PROMPT_SUFFIX" \
    $THINKING_FLAG 1>$out5 2>$err5 &

export CUDA_VISIBLE_DEVICES=6
export UCX_NET_DEVICES=mlx5_6:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/00006.jsonl" \
    --text_column "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "00006.jsonl" \
    --max_length "$MAX_LENGTH" \
    --max_chunk_size "$MAX_CHUNK_SIZE" \
    --chunk_once \
    --temperature "$TEMPERATURE" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --system "$SYSTEM" \
    --prompt_prefix "$PROMPT_PREFIX" \
    --prompt_suffix "$PROMPT_SUFFIX" \
    $THINKING_FLAG 1>$out6 2>$err6 &

export CUDA_VISIBLE_DEVICES=7
export UCX_NET_DEVICES=mlx5_7:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/00007.jsonl" \
    --text_column "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "00007.jsonl" \
    --max_length "$MAX_LENGTH" \
    --max_chunk_size "$MAX_CHUNK_SIZE" \
    --chunk_once \
    --temperature "$TEMPERATURE" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --system "$SYSTEM" \
    --prompt_prefix "$PROMPT_PREFIX" \
    --prompt_suffix "$PROMPT_SUFFIX" \
    $THINKING_FLAG 1>$out7 2>$err7 &

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
