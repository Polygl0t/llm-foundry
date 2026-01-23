#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
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
username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="mlnvme"                       # <-- Change to your filesystem
workspace_name="nanotronics"               # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/synth/synth_logs"
cd "$workdir"
ulimit -c 0

for i in $(seq 0 $((SLURM_NTASKS_PER_NODE - 1))); do
    eval "out$i=\"\$workdir/synth/synth_logs/out$i.\$SLURM_JOB_ID\""
    eval "err$i=\"\$workdir/synth/synth_logs/err$i.\$SLURM_JOB_ID\""
done

#############################################
# Environment Setup
#############################################
source $workdir/.modules_amd.sh                       # <-- Load necessary modules
# python3 -m venv $workdir/.venv_amd                  # <-- Create virtual environment
source $workdir/.venv_amd/bin/activate                # <-- Activate virtual environment
# pip3 install --upgrade pip --no-cache-dir
# pip3 install torch==2.8.0 --no-cache-dir
# pip3 install torchaudio==2.8.0 --no-cache-dir
# pip3 install torchvision==0.23.0 --no-cache-dir
# pip3 install transformers --no-cache-dir
# pip3 install vllm --no-cache-dir

export HF_TOKEN=""  # <-- Change to your Hugging Face token
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion
export MODEL_NAME_OR_PATH="Qwen/Qwen2.5-7B-Instruct" # <-- Change to your model name or path
export DATASET_PATH="PATH/TO/YOUR/DATASET"  # <-- Change to your dataset path
export TEXT_COLUMN="text"  # <-- Change to your dataset text column name
export OUTPUT_DIR="$workdir/outputs" # <-- Change to your desired output directory
export SYSTEM="Your system prompt here"  # <-- Change to your system prompt if needed
export PROMPT_PREFIX="Your prompt prefix here"  # <-- Change to your prompt prefix if needed
export PROMPT_SUFFIX="Your prompt suffix here"  # <-- Change to your prompt suffix if needed
export MAX_LENGTH=4096
export MAX_CHUNK_SIZE=5000
export TEMPERATURE=0.2
export TOP_K=50
export TOP_P=0.9
export REPETITION_PENALTY=1.2
export NUM_RETURN_SEQUENCES=1
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
    echo "# Working directory: $workdir" >> "$out_var"
    echo "# Python executable: $(which python3)" >> "$out_var"
done

#############################################
# Main Job Execution (Parallel Generation)
#############################################
export CUDA_VISIBLE_DEVICES=0
export UCX_NET_DEVICES=mlx5_0:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate_base.py \
    --model_name "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/chunk_0.jsonl" \
    --column_name "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "chunk_0.jsonl" \
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
    --prompt_suffix "$PROMPT_SUFFIX" 1>$out0 2>$err0 &

export CUDA_VISIBLE_DEVICES=1
export UCX_NET_DEVICES=mlx5_1:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate_base.py \
    --model_name "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/chunk_1.jsonl" \
    --column_name "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "chunk_1.jsonl" \
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
    --prompt_suffix "$PROMPT_SUFFIX" 1>$out1 2>$err1 &

export CUDA_VISIBLE_DEVICES=2
export UCX_NET_DEVICES=mlx5_2:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate_base.py \
    --model_name "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/chunk_2.jsonl" \
    --column_name "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "chunk_2.jsonl" \
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
    --prompt_suffix "$PROMPT_SUFFIX" 1>$out2 2>$err2 &

export CUDA_VISIBLE_DEVICES=3
export UCX_NET_DEVICES=mlx5_3:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate_base.py \
    --model_name "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/chunk_3.jsonl" \
    --column_name "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "chunk_3.jsonl" \
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
    --prompt_suffix "$PROMPT_SUFFIX" 1>$out3 2>$err3 &

export CUDA_VISIBLE_DEVICES=4
export UCX_NET_DEVICES=mlx5_4:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate_base.py \
    --model_name "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/chunk_4.jsonl" \
    --column_name "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "chunk_4.jsonl" \
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
    --prompt_suffix "$PROMPT_SUFFIX" 1>$out4 2>$err4 &

export CUDA_VISIBLE_DEVICES=5
export UCX_NET_DEVICES=mlx5_5:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate_base.py \
    --model_name "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/chunk_5.jsonl" \
    --column_name "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "chunk_5.jsonl" \
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
    --prompt_suffix "$PROMPT_SUFFIX" 1>$out5 2>$err5 &

export CUDA_VISIBLE_DEVICES=6
export UCX_NET_DEVICES=mlx5_6:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate_base.py \
    --model_name "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/chunk_6.jsonl" \
    --column_name "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "chunk_6.jsonl" \
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
    --prompt_suffix "$PROMPT_SUFFIX" 1>$out6 2>$err6 &

export CUDA_VISIBLE_DEVICES=7
export UCX_NET_DEVICES=mlx5_7:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/generate_base.py \
    --model_name "$MODEL_NAME_OR_PATH" \
    --dataset_path "$DATASET_PATH/chunk_7.jsonl" \
    --column_name "$TEXT_COLUMN" \
    --output_dir $OUTPUT_DIR \
    --output_file "chunk_7.jsonl" \
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
    --prompt_suffix "$PROMPT_SUFFIX" 1>$out7 2>$err7 &

wait

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

for i in $(seq 0 $((SLURM_NTASKS_PER_NODE - 1))); do
    eval "out_var=\"\$out$i\""
    eval "err_var=\"\$err$i\""
    echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out_var"
done
