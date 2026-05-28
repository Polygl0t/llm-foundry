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
#SBATCH --partition=mlgpu_long             # <-- Change to your partition
#SBATCH --job-name=run-annotator
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
    eval "out$i=\"\$workdir/run_filter/out$i.\$SLURM_JOB_ID\""
    eval "err$i=\"\$workdir/run_filter/err$i.\$SLURM_JOB_ID\""
done

#############################################
# Modules & Libraries Setup
#############################################

source $workdir/.modules.sh
# python3 -m venv "$workdir/.venv_amd"
source "$workdir/.venv_amd/bin/activate"

# ==== For this script, you will also need PyTorch for GPU support ====
# pip3 install torch --no-cache-dir

# ===== LLM Foundry Install =====
# pip3 install --upgrade pip --no-cache-dir
# git clone --depth 1 --branch main https://github.com/Polygl0t/llm-foundry.git
# pip3 install -e "$workdir/llm-foundry/.[data]" --no-cache-dir

#############################################
# Environment Setup
#############################################

export HF_TOKEN="<your-token-here>"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MODEL_NAME="Polygl0t/portuguese-bertimbau-toxicity-classifier"
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export DATASET_PATH="/path/to/dataset"
export OUTPUT_FOLDER="/path/to/output"
export BATCH_SIZE=32
export NUM_PROC=16
export FLOAT_SCORE="instruct_score"
export INT_SCORE="instruct_int_score"
export TEXT_COLUMN="messages"
export MAX_LENGTH=512
export CLEAN_CACHE="1"  # <-- Set to "1" to clean cache after job completion

for i in $(seq 0 $((SLURM_NTASKS_PER_NODE - 1))); do
    eval "out_var=\"\$out$i\""
    eval "err_var=\"\$err$i\""
    echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out_var"
    echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out_var"
    echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out_var"
    echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out_var"
    echo "# [${SLURM_JOB_ID}] GLIBC version: $(ldd --version | head -n1)" >> "$out_var"
    echo "# [${SLURM_JOB_ID}] Working directory: $workdir" >> "$out_var"
    echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version) — $(python3 --version)" >> "$out_var"
done

#############################################
# Main Job Execution (Parallel Classification)
#############################################

export CUDA_VISIBLE_DEVICES=0
export UCX_NET_DEVICES=mlx5_0:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/llm-foundry/data/filters/run_annotator.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/train-00000-of-00007.jsonl" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --token $HF_TOKEN \
    --batch_size $BATCH_SIZE \
    --output_folder "$OUTPUT_FOLDER" \
    --num_proc $NUM_PROC \
    --float_score $FLOAT_SCORE \
    --int_score $INT_SCORE \
    --max_length $MAX_LENGTH 1>$out0 2>$err0 &

export CUDA_VISIBLE_DEVICES=1
export UCX_NET_DEVICES=mlx5_1:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/llm-foundry/data/filters/run_annotator.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/train-00001-of-00007.jsonl" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --token $HF_TOKEN \
    --batch_size $BATCH_SIZE \
    --output_folder "$OUTPUT_FOLDER" \
    --num_proc $NUM_PROC \
    --float_score $FLOAT_SCORE \
    --int_score $INT_SCORE \
    --max_length $MAX_LENGTH 1>$out1 2>$err1 &

export CUDA_VISIBLE_DEVICES=2
export UCX_NET_DEVICES=mlx5_2:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/llm-foundry/data/filters/run_annotator.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/train-00002-of-00007.jsonl" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --token $HF_TOKEN \
    --batch_size $BATCH_SIZE \
    --output_folder "$OUTPUT_FOLDER" \
    --num_proc $NUM_PROC \
    --float_score $FLOAT_SCORE \
    --int_score $INT_SCORE \
    --max_length $MAX_LENGTH 1>$out2 2>$err2 &

export CUDA_VISIBLE_DEVICES=3
export UCX_NET_DEVICES=mlx5_3:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/llm-foundry/data/filters/run_annotator.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/train-00003-of-00007.jsonl" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --token $HF_TOKEN \
    --batch_size $BATCH_SIZE \
    --output_folder "$OUTPUT_FOLDER" \
    --num_proc $NUM_PROC \
    --float_score $FLOAT_SCORE \
    --int_score $INT_SCORE \
    --max_length $MAX_LENGTH 1>$out3 2>$err3 &

export CUDA_VISIBLE_DEVICES=4
export UCX_NET_DEVICES=mlx5_4:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/llm-foundry/data/filters/run_annotator.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/train-00004-of-00007.jsonl" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --token $HF_TOKEN \
    --batch_size $BATCH_SIZE \
    --output_folder "$OUTPUT_FOLDER" \
    --num_proc $NUM_PROC \
    --float_score $FLOAT_SCORE \
    --int_score $INT_SCORE \
    --max_length $MAX_LENGTH 1>$out4 2>$err4 &

export CUDA_VISIBLE_DEVICES=5
export UCX_NET_DEVICES=mlx5_5:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/llm-foundry/data/filters/run_annotator.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/train-00005-of-00007.jsonl" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --token $HF_TOKEN \
    --batch_size $BATCH_SIZE \
    --output_folder "$OUTPUT_FOLDER" \
    --num_proc $NUM_PROC \
    --float_score $FLOAT_SCORE \
    --int_score $INT_SCORE \
    --max_length $MAX_LENGTH 1>$out5 2>$err5 &

export CUDA_VISIBLE_DEVICES=6
export UCX_NET_DEVICES=mlx5_6:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/llm-foundry/data/filters/run_annotator.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/train-00006-of-00007.jsonl" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --token $HF_TOKEN \
    --batch_size $BATCH_SIZE \
    --output_folder "$OUTPUT_FOLDER" \
    --num_proc $NUM_PROC \
    --float_score $FLOAT_SCORE \
    --int_score $INT_SCORE \
    --max_length $MAX_LENGTH 1>$out6 2>$err6 &

export CUDA_VISIBLE_DEVICES=7
export UCX_NET_DEVICES=mlx5_7:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/llm-foundry/data/filters/run_annotator.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/train-00007-of-00007.jsonl" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --token $HF_TOKEN \
    --batch_size $BATCH_SIZE \
    --output_folder "$OUTPUT_FOLDER" \
    --num_proc $NUM_PROC \
    --float_score $FLOAT_SCORE \
    --int_score $INT_SCORE \
    --max_length $MAX_LENGTH 1>$out7 2>$err7 &

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

