#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=mlgpu_long             # <-- Change to your partition
#SBATCH --job-name=run-classifier
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
username="nklugeco_hpc"                         # <-- Change to the corresponding username that created the workspace
file_system="mlnvme"                            # <-- Change to your filesystem
workspace_name="nanotronics"                    # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_filter"
cd "$workdir"
ulimit -c 0

for i in $(seq 0 $((SLURM_NTASKS_PER_NODE - 1))); do
    eval "out$i=\"\$workdir/run_filter/out$i.\$SLURM_JOB_ID\""
    eval "err$i=\"\$workdir/run_filter/err$i.\$SLURM_JOB_ID\""
done

#############################################
# Environment Setup
#############################################
source "$workdir/.modules_amd.sh"
source "$workdir/.venv_amd/bin/activate"

export HF_TOKEN="<your-token-here>" # <-- Change to your HF token
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MODEL_NAME="Polygl0t/portuguese-bertimbau-toxicity-classifier"
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion
export DATASET_PATH="/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft"
export OUTPUT_FOLDER="/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/gigaverbo-v2-sft-processed"
export BATCH_SIZE=32
export NUM_PROC=16
export FLOAT_SCORE="instruct_score"
export INT_SCORE="instruct_int_score"
export TEXT_COLUMN="messages"
export MAX_LENGTH=6032

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
# Main Job Execution (Parallel Classification)
#############################################

export CUDA_VISIBLE_DEVICES=0
export UCX_NET_DEVICES=mlx5_0:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/run_classifier.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/gigaverbo-v2-aes-enem.jsonl" \
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
python3 $workdir/run_classifier.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/gigaverbo-v2-auggsm8k.jsonl" \
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
python3 $workdir/run_classifier.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/gigaverbo-v2-code-parrot.jsonl" \
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
python3 $workdir/run_classifier.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/gigaverbo-v2-extract-personas.jsonl" \
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
python3 $workdir/run_classifier.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/gigaverbo-v2-extract-summaries.jsonl" \
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
python3 $workdir/run_classifier.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/gigaverbo-v2-healthcare.jsonl" \
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
python3 $workdir/run_classifier.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/gigaverbo-v2-generate-personas.jsonl" \
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
python3 $workdir/run_classifier.py \
    --model_name "$MODEL_NAME" \
    --apply_chat_template \
    --text_column "$TEXT_COLUMN" \
    --dataset_path "$DATASET_PATH/gigaverbo-v2-gsm8k.jsonl" \
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

