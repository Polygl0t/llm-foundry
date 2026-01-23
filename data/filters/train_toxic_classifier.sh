#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=sgpu_long              # <-- Change to your partition
#SBATCH --job-name=train-toxicity-classifier-pt
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
username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="mlnvme"                       # <-- Change to your filesystem
workspace_name="nanotronics"               # <-- Change to your workspace/project name

workdir="$workdir"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

for i in $(seq 0 $((SLURM_NTASKS_PER_NODE - 1))); do
    eval "out$i=\"\$workdir/run_outputs/out$i.\$SLURM_JOB_ID\""
    eval "err$i=\"\$workdir/run_outputs/err$i.\$SLURM_JOB_ID\""
done

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
source "$workdir/.modules_amd.sh"
source "$workdir/.venv_amd/bin/activate"

export HF_TOKEN="<your-token-here>" # <-- Change to your HF token
export WANDB_TOKEN="<your-token-here>" # <-- Change to your wandb token
export ID_LABEL="Toxic-Score"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion

hf auth login --token "$HF_TOKEN"
wandb login "$WANDB_TOKEN"

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
# Main Job Execution (Parallel Training)
#############################################
export CUDA_VISIBLE_DEVICES=0
export UCX_NET_DEVICES=mlx5_0:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/train_classifier.py \
    --dataset_path "./gigaverbo-v2-dedup/toxic_classification_dataset" \
    --cache_dir "$workdir/.cache" \
    --model_name "neuralmind/bert-base-portuguese-cased" \
    --checkpoint_dir "$workdir/checkpoints/bertimbau-toxic" \
    --freeze \
    --hub_token "$HF_TOKEN" \
    --bf16 \
    --tf32 \
    --id_label $ID_LABEL 1>$out0 2>$err0 &

export CUDA_VISIBLE_DEVICES=1
export UCX_NET_DEVICES=mlx5_1:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/train_classifier.py \
    --dataset_path "./gigaverbo-v2-dedup/toxic_classification_dataset" \
    --cache_dir "$workdir/.cache" \
    --model_name "neuralmind/bert-large-portuguese-cased" \
    --checkpoint_dir "$workdir/checkpoints/bertimbau-large-toxic" \
    --freeze \
    --hub_token "$HF_TOKEN" \
    --bf16 \
    --tf32 \
    --id_label $ID_LABEL 1>$out1 2>$err1 &

export CUDA_VISIBLE_DEVICES=2
export UCX_NET_DEVICES=mlx5_3:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/train_classifier.py \
    --dataset_path "./gigaverbo-v2-dedup/toxic_classification_dataset" \
    --cache_dir "$workdir/.cache" \
    --model_name "pablocosta/bertabaporu-base-uncased" \
    --checkpoint_dir "$workdir/checkpoints/bertabaporu-toxic" \
    --freeze \
    --hub_token "$HF_TOKEN" \
    --bf16 \
    --tf32 \
    --id_label $ID_LABEL 1>$out2 2>$err2 &

export CUDA_VISIBLE_DEVICES=3
export UCX_NET_DEVICES=mlx5_2:1
srun -n 1 -N 1 --gpus=1 --exclusive \
python3 $workdir/train_classifier.py \
    --dataset_path "./gigaverbo-v2-dedup/toxic_classification_dataset" \
    --cache_dir "$workdir/.cache" \
    --model_name "pablocosta/bertabaporu-large-uncased" \
    --checkpoint_dir "$workdir/checkpoints/bertabaporu-large-toxic" \
    --freeze \
    --hub_token "$HF_TOKEN" \
    --bf16 \
    --tf32 \
    --id_label $ID_LABEL 1>$out3 2>$err3 &

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