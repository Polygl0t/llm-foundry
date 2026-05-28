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
#SBATCH --partition=A40short               # <-- Change to your partition
#SBATCH --job-name=env-doctor
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --gpus=1

#############################################
# Working Directory Setup
#############################################

workdir="/home/nklugeco"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/env-doctor-out.$SLURM_JOB_ID"
exec > "$out" 2>&1

#############################################
# Modules & Libraries Setup
#############################################

source "$workdir/.modules.sh"
source "$workdir/.venv_intel/bin/activate" # <-- Activate the same environment you want to diagnose with env-doctor

# ===== Install env-doctor =====
# See https://github.com/mitulgarg/env-doctor
pip3 install env-doctor --no-cache-dir

#############################################
# Environment Setup
#############################################

export CUDA_VISIBLE_DEVICES=0

echo "# [${SLURM_JOB_ID}] Job started at: $(date)"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES node(s)"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')"
echo "# [${SLURM_JOB_ID}] GLIBC version: $(ldd --version | head -n1)"
echo "# [${SLURM_JOB_ID}] Working directory: $workdir"
echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version) — $(python3 --version)"

#############################################
# Main Job Execution
#############################################
# Diagnose the GPU/CUDA/PyTorch stack on this node.
#
# Useful commands:
#   env-doctor check              — full GPU driver -> CUDA -> library diagnosis
#   env-doctor check --json       — machine-readable output for CI
#   env-doctor python-compat      — Python version conflicts with AI libs
#   env-doctor cuda-info          — detailed CUDA toolkit analysis
#   env-doctor install <lib>      — safe install command for your driver
#   env-doctor model <name>       — VRAM fit check for a model
#############################################

srun --cpu-bind=none env-doctor check

#############################################
# End of Script
#############################################
echo "# [${SLURM_JOB_ID}] Job finished at: $(date)"
