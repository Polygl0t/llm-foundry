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
#SBATCH --account=ag_cst_gabriel            # <-- Change to your SLURM account
#SBATCH --partition=lm_short                # <-- Change to your partition
#SBATCH --job-name=tok-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=8:00:00
#SBATCH --mem=500G
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################

# Set this to your workspace root (where you have the .venv and .modules.sh files).
workdir="/lustre/mlnvme/data/polyglot"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-tok-eval.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-tok-eval.$SLURM_JOB_ID"

#############################################
# Modules & Libraries Setup
#############################################

source $workdir/.modules.sh > "$out" 2>&1
# python3 -m venv $workdir/.venv_intel  
source $workdir/.venv_intel/bin/activate

# ===== LLM Foundry Install =====
# pip3 install --upgrade pip
# git clone --depth 1 --branch main https://github.com/Polygl0t/llm-foundry.git
# pip3 install -e "$workdir/llm-foundry/.[tokenizer]" --no-cache-dir

#############################################
# Environment Setup
#############################################

export HF_TOKEN=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_CPUS_PER_TASK CPUs per task" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# [${SLURM_JOB_ID}] GLIBC version: $(ldd --version | head -n1)" >> "$out"
echo "# [${SLURM_JOB_ID}] Working directory: $workdir" >> "$out"
echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version) — $(python3 --version)" >> "$out"

#############################################
# Main Job Execution
#############################################

python3 $workdir/llm-foundry/tokenizer/tokenizer_eval.py \
    --tokenizers_to_evaluate "ibm-granite/granite-3.3-2b-base" \
    "meta-llama/Llama-3.2-1B" \
    "Qwen/Qwen2.5-0.5B" \
    "allenai/OLMo-2-0425-1B" \
    "HuggingFaceTB/SmolLM3-3B-Base" \
    --input_file "$workdir/sample.txt" \
    --output_file "$workdir/results.json" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --token "$HF_TOKEN" 1>>"$out" 2>>"$err"

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
