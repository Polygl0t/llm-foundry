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
#SBATCH --partition=lm_short                # <-- Change to your partition
#SBATCH --job-name=model_merge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=8:00:00
#SBATCH --mem=1900G
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################

# Set this to your workspace root (where you have the .venv and .modules.sh files).
workdir="/lustre/mlnvme/data/polyglot"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out_merge.$SLURM_JOB_ID"
err="$workdir/run_outputs/err_merge.$SLURM_JOB_ID"

#############################################
# Modules & Libraries Setup
#############################################

source $workdir/.modules.sh > "$out" 2>&1
# python3 -m venv $workdir/.venv_intel_merge
source $workdir/.venv_intel_merge/bin/activate

# ===== Upgrade PIP =====
# pip3 install --upgrade pip

# ===== Install Mergekit =====

# git clone https://github.com/arcee-ai/mergekit.git
# pip3 install -e "$workdir/mergekit" --no-cache-dir

#############################################
# Environment Setup
#############################################

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_CPUS_PER_TASK CPUs per task" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# [${SLURM_JOB_ID}] GLIBC version: $(ldd --version | head -n1)" >> "$out"
echo "# [${SLURM_JOB_ID}] Working directory: $workdir" >> "$out"
echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version)" >> "$out"

#############################################
# Basic Mergekit Usage Information
#############################################
# Mergekit Model Merging Script
# Basic usage: mergekit-yaml <config_file> <output_dir> [options]

# SAFETY OPTIONS:
# --allow-crimes          - Allow mixing different model architectures (use with caution)
# --trust-remote-code     - Trust remote code from HuggingFace repos (security risk)

# STORAGE OPTIONS:
# --transformers-cache    - Override default cache path for downloaded models
# --lora-merge-cache      - Specify cache path for merged LoRA models

# PERFORMANCE OPTIONS (for large models):
# --cuda                  - Use GPU for matrix operations (much faster)
# --low-cpu-memory        - Store results on GPU/accelerator (useful when VRAM > RAM)
# --read-to-gpu           - Read model weights directly to GPU
# --multi-gpu             - Use multiple GPUs for parallel processing
# --gpu-rich              - Shortcut for all GPU optimizations
# --lazy-unpickle         - Experimental feature for lower memory usage
# -j, --num-threads       - Set number of CPU threads (default used below)

# OUTPUT OPTIONS:
# --out-shard-size        - Parameters per output shard (default: 5B)
# --copy-tokenizer        - Include tokenizer in output (recommended)
# --safe-serialization    - Save as safetensors format (recommended for safety)
# --write-model-card      - Generate README.md with merge details
# --clone-tensors         - Allow same layer multiple times in merge

# MISC OPTIONS:
# --lora-merge-dtype      - Override data type for LoRA merging
# --random-seed           - Set seed for reproducible results
# -v                      - Verbose logging (use -vv for more detail)
# --quiet                 - Suppress progress bars

# CONFIG PARAMETER HANDLING:
# - Use 'base_model' in YAML to specify which model's config to inherit
# - The base_model's config (max_position_embeddings, vocab_size, etc.) will be used
# - You can override specific parameters in the 'parameters' section of the YAML
# - Always choose the model with the largest/most compatible config as base_model

#############################################
# ARGUMENT SETUP
#############################################
output_model="./merged-model"               # <-- Change to the desired output path
config_file="$workdir/mergekit_config.yml"  # <-- Change to the path of your mergekit YAML config file
num_threads=32                              # <-- Number of threads to use in CPU operations

#############################################
# Main Job Execution
#############################################
mergekit-yaml $config_file $output_model --num-threads $num_threads 1>>"$out" 2>>"$err"

#############################################
# End of Script
#############################################
echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out"
