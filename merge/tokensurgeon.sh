#!/bin/bash -l
#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=lm_short                # <-- Change to your partition
#SBATCH --job-name=tokensurgeon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=8:00:00
#SBATCH --mem=1900G
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="mlnvme"                       # <-- Change to your filesystem
workspace_name="nanotronics"               # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-tokensurgeon.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-tokensurgeon.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
source $workdir/.modules_intel.sh
# python3 -m venv $workdir/.venv_intel_merge
source $workdir/.venv_intel_merge/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_CPUS_PER_TASK CPUs per task" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Mergekit Installation (if needed)
#############################################
if [ ! -d "mergekit" ]; then
    git clone https://github.com/arcee-ai/mergekit.git
    cd mergekit
    pip3 install -e .
    cd ..
fi

#############################################
# ARGUMENT SETUP
#############################################
base_model="$workdir/.eval_cache/Qwen3-0.6B-Base"                          # <-- Change to the path of the base model
donor_model="$workdir/.eval_cache/Tucano2-0.6B-Base"                       # <-- Change to the path of the donor model
output_model="$workdir/.eval_cache/Tucano2-qwen-0.6B-Base"                 # <-- Change to the desired output path
k=64                                        # <-- Change to the desired sparsity level or number of neighbors
approximation_method="omp"                  # <-- Change to the desired approximation method
device="cpu"                                # <-- Change to the desired device (e.g., "cuda", "cpu")
num_threads=$SLURM_CPUS_PER_TASK            # <-- Number of threads to use in CPU operations

#############################################
# IMPORTANT NOTE:
# For the tokensurgeon tool to work correctly, 
# ensure that the donor model has a `tokenizer_class`
# in the `tokenizer_config.json` file that matches one 
# of the following styles:
#
# - GPT2Tokenizer
# - GPT2TokenizerFast
# - OpenAIGPTTokenizer
# - OpenAIGPTTokenizerFast
# - Qwen2Tokenizer
# - Qwen2TokenizerFast
# - LlamaTokenizer
# - LlamaTokenizerFast
# - T5Tokenizer
# - T5TokenizerFast
# - GemmaTokenizer
# - GemmaTokenizerFast
#
# This is important because that is how the tokensurgeon tool
# identifies the normalization scheme used for the token embeddings.
# If the normalization scheme is not identified correctly, the
# tool will default to a GPT2Tokenizer style normalization, which
# may not be appropriate for all models/tokenizers.
#############################################

#############################################
# Main Job Execution
# Other optional arguments worth considering:
#--prefix-match / --byte-match: Reuse existing embeddings that share a prefix or byte representation with donor tokens.
#--magikarp: Filter out poorly trained tokens using the Magikarp heuristic before approximation.
# More info: https://github.com/arcee-ai/mergekit/blob/main/docs/tokensurgeon.md
#############################################
mergekit-tokensurgeon $base_model $donor_model $output_model \
    --k $k \
    --approximation-method $approximation_method \
    --device $device \
    --num-threads $SLURM_CPUS_PER_TASK 1>>"$out" 2>>"$err"

#############################################
# End of Script
#############################################
echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out"