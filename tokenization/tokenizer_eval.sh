#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
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
username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="scratch"                      # <-- Change to your filesystem
workspace_name="polyglot_datasets"        # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-tok-eval.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-tok-eval.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
source $workdir/.modules_intel.sh
source $workdir/.venv_intel/bin/activate

export HF_TOKEN="<your-token-here>" # <-- Change to your Hugging Face token
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_CPUS_PER_TASK CPUs per task" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution
#############################################
python3 $workdir/tokenizer_eval.py \
    --tokenizers_to_evaluate "TucanoBR/Tucano-1b1" \
    "ClassiCC-Corpus/Curio-1.1b" \
    "ibm-granite/granite-3.3-2b-base" \
    "meta-llama/Llama-3.2-1B" \
    "Qwen/Qwen2.5-0.5B" \
    "allenai/OLMo-2-0425-1B" \
    "NOVA-vision-language/GlorIA-1.3B" \
    "PORTULAN/gervasio-7b-portuguese-ptbr-decoder" \
    "neuralmind/bert-base-portuguese-cased" \
    "pablocosta/bertabaporu-base-uncased" \
    "sagui-nlp/debertinha-ptbr-xsmall" \
    "PORTULAN/albertina-100m-portuguese-ptbr-encoder" \
    "unicamp-dl/ptt5-base-portuguese-vocab" \
    "eduagarcia/RoBERTaCrawlPT-base" \
    "eduagarcia/RoBERTaLexPT-base" \
    "HuggingFaceTB/SmolLM3-3B-Base" \
    --input_file "$workdir/portuguese/checkpoints/tokenizers/sample.txt" \
    --output_file "$workdir/portuguese/checkpoints/tokenizers/eval-pt.json" \
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
