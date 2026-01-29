#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel            # <-- Change to your SLURM account
#SBATCH --partition=sgpu_short             # <-- Change to your partition
#SBATCH --job-name=alpaca-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive
#SBATCH --dependency=afterany:22556319      # <-- Change dependency as needed

#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="mlnvme"                       # <-- Change to your filesystem
workspace_name="polyglot"               # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-alpaca.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-alpaca.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
source "$workdir/.modules_amd.sh"
source "$workdir/.venv_amd/bin/activate"
# pip3 install alpaca-eval --no-cache-dir

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion
export OPENAI_API_KEY="<your-token-here>"
export MODEL_ID="/lustre/scratch/data/nklugeco_hpc-polyglot_datasets/portuguese/checkpoints/models/Tucano2-qwen-0.5B-Instruct"
export OUTPUT_FOLDER="$workdir/dpo/.evals"
export MODEL_NAME=$(basename "$MODEL_ID")
export REFERENCE_FILE="$OUTPUT_FOLDER/references.json"
export MODEL_OUTPUT_FILE="$OUTPUT_FOLDER/${MODEL_NAME}.json"
export EVAL_OUTPUT_PATH="$OUTPUT_FOLDER/results_${MODEL_NAME}"

mkdir -p "$OUTPUT_FOLDER"
mkdir -p "$EVAL_OUTPUT_PATH"                   # <-- Set your OpenAI API key here

echo "# [${SLURM_JOB_ID}] Job started on $SLURM_JOB_NODELIST at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution
#############################################
echo "# =============================================" >> "$out"
echo "# STEP 1: Generate reference outputs" >> "$out"
echo "# =============================================" >> "$out"

# Generate references.json if it doesn't exist
if [ ! -f "$REFERENCE_FILE" ]; then
    echo "# Generating references.json from dataset..." >> "$out"
    python3 -c "
from datasets import load_dataset
import json

ds = load_dataset('TucanoBR/alpaca-eval-pt', cache_dir='$HF_DATASETS_CACHE')['eval']
data = [sample for sample in ds]

with open('$REFERENCE_FILE', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    
print(f'References saved to: $REFERENCE_FILE')
print(f'Total samples: {len(data)}')
" 1>>$out 2>>$err
else
    echo "# References file already exists: $REFERENCE_FILE" >> "$out"
fi

echo "" >> "$out"
echo "# =============================================" >> "$out"
echo "# STEP 2: Generate model outputs" >> "$out"
echo "# =============================================" >> "$out"

export CUDA_VISIBLE_DEVICES=0
python3 $workdir/alpaca_eval_generate.py \
    --model_id "$MODEL_ID" \
    --cache_dir "$HF_DATASETS_CACHE" \
    --attn_implementation "flash_attention_2" \
    --precision "bfloat16" \
    --max_new_tokens 2048 \
    --max_input_length 4096 \
    --do_sample \
    --temperature 0.1 \
    --repetition_penalty 1.2 \
    --system_prompt "Você é um assistente útil e educado que sempre tenta ajudar os usuários da melhor forma possível." \
    --output_folder "$OUTPUT_FOLDER" \
    --batch_size 8 1>>$out 2>>$err

if [ $? -ne 0 ]; then
    echo "# ERROR: Model generation failed!" >> "$err"
    exit 1
fi

echo "" >> "$out"
echo "# =============================================" >> "$out"
echo "# STEP 3: Run AlpacaEval evaluation" >> "$out"
echo "# =============================================" >> "$out"

# Check if model output file exists
if [ ! -f "$MODEL_OUTPUT_FILE" ]; then
    echo "# ERROR: Model output file not found: $MODEL_OUTPUT_FILE" >> "$err"
    exit 1
fi

# Run alpaca_eval
echo "# Running alpaca_eval with GPT-4 turbo as judge..." >> "$out"
echo "# Model outputs: $MODEL_OUTPUT_FILE" >> "$out"
echo "# Reference outputs: $REFERENCE_FILE" >> "$out"
echo "# Output path: $EVAL_OUTPUT_PATH" >> "$out"
echo "# WARNING: This will use OpenAI API and incur costs (~$6)" >> "$out"

alpaca_eval \
    --model_outputs "$MODEL_OUTPUT_FILE" \
    --reference_outputs "$REFERENCE_FILE" \
    --output_path "$EVAL_OUTPUT_PATH" 1>>$out 2>>$err

if [ $? -eq 0 ]; then
    echo "" >> "$out"
    echo "# =============================================" >> "$out"
    echo "# EVALUATION COMPLETE!" >> "$out"
    echo "# =============================================" >> "$out"
    echo "# Results saved to: $EVAL_OUTPUT_PATH" >> "$out"
    
    # Display the leaderboard if it exists
    if [ -f "$EVAL_OUTPUT_PATH/leaderboard.csv" ]; then
        echo "" >> "$out"
        echo "# Leaderboard Results:" >> "$out"
        cat "$EVAL_OUTPUT_PATH/leaderboard.csv" >> "$out"
    fi
else
    echo "# ERROR: AlpacaEval failed! Check that OPENAI_API_KEY is set correctly." >> "$err"
    exit 1
fi

#############################################
# End of Script
#############################################
# Clean cache folder if requested
if [ "$CLEAN_CACHE" = "1" ]; then
    echo "# [${SLURM_JOB_ID}] Cleaning HF_DATASETS_CACHE" >> "$out"
    if [ -d "$HF_DATASETS_CACHE" ]; then
        find "$HF_DATASETS_CACHE" -mindepth 1 -delete 2>/dev/null || true
    fi
else
    echo "# [${SLURM_JOB_ID}] Skipping cache cleanup (CLEAN_CACHE=$CLEAN_CACHE)" >> "$out"
fi

echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out"
