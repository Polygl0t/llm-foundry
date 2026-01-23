#!/bin/bash -l
#############################################
# LM Evaluation Harness - Portuguese Language Models
# 
# This script automates the evaluation of Portuguese language models using the lm-evaluation-harness framework.
# It can evaluate either local checkpoints or HuggingFace models in parallel across GPUs.
# Results are post-processed to YAML format for easy analysis.
#
# Learn more about SLURM options at: https://slurm.schedmd.com/sbatch.html
# Learn more about lm-evaluation-harness at: https://github.com/EleutherAI/lm-evaluation-harness
#############################################

#############################################
# SLURM Job Configuration
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=mlgpu_short            # <-- Change to your partition
#SBATCH --job-name=eval-harness-portuguese # <-- Name of the job (appears in squeue)
#SBATCH --nodes=1                          # <-- Number of compute nodes (keep at 1 for this script)
#SBATCH --ntasks-per-node=1                # <-- Number of tasks per node
#SBATCH --threads-per-core=1               # <-- Threads per CPU core
#SBATCH --cpus-per-task=16                 # <-- CPUs per task
#SBATCH --time=08:00:00                    # <-- Time limit (days-hrs:min:sec)
#SBATCH --gres=gpu:a40:1                   # <-- Request 1 GPU initially (script will use up to 8 based on items to evaluate)
#SBATCH --exclusive                        # <-- Request exclusive node access (recommended for performance)
#SBATCH --dependency=afterany:24006883     # <-- OPTIONAL: Wait for another job to finish (change job ID or comment out) 
#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                    # <-- Change to your HPC username
file_system="mlnvme"                       # <-- Change to your filesystem (mlnvme, scratch, etc.)
workspace_name="nanotronics"               # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"  # <-- Constructs the full workspace path
mkdir -p "$workdir"                        # <-- Create workspace directory if it doesn't exist
cd "$workdir"                              # <-- Change to workspace directory
ulimit -c 0                                # <-- Disable core dumps (saves disk space)

#############################################
# Environment Setup
#############################################
source $workdir/.modules_amd.sh            # <-- Load required modules (Python, CUDA, etc.)
# python3 -m venv $workdir/.venv_eval_pt   # <-- UNCOMMENT on first run to create virtual environment
source $workdir/.venv_eval_pt/bin/activate  # <-- Activate the virtual environment

# UNCOMMENT the following lines on first run to set up lm-evaluation-harness:
# This is our fork of the lm-evaluation-harness with Portuguese tasks added
# git clone --branch polyglot_harness_portuguese https://github.com/Polygl0t/lm-evaluation-harness.git
# mv $workdir/lm-evaluation-harness $workdir/lm_evaluation_harness
# pip3 install -e $workdir/lm_evaluation_harness
# pip3 install "lm_eval[hf,vllm]"          # <-- Install lm-eval with HuggingFace and vLLM support
# Required for IFEval:
# pip3 install langdetect --no-cache-dir
# pip3 install immutabledict --no-cache-dir

# Available Portuguese evaluation tasks:
# - General Benchmarks: arc_challenge_poly_pt, mmlu_poly_pt, hellaswag_poly_pt, lambada_poly_pt
#                       calame_pt, global_piqa_completions_por_latn_braz, assin_paraphrase, assin_entailment, belebele_por_Latn
# - Instruction Following and Math: gsm8k_pt, ifeval_pt
# - Long context: ruler_pt

#############################################
# Configuration Variables
#############################################
export MAX_GPUS_PER_NODE=8                                                      # <-- Set to 8 for MLGPU nodes, 4 for SGPU nodes
export HF_TOKEN="<your-token-here>"                         # <-- Add your Hugging Face token here (required for gated models)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK                                     # <-- Set OpenMP threads to match CPU allocation
export HF_DATASETS_CACHE="$workdir/.eval_cache"                                 # <-- Cache directory for datasets
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE/models"                        # <-- Cache directory for models
export CLEAN_CACHE="0"                                                          # <-- Set to "1" to clean cache after job completion, "0" to keep cache
# General Benchmarks for Portuguese LLMs
export TASKS="\
arc_challenge_poly_pt,\
mmlu_poly_pt,\
hellaswag_poly_pt,\
lambada_poly_pt,\
calame_pt,\
global_piqa_completions_por_latn_braz,\
assin_paraphrase,\
assin_entailment,\
belebele_por_Latn"
# Evals for Instruction Following and Math
# export TASKS="gsm8k_pt,ifeval_pt"
# Evals for long context
# export TASKS="ruler_pt"
export NUM_FEWSHOT=5                                                            # <-- Number of few-shot examples (0 for zero-shot)
export EVAL_MODE="checkpoints"                                                  # <-- Options: "checkpoints" or "models"
export CHECKPOINT_DIR="$workdir/checkpoints"                                    # <-- Path to checkpoint directory
export MODELS_TO_EVAL="TucanoBR/Tucano-1b1 TucanoBR/Tucano-2b4"                 # <-- HuggingFace model IDs to evaluate
export EVAL_OUTPUT_DIR="$CHECKPOINT_DIR/.evals_new"                             # <-- Directory for evaluation results (YAML files)
export LOGS_FOLDER="$workdir/.eval_logs_portuguese"                             # <-- Directory for evaluation logs and JSON results
mkdir -p "$EVAL_OUTPUT_DIR"                                                     # <-- Create output directory
mkdir -p "$LOGS_FOLDER"                                                         # <-- Create logs directory

#############################################
# Model Download Phase (for EVAL_MODE="models")
#############################################
# Authenticate with HuggingFace (required for gated/private models)
if [ -n "$HF_TOKEN" ]; then
    hf auth login --token "$HF_TOKEN"      # <-- Login with token
fi

# Download models to local cache before evaluation (only in models mode)
# This prevents re-downloading during evaluation and ensures all models are available locally
if [ "$EVAL_MODE" == "models" ]; then
    echo "====================================="
    echo "Downloading HuggingFace models to cache"
    echo "====================================="
    
    # Convert space-separated string to array for download
    read -ra MODEL_DOWNLOAD_LIST <<< "$MODELS_TO_EVAL"
    
    for model in "${MODEL_DOWNLOAD_LIST[@]}"; do
        model_name=$(basename "$model")        # <-- Extract model name from HuggingFace ID
        model_dir="$HUGGINGFACE_HUB_CACHE/$model_name"  # <-- Construct local cache path
        if [ ! -d "$model_dir" ]; then
            echo "Downloading model $model to $model_dir"
            if hf download "$model" --local-dir "$model_dir" >/dev/null 2>&1; then  # <-- Download silently
                parent_dir="$(dirname "$model_dir")"
                # Remove duplicate cache directory created by hf download (cleanup)
                if [ -d "$parent_dir/models--$model_name" ]; then
                    rm -rf "$parent_dir/models--$model_name"
                fi
                echo "Download completed for $model"
            else
                echo "Failed to download model $model"
                exit 1                         # <-- Exit on download failure
            fi
        else
            echo "Model $model already downloaded in $model_dir"  # <-- Skip if already cached
        fi
    done
    echo "====================================="
fi

#############################################
# Discover What Needs Evaluation
# 
# This section checks which models/checkpoints have already been evaluated (by looking for YAML files)
# and creates a list of items that still need evaluation.
#############################################
echo "====================================="
echo "Evaluation Mode: $EVAL_MODE"
echo "====================================="

declare -a MODELS_TO_EVAL_ARRAY=()         # <-- Array to store models/checkpoints needing evaluation
declare -a MODEL_NAMES=()                  # <-- Array to store model names (for file naming)

if [ "$EVAL_MODE" == "checkpoints" ]; then
    echo "Discovering checkpoints in: $CHECKPOINT_DIR"
    
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "ERROR: Checkpoint directory does not exist: $CHECKPOINT_DIR"
        exit 1
    fi
    
    # Find all step_* directories
    for checkpoint in "$CHECKPOINT_DIR"/step_*; do
        if [ -d "$checkpoint" ]; then
            checkpoint_name=$(basename "$checkpoint")
            # Check if evaluation already exists (look for results file with checkpoint name)
            eval_file="$EVAL_OUTPUT_DIR/${checkpoint_name}.yaml"
            
            # Check if evaluation already exists
            if [ -f "$eval_file" ]; then
                echo "✓ $checkpoint_name already evaluated (found $eval_file)"  # <-- Skip this checkpoint
            else
                echo "✗ $checkpoint_name needs evaluation"
                MODELS_TO_EVAL_ARRAY+=("$checkpoint")   # <-- Add to evaluation queue
                MODEL_NAMES+=("$checkpoint_name")
            fi
        fi
    done

elif [ "$EVAL_MODE" == "models" ]; then
    echo "Evaluating HuggingFace models from cache"
    
    # Convert space-separated string to array
    read -ra MODEL_LIST <<< "$MODELS_TO_EVAL"
    
    for model in "${MODEL_LIST[@]}"; do
        # Extract model name for filename (get string after last /)
        model_name=$(basename "$model")
        eval_file="$EVAL_OUTPUT_DIR/${model_name}.yaml"  # <-- Check if YAML result exists
        
        # Check if evaluation already exists
        if [ -f "$eval_file" ]; then
            echo "✓ $model already evaluated (found $eval_file)"  # <-- Skip this model
        else
            echo "✗ $model needs evaluation"
            MODELS_TO_EVAL_ARRAY+=("$model")   # <-- Add to evaluation queue
            MODEL_NAMES+=("$model_name")
        fi
    done
else
    echo "ERROR: Invalid EVAL_MODE '$EVAL_MODE'. Must be 'checkpoints' or 'models'"
    exit 1
fi

# Count how many models/checkpoints need evaluation
NUM_TO_EVAL=${#MODELS_TO_EVAL_ARRAY[@]}

echo "====================================="
echo "Total items needing evaluation: $NUM_TO_EVAL"
echo "====================================="

# Exit if nothing to evaluate
if [ $NUM_TO_EVAL -eq 0 ]; then
    echo "All items have already been evaluated. Exiting."
    exit 0                                  # <-- Graceful exit when no work to do
fi

# Determine how many GPUs we actually need (capped by node type)
NUM_GPUS_NEEDED=$NUM_TO_EVAL
if [ $NUM_GPUS_NEEDED -gt $MAX_GPUS_PER_NODE ]; then
    NUM_GPUS_NEEDED=$MAX_GPUS_PER_NODE      # <-- Cap at MAX_GPUS_PER_NODE (8 for MLGPU, 4 for SGPU)
fi

echo "Will use $NUM_GPUS_NEEDED GPU(s) for evaluation"
echo "====================================="

#############################################
# Setup Output Files
# 
# Create separate output and error files for each evaluation job.
# This allows for debugging individual model evaluations.
#############################################
mkdir -p "$workdir/job_outputs"             # <-- Directory for stdout/stderr logs

declare -a OUT_FILES=()                     # <-- Array of stdout file paths
declare -a ERR_FILES=()                     # <-- Array of stderr file paths

for i in $(seq 0 $((NUM_TO_EVAL - 1))); do
    OUT_FILES+=("$workdir/job_outputs/out-eval-portuguese-auto-$((i+1)).$SLURM_JOB_ID")
    ERR_FILES+=("$workdir/job_outputs/err-eval-portuguese-auto-$((i+1)).$SLURM_JOB_ID")
    # Write job header to output file
    echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "${OUT_FILES[$i]}"
    echo "# Working directory: $workdir" >> "${OUT_FILES[$i]}"
    echo "# Python executable: $(which python3)" >> "${OUT_FILES[$i]}"
done

#############################################
# Main Job Execution
# 
# This section launches evaluation jobs in parallel, one per model/checkpoint.
# Each evaluation runs on a separate GPU.
# Jobs run in the background (&) and we track their process IDs.
#############################################
echo "Starting evaluations..."

# Launch evaluation jobs for each item that needs it
for i in $(seq 0 $((NUM_TO_EVAL - 1))); do
    model="${MODELS_TO_EVAL_ARRAY[$i]}"     # <-- Model path (checkpoint dir or HuggingFace ID)
    model_name="${MODEL_NAMES[$i]}"         # <-- Model name for file naming
    gpu_id=$((i % NUM_GPUS_NEEDED))         # <-- Assign GPU in round-robin fashion
    ucx_device=$gpu_id                      # <-- Network device ID (matches GPU ID)
    
    out_file="${OUT_FILES[$i]}"
    err_file="${ERR_FILES[$i]}"
    
    echo "Launching evaluation for $model_name on GPU $gpu_id"
    
    export CUDA_VISIBLE_DEVICES=$gpu_id                     # <-- Make only this GPU visible to the process
    export UCX_NET_DEVICES=mlx5_${ucx_device}:1             # <-- Set network device for GPU communication
    
    # Set MODEL_PATH based on evaluation mode
    if [ "$EVAL_MODE" == "checkpoints" ]; then
        export MODEL_PATH="$model"                          # <-- Use checkpoint directory directly
    else
        export MODEL_PATH="$HUGGINGFACE_HUB_CACHE/$model_name"  # <-- Path to locally cached model
    fi
    
    # Launch evaluation in background
    # For more details on available arguments, check the user guide:
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
    #
    # Common Options:
    # --apply_chat_template                       : Use for chat models
    # --fewshot_as_multiturn                      : Use for multi-turn fewshot examples
    # --model_args=...,enable_thinking=True       : Enable thinking mode for chain-of-thought prompting
    # --metadata='{"max_seq_lengths":[...]}'      : For long context models
    # --model_args=...,max_length=32768           : Set maximum sequence length
    # --model_args=...,think_end_token='</think>' : End token for reasoning models
    CUDA_VISIBLE_DEVICES=$gpu_id python3 $workdir/lm_evaluation_harness/lm_eval \
        --model hf \
        --model_args pretrained="$MODEL_PATH" \
        --tasks "$TASKS" \
        --batch_size "auto" \
        --num_fewshot $NUM_FEWSHOT \
        --device "cuda" \
        --output_path "$LOGS_FOLDER" 1>"$out_file" 2>"$err_file" &  # <-- Run in background
    
    # Store the PID for tracking
    PIDS[$i]=$!                             # <-- Save process ID to wait for it later
    
    # Small delay to stagger launches
    sleep 2                                 # <-- Prevents overwhelming the system with simultaneous starts
done

echo "All evaluation jobs launched. Waiting for completion..."

# Wait for all background jobs to complete
for i in $(seq 0 $((NUM_TO_EVAL - 1))); do
    wait ${PIDS[$i]}                        # <-- Block until this process completes
    exit_code=$?                            # <-- Capture exit code
    model_name="${MODEL_NAMES[$i]}"
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ Evaluation completed successfully for $model_name"
    else
        echo "✗ Evaluation failed for $model_name (exit code: $exit_code)"
    fi
done

#############################################
# Post-processing (JSON to YAML Conversion)
# 
# Converts JSON evaluation results to YAML format for easier analysis.
# Flattens nested result structures and extracts model metadata.
# Requires: jq (JSON processor)
#############################################
echo "====================================="
echo "Running post-processing..."
echo "====================================="

YAML_OUTPUT_DIR="$EVAL_OUTPUT_DIR"
mkdir -p "$YAML_OUTPUT_DIR"                 # <-- Ensure output directory exists

# Find all JSON files recursively in logs folder
json_files=()
while IFS= read -r -d '' file; do
    json_files+=("$file")
done < <(find "$LOGS_FOLDER" -type f -name "*.json" -print0)

if [ ${#json_files[@]} -eq 0 ]; then
    echo "No JSON files found in $LOGS_FOLDER"
else
    echo "Found ${#json_files[@]} JSON file(s) to process"
    
    # Check if jq is available (required for JSON parsing)
    if ! command -v jq &> /dev/null; then
        echo "WARNING: jq is not installed. Skipping post-processing."
        echo "Install with: module load jq (or apt-get install jq)"
    else
        # Process each JSON file
        for file in "${json_files[@]}"; do
            echo "Processing $(basename "$file")..."
            
            # Extract pretrained model path from JSON config
            pretrained=$(jq -r '.config.model_args.pretrained // empty' "$file" 2>/dev/null || echo "")
            
            # Determine model name (from pretrained path or filename)
            if [ -n "$pretrained" ]; then
                model_name=$(basename "$pretrained" | sed 's:/*$::')  # <-- Use model path
            else
                # Extract from filename as fallback
                fname=$(basename "$file" .json)
                # Remove common prefixes
                fname="${fname#results_}"
                fname="${fname#result_}"
                fname="${fname#eval_}"
                model_name="$fname"         # <-- Use filename
            fi
            
            output_file="$YAML_OUTPUT_DIR/${model_name}.yaml"
            
            # Skip if already exists
            if [ -f "$output_file" ]; then
                echo "  ✓ YAML already exists, skipping"
                continue
            fi
            
            # Build YAML output
            {
                echo "model_name: $model_name"
                if [ -n "$pretrained" ]; then
                    echo "model_pretrained: $pretrained"
                else
                    echo "model_pretrained: null"
                fi
                echo "results:"
                
                # Extract and flatten results from nested JSON structure
                # Removes ",none" suffix from metric names for cleaner output
                jq -r '
                    (.results // .) | 
                    to_entries[] | 
                    if .value | type == "object" then
                        .value | to_entries[] | 
                        "  " + (.key | sub(",none$"; "")) + ": " + (.value | tostring)
                    else
                        "  " + .key + ": " + (.value | tostring)
                    end
                ' "$file" 2>/dev/null || echo "  error: failed to parse results"
                
            } > "$output_file"
            
            echo "  ✓ Created $(basename "$output_file")"
        done
        
        echo "✓ Post-processing completed successfully"
    fi
fi

#############################################
# Finalize
#############################################
for i in $(seq 0 $((NUM_TO_EVAL - 1))); do
    echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "${OUT_FILES[$i]}"
done

echo "====================================="
echo "Job completed at: $(date)"
echo "Evaluated $NUM_TO_EVAL item(s) in $EVAL_MODE mode"
echo "====================================="

#############################################
# Cleanup and Validation
# 
# Check if all evaluations completed successfully and optionally clean up cache.
#############################################
ALL_SUCCESS=true
for i in $(seq 0 $((NUM_TO_EVAL - 1))); do
    model_name="${MODEL_NAMES[$i]}"
    eval_file="$EVAL_OUTPUT_DIR/${model_name}.yaml"
    
    # Check if the evaluation file was created (indicates success)
    if [ ! -f "$eval_file" ]; then
        ALL_SUCCESS=false
        break
    fi
done

if [ "$ALL_SUCCESS" = true ]; then
    echo "✓ All evaluations completed successfully."
else
    echo "⚠ Some evaluations failed. Check logs in $workdir/job_outputs for details."
fi

#############################################
# Cache Cleanup (Optional)
#############################################
# Clean HF_DATASETS_CACHE folder if requested
# This can save significant disk space but will require re-downloading on next run
if [ "$CLEAN_CACHE" = "1" ]; then
    echo "Cleaning HF_DATASETS_CACHE..."
    if [ -d "$HF_DATASETS_CACHE" ]; then
        find "$HF_DATASETS_CACHE" -mindepth 1 -delete 2>/dev/null || true  # <-- Delete all cache contents
        echo "✓ Cache cleaned"
    fi
else
    echo "Skipping cache cleanup (CLEAN_CACHE=$CLEAN_CACHE)"
fi

#############################################
# End of Script
#############################################
echo "====================================="
echo "Results available at: $EVAL_OUTPUT_DIR/"
echo "Job logs available at: $workdir/job_outputs/"
echo "====================================="