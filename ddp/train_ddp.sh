#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=sgpu_long              # <-- Change to your partition
#SBATCH --job-name=ddp-training
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
workspace_name="nanotronics"                 # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_training_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_training_outputs/out.$SLURM_JOB_ID"
err="$workdir/run_training_outputs/err.$SLURM_JOB_ID"

#############################################
# Working Build : )
# Python 3.12, CUDA 12.6, PyTorch 2.8, and CXX11 ABI set to TRUE.
#############################################
source $workdir/.modules_amd.sh
# python3 -m venv $workdir/.venv_ddp
source $workdir/.venv_ddp/bin/activate
# pip3 install --upgrade pip
# pip3 install wheel==0.45.1 packaging==25.0 --no-cache-dir
# pip3 install -r "$workdir/ddp/requirements.txt" --no-cache-dir
# pip3 install flash_attn==2.8.2 --no-build-isolation --no-cache-dir

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
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export WANDB_DIR="$HF_DATASETS_CACHE/wandb"
export TRITON_CACHE_DIR="$HF_DATASETS_CACHE/triton_cache/$SLURM_JOB_ID"
export NCCL_TIMEOUT=300
export TORCH_FR_BUFFER_SIZE=1000
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_IB_TIMEOUT=20
export NCCL_IB_RETRY_CNT=7
#export NCCL_DEBUG=INFO # Uncomment for NCCL debugging
MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)" # <-- Get the master node address
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')" # <-- Resolve to IP address
export MASTER_PORT=12340 # <-- Ensure this port is open in your SLURM cluster
export SPECS_FILE="$workdir/ddp/train_config.yaml"  # <-- Change to your specs file path

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES node(s)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Main Job Execution
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/srun.html
#############################################
srun --cpu-bind=none python3 "$workdir/ddp/train_ddp.py" \
    --specs "$SPECS_FILE" \
    --slurm-job-id "$SLURM_JOB_ID" \
    --hardware "a100" 1>>"$out" 2>>"$err"

#############################################
# Auto-Resubmit on Failure
#############################################
JOB_EXIT_CODE=$?
echo "# [${SLURM_JOB_ID}] Job finished at: $(date) with exit code $JOB_EXIT_CODE" >> "$out"

# If the job failed (non-zero exit code), automatically resubmit with checkpoint resume
if [ $JOB_EXIT_CODE -ne 0 ]; then
    echo "# [${SLURM_JOB_ID}] Job failed with exit code $JOB_EXIT_CODE. Resubmitting..." >> "$out"

    # Verify specs file exists
    if [ ! -f "$SPECS_FILE" ]; then
        echo "# [${SLURM_JOB_ID}] ERROR: Specs file not found at $SPECS_FILE. Cannot resubmit." >> "$out"
        exit 1
    fi

    # Look for the `checkpoint_dir` in the specs file, and remove any '" characters wrapping the path
    CHECKPOINT_DIR=$(grep -i 'checkpoint_dir:' "$SPECS_FILE" | awk '{print $2}' | tr -d '"' | xargs)
    
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "# [${SLURM_JOB_ID}] ERROR: Could not find checkpoint_dir in specs file. Not resubmitting." >> "$out"
        exit 1
    fi

    # Look for the current `stage_name` in the specs file
    STAGE_NAME=$(grep -i 'stage_name:' "$SPECS_FILE" | awk '{print $2}' | tr -d '"' | xargs)
    
    # Check what is the folder that starts with "slurm_job_" that is the latest to be updated
    LATEST_JOB_DIR=$(ls -td "$CHECKPOINT_DIR"/slurm_job_* 2>/dev/null | head -n 1)
    
    if [ -z "$LATEST_JOB_DIR" ]; then
        echo "# [${SLURM_JOB_ID}] No checkpoint job directory found in $CHECKPOINT_DIR. Not resubmitting." >> "$out"
        exit 1
    fi

    # Add the `stage_name` to the path
    FULL_CHECKPOINT_DIR="$LATEST_JOB_DIR/$STAGE_NAME"
    
    # Verify that the checkpoint directory has folders that start with "step_"
    if [ -d "$FULL_CHECKPOINT_DIR" ] && ls "$FULL_CHECKPOINT_DIR"/step_* 1> /dev/null 2>&1; then
        echo "# [${SLURM_JOB_ID}] Found checkpoint directory at $FULL_CHECKPOINT_DIR. Updating specs file for resumption." >> "$out"
        
        # Create a backup of the specs file
        cp "$SPECS_FILE" "${SPECS_FILE}.backup.${SLURM_JOB_ID}"
        
        # Update the specs file to include the checkpoint path for resumption and set begin_new_stage to false
        sed -i "/^resume_from_checkpoint:/c\resume_from_checkpoint: \"$FULL_CHECKPOINT_DIR\"" "$SPECS_FILE"
        sed -i "/^begin_new_stage:/c\begin_new_stage: false" "$SPECS_FILE"
        
        # Verify the sed commands succeeded
        if ! grep -q "resume_from_checkpoint: \"$FULL_CHECKPOINT_DIR\"" "$SPECS_FILE" || ! grep -q "begin_new_stage: false" "$SPECS_FILE"; then
            echo "# [${SLURM_JOB_ID}] ERROR: Failed to update specs file. Restoring backup." >> "$out"
            mv "${SPECS_FILE}.backup.${SLURM_JOB_ID}" "$SPECS_FILE"
            exit 1
        fi
        
        # Clean up old backup files (keep only the 3 most recent)
        ls -t "${SPECS_FILE}.backup."* 2>/dev/null | tail -n +4 | xargs -r rm -f
        
        # Get the current script path
        JOB_SCRIPT=$(readlink -f "$0")
        
        # Create a temporary script without the dependency line to avoid chaining issues
        TEMP_SCRIPT="${JOB_SCRIPT}.resubmit.${SLURM_JOB_ID}"
        grep -v "^#SBATCH --dependency=" "$JOB_SCRIPT" > "$TEMP_SCRIPT"
        
        # Add a resubmission counter to prevent infinite loops
        RESUBMIT_COUNT_FILE="${workdir}/.resubmit_count_${SPECS_FILE##*/}"
        RESUBMIT_LOCK_FILE="${RESUBMIT_COUNT_FILE}.lock"
        MAX_RESUBMITS=5  # Maximum number of automatic resubmissions
        
        # Use file locking to prevent race conditions
        (
            flock -x 200 || exit 1
            
            if [ -f "$RESUBMIT_COUNT_FILE" ]; then
                RESUBMIT_COUNT=$(cat "$RESUBMIT_COUNT_FILE")
            else
                RESUBMIT_COUNT=0
            fi
            
            if [ "$RESUBMIT_COUNT" -ge "$MAX_RESUBMITS" ]; then
                echo "# [${SLURM_JOB_ID}] Maximum resubmission limit ($MAX_RESUBMITS) reached. Not resubmitting." >> "$out"
                rm -f "$RESUBMIT_COUNT_FILE"
                exit 1
            fi
            
            # Increment and save counter
            echo $((RESUBMIT_COUNT + 1)) > "$RESUBMIT_COUNT_FILE"
            echo "$RESUBMIT_COUNT"
        ) 200>"$RESUBMIT_LOCK_FILE"
        
        RESUBMIT_COUNT=$?
        
        # Get the current count from file for display
        CURRENT_RESUBMIT=$(cat "$RESUBMIT_COUNT_FILE" 2>/dev/null || echo 1)
        
        # Resubmit the job
        NEW_JOB_ID=$(sbatch "$TEMP_SCRIPT" 2>&1 | awk '{print $NF}')
        
        if [ -n "$NEW_JOB_ID" ] && [[ "$NEW_JOB_ID" =~ ^[0-9]+$ ]]; then
            echo "# [${SLURM_JOB_ID}] Job resubmitted successfully with ID: $NEW_JOB_ID (attempt ${CURRENT_RESUBMIT}/$MAX_RESUBMITS)" >> "$out"
            # Clean up temporary script
            rm -f "$TEMP_SCRIPT"
        else
            echo "# [${SLURM_JOB_ID}] ERROR: Failed to resubmit job. sbatch output: $NEW_JOB_ID" >> "$out"
            rm -f "$TEMP_SCRIPT"
            # Restore the original specs file on submission failure
            mv "${SPECS_FILE}.backup.${SLURM_JOB_ID}" "$SPECS_FILE" 2>/dev/null
            exit 1
        fi
    else
        echo "# [${SLURM_JOB_ID}] No valid checkpoint directory found at $FULL_CHECKPOINT_DIR. Not resubmitting." >> "$out"
    fi

else
    echo "# [${SLURM_JOB_ID}] Job completed successfully." >> "$out"
fi

#############################################
# Cleanup
#############################################
# Reset resubmission counter on success (exit code 0)
if [ $JOB_EXIT_CODE -eq 0 ]; then
    RESUBMIT_COUNT_FILE="${workdir}/.resubmit_count_${SPECS_FILE##*/}"
    rm -f "$RESUBMIT_COUNT_FILE"
fi

# Clean up temporary resubmit script if it exists
TEMP_SCRIPT="${0}.resubmit.${SLURM_JOB_ID}"
rm -f "$TEMP_SCRIPT"

# Remove the triton cache folder at the end.
rm -rf "$TRITON_CACHE_DIR"