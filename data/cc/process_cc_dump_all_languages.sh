#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=lm_long                # <-- Change to your partition
#SBATCH --job-name=cc-lang-filter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=7-00:00:00
#SBATCH --mem=1900G
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                        # <-- Change to the corresponding username that created the workspace
file_system="scratch"                          # <-- Change to your filesystem
workspace_name="polyglot_datasets"             # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/process-cc-all-languages-out.$SLURM_JOB_ID"
err="$workdir/run_outputs/process-cc-all-languages-err.$SLURM_JOB_ID"

#############################################
# Environment Setup
#############################################
source $workdir/.modules_intel.sh
#python3 -m venv $workdir/.venv_intel
source $workdir/.venv_intel/bin/activate

#pip3 install --upgrade pip --no-cache-dir
#pip3 install datatrove[io,processing] --no-cache-dir
#pip3 install lxml[html_clean] --no-cache-dir
#pip3 install stanza --no-cache-dir
#pip3 install spacy --no-cache-dir
#pip3 install pyyaml==6.0.2 --no-cache-dir

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_CPUS_PER_TASK CPUs per task" >> "$out"
echo "# Working directory: $workdir" >> "$out"
echo "# Python executable: $(which python3)" >> "$out"

#############################################
# Job Time Management Functions
#############################################
get_remaining_seconds() {
    local job_start=$(squeue -j $SLURM_JOB_ID -h -o %S 2>/dev/null || echo "")
    local job_timelimit=$(squeue -j $SLURM_JOB_ID -h -o %l 2>/dev/null || echo "7-00:00:00")
    
    # Convert time limit to seconds (assuming format like "7-00:00:00")
    local days=$(echo $job_timelimit | cut -d'-' -f1)
    local time_part=$(echo $job_timelimit | cut -d'-' -f2)
    local hours=$(echo $time_part | cut -d':' -f1)
    local minutes=$(echo $time_part | cut -d':' -f2)
    local seconds=$(echo $time_part | cut -d':' -f3)
    
    local total_seconds=$((days * 86400 + hours * 3600 + minutes * 60 + seconds))
    local elapsed_seconds=$SECONDS
    local remaining=$((total_seconds - elapsed_seconds))
    
    echo $remaining
}

count_available_warc_paths() {
    # Count available WARC paths from the warc.paths file
    local warc_paths_file="$workdir/common_crawl/$DUMP/warc.paths"
    
    if [[ -f "$warc_paths_file" ]]; then
        local count=$(wc -l < "$warc_paths_file" 2>/dev/null || echo "0")
    else
        local count=0
    fi
    
    echo $count
}

#############################################
# CommonCrawl Processing Variables
#############################################
export DUMP="CC-MAIN-2025-51"  # <-- Change to your desired CommonCrawl dump
export WARC_FILES_FOLDER="$workdir/common_crawl/$DUMP/warc_files"
export LOGS_FOLDER="$workdir/common_crawl/$DUMP/logs"
export TEMP_OUTPUT_FOLDER="$workdir/common_crawl/$DUMP/language_filter_output"
export OUTPUT_FOLDER="$workdir/common_crawl/$DUMP/all_languages"
export LANGUAGE_FILTER_BACKEND="ft176"  # ft176 AND glotlid
export LANGUAGE_THRESHOLD=0.65 # Language detection confidence threshold
export TOKENIZER_NAME_OR_PATH="Qwen/Qwen3-0.6B-Base"  # Good out-of-the-box tokenizer for many languages
export TOKENIZERS_PARALLELISM="false"  # Disable parallelism to avoid issues with tokenizers
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"
export WARCS_PER_CICLE=1000

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out"

#############################################
# Main Processing Loop
#############################################
iteration=1
min_time_buffer=3600  # Reserve 1 hour before job ends

# Before starting the loop, clean the folders in case they contain old data
mkdir -p "$WARC_FILES_FOLDER" "$LOGS_FOLDER" "$TEMP_OUTPUT_FOLDER" "$OUTPUT_FOLDER"
find "$WARC_FILES_FOLDER" -mindepth 1 -delete 2>/dev/null || true
find "$LOGS_FOLDER" -mindepth 1 -delete 2>/dev/null || true
find "$TEMP_OUTPUT_FOLDER" -mindepth 1 -delete 2>/dev/null || true

while true; do
    remaining_time=$(get_remaining_seconds)
    
    echo "# [${SLURM_JOB_ID}] Starting iteration $iteration at: $(date)" >> "$out"
    echo "# [${SLURM_JOB_ID}] Estimated remaining time: $remaining_time seconds" >> "$out"
    
    # Check available WARC paths
    available_warcs=$(count_available_warc_paths)
    echo "# [${SLURM_JOB_ID}] Available WARC paths: $available_warcs" >> "$out"
    
    # Check if we have enough WARC paths (at least 10)
    if [ $available_warcs -lt 10 ]; then
        echo "# [${SLURM_JOB_ID}] Not enough WARC paths remaining ($available_warcs < 10). Stopping." >> "$out"
        break
    fi
    
    # Check if we have enough time for another iteration (at least 2 hours)
    if [ $remaining_time -lt $((min_time_buffer + 7200)) ]; then
        echo "# [${SLURM_JOB_ID}] Not enough time remaining for another iteration. Stopping." >> "$out"
        break
    fi
    
    #############################################
    # Download Warcs
    #############################################
    echo "# [${SLURM_JOB_ID}] Iteration $iteration: Starting download phase" >> "$out"
    echo "# [${SLURM_JOB_ID}] Processing DUMP: $DUMP" >> "$out"
    bash $workdir/warc_files_download.sh $WARCS_PER_CICLE $DUMP --remove-downloaded >/dev/null 2>&1 &
    wait
    
    #############################################
    # Language Filtering Processing
    #############################################
    echo "# [${SLURM_JOB_ID}] Iteration $iteration: Starting language filtering of warcs" >> "$out"
    python3 -u "$workdir/process_cc_dump_all_languages.py" \
        --warc_files_folder "$WARC_FILES_FOLDER" \
        --temp_output_folder "$TEMP_OUTPUT_FOLDER" \
        --output_folder "$OUTPUT_FOLDER" \
        --logs_folder "$LOGS_FOLDER" \
        --dump "$DUMP" \
        --language_filter_backend "$LANGUAGE_FILTER_BACKEND" \
        --language_threshold $LANGUAGE_THRESHOLD \
        --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH" \
        --expand_metadata \
        --tasks $SLURM_CPUS_PER_TASK \
        --workers $SLURM_CPUS_PER_TASK 1>>"$out" 2>>"$err" &
    wait

    echo "# [${SLURM_JOB_ID}] Iteration $iteration: Processing completed" >> "$out"
    
    #############################################
    # Split Large JSONL Files
    #############################################
    echo "# [${SLURM_JOB_ID}] Iteration $iteration: Splitting large JSONL files" >> "$out"
    
    # Process each language subdirectory in OUTPUT_FOLDER
    if [ -d "$OUTPUT_FOLDER" ]; then
        for lang_dir in "$OUTPUT_FOLDER"/*/ ; do
            if [ -d "$lang_dir" ]; then
                lang_name=$(basename "$lang_dir")
                
                # Skip hidden directories (starting with .)
                if [[ "$lang_name" == .* ]]; then
                    continue
                fi
                
                python3 -u "$workdir/process_cc_dump_split_jsonl.py" \
                    --directory "$lang_dir" \
                    --max_tokens_per_chunk 100000000 \
                    --size_threshold_gb 1.0 1>>"$out" 2>>"$err"
            fi
        done
    fi
    
    echo "# [${SLURM_JOB_ID}] Iteration $iteration: File splitting completed" >> "$out"
    
    #############################################
    # Delete the content of temporary folders
    #############################################
    echo "# [${SLURM_JOB_ID}] Iteration $iteration: Cleaning up temporary files" >> "$out"
    find "$WARC_FILES_FOLDER" -mindepth 1 -delete 2>/dev/null || true
    find "$LOGS_FOLDER" -mindepth 1 -delete 2>/dev/null || true
    find "$TEMP_OUTPUT_FOLDER" -mindepth 1 -delete 2>/dev/null || true
    
    # Clean HF_DATASETS_CACHE folder
    echo "# [${SLURM_JOB_ID}] Iteration $iteration: Cleaning HF_DATASETS_CACHE" >> "$out"
    if [ -d "$HF_DATASETS_CACHE" ]; then
        find "$HF_DATASETS_CACHE" -mindepth 1 -delete 2>/dev/null || true
    fi
    
    echo "# [${SLURM_JOB_ID}] Iteration $iteration completed at: $(date)" >> "$out"
    
    #############################################
    # Archive and clean log files
    #############################################
    # Archive current iteration logs
    iteration_out="$workdir/run_outputs/process-cc-all-languages-out.$SLURM_JOB_ID.iter_$iteration"
    iteration_err="$workdir/run_outputs/process-cc-all-languages-err.$SLURM_JOB_ID.iter_$iteration"
    
    cp "$out" "$iteration_out"
    cp "$err" "$iteration_err"
    
    # Keep only the summary in main files and clear the rest
    echo "# [${SLURM_JOB_ID}] Job started at: $(date)" > "$out.tmp"
    echo "# [${SLURM_JOB_ID}] Completed iterations: $iteration" >> "$out.tmp"
    echo "# [${SLURM_JOB_ID}] Last iteration completed at: $(date)" >> "$out.tmp"
    echo "# [${SLURM_JOB_ID}] Detailed logs archived to: $iteration_out" >> "$out.tmp"
    mv "$out.tmp" "$out"
    
    # Clear error file but keep a summary
    echo "# [${SLURM_JOB_ID}] Error log cleared after iteration $iteration at: $(date)" > "$err"
    echo "# [${SLURM_JOB_ID}] Detailed error logs archived to: $iteration_err" >> "$err"
    
    iteration=$((iteration + 1))
    
    # Brief pause between iterations
    sleep 60
done

#############################################
# End of Script
#############################################
echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out" 