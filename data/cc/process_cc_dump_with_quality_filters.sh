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
#SBATCH --account=ag_bit_flek              # <-- Change to your SLURM account
#SBATCH --partition=lm_long                # <-- Change to your partition
#SBATCH --job-name=cc-2025-30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=7-00:00:00
#SBATCH --mem=1800G
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################

# Set this to your workspace root (where you have the .venv and .modules.sh files).
workdir="/lustre/mlnvme/data/polyglot"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/process-common-crawl-out.$SLURM_JOB_ID"
err="$workdir/run_outputs/process-common-crawl-err.$SLURM_JOB_ID"

#############################################
# Modules & Libraries Setup
#############################################

source $workdir/.modules.sh > "$out" 2>&1
# python3 -m venv $workdir/.venv_intel
source $workdir/.venv_intel/bin/activate

# ===== LLM Foundry Install =====
# pip3 install --upgrade pip --no-cache-dir
# git clone --depth 1 --branch main https://github.com/Polygl0t/llm-foundry.git
# pip3 install -e "$workdir/llm-foundry/.[data]" --no-cache-dir

# ===== Or, Manual Install without cloning the whole repo =====
# pip3 install --upgrade pip --no-cache-dir
# pip3 install datatrove[io,processing] \
#    lxml[html_clean] \
#    stanza \
#    spacy \
#    pyyaml==6.0.2 \
#    --no-cache-dir

# ===== Optional: Install Indic NLP Library for Indic languages =====
# pip3 install indic-nlp-library --no-cache-dir

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_CPUS_PER_TASK CPUs per task" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# [${SLURM_JOB_ID}] GLIBC version: $(ldd --version | head -n1)" >> "$out"
echo "# [${SLURM_JOB_ID}] Working directory: $workdir" >> "$out"
echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version) — $(python3 --version)" >> "$out"

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
export DUMP="CC-MAIN-2025-30"                                               # <-- Change to your desired CommonCrawl dump
export CONFIG_FOLDER="$workdir/llm-foundry/data/.configs"                   # <-- Change to your configuration folder if needed
export WARC_FILES_FOLDER="$workdir/common_crawl/$DUMP/warc_files"           # <-- Change to your desired WARC files folder if needed
export LOGS_FOLDER="$workdir/common_crawl/$DUMP/logs"                       # <-- Change to your desired logs folder if needed
export WARC_EXTRACTION_OUTPUT="$workdir/common_crawl/$DUMP/extracted_data"  # <-- Change to your desired WARC extraction output folder if needed
export QUALITY_FILTER_OUTPUT="$workdir/common_crawl/$DUMP/quality_filter"   # <-- Change to your desired quality filter output folder if needed
export FINAL_OUTPUT_FOLDER="$workdir/common_crawl/$DUMP/language_data"      # <-- Change to your desired final output folder if needed
export OUTPUT_FILE="$DUMP.jsonl"                                            # <-- Final output file name
export TOKENIZERS_PARALLELISM="false"                                       # <-- Disable parallelism to avoid issues with tokenizers
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK                                 # <-- Set OMP_NUM_THREADS to match the number of CPUs allocated per task
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"                    # <-- Set Hugging Face datasets cache to a job-specific directory to avoid conflicts between iterations and ensure proper cleanup
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"                           # <-- Set Hugging Face Hub cache to the same job-specific directory
export WARCS_PER_CICLE=1000                                                 # <-- Set the number of WARCs to process per cycle
export TOKENIZER_NAME_OR_PATH="Qwen/Qwen3-0.6B-Base"                        # <-- Good out-of-the-box tokenizer for many languages
export LANGUAGES="bn pt hi"                                                 # <-- Set the languages to filter (e.g., Bengali, Portuguese, Hindi)

#############################################
# Main Processing Loop
#############################################
iteration=1
min_time_buffer=3600  # Reserve 1 hour before job ends

# Before starting the loop, clean the folders in case they contain old data
mkdir -p "$WARC_FILES_FOLDER" "$LOGS_FOLDER" "$WARC_EXTRACTION_OUTPUT" "$QUALITY_FILTER_OUTPUT"
find "$WARC_FILES_FOLDER" -mindepth 1 -delete 2>/dev/null || true
find "$LOGS_FOLDER" -mindepth 1 -delete 2>/dev/null || true
find "$WARC_EXTRACTION_OUTPUT" -mindepth 1 -delete 2>/dev/null || true
find "$QUALITY_FILTER_OUTPUT" -mindepth 1 -delete 2>/dev/null || true

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
    # Pre-processing & Post-processing
    #############################################
    echo "# [${SLURM_JOB_ID}] Iteration $iteration: Starting Processing of warcs" >> "$out"
    python3 -u "$workdir/llm-foundry/data/cc/process_cc_dump_with_quality_filters.py" \
        --config_folder "$CONFIG_FOLDER" \
        --warc_files_folder "$WARC_FILES_FOLDER" \
        --logs_folder "$LOGS_FOLDER" \
        --warc_extraction_output "$WARC_EXTRACTION_OUTPUT" \
        --quality_filter_output "$QUALITY_FILTER_OUTPUT" \
        --final_output_folder "$FINAL_OUTPUT_FOLDER" \
        --output_file "$OUTPUT_FILE" \
        --dump "$DUMP" \
        --expand_metadata \
        --languages $LANGUAGES \
        --tokenizer_name_or_path "$TOKENIZER_NAME_OR_PATH" \
        --tasks $SLURM_CPUS_PER_TASK \
        --workers $SLURM_CPUS_PER_TASK 1>>"$out" 2>>"$err" &
    wait

    echo "# [${SLURM_JOB_ID}] Iteration $iteration: Processing completed" >> "$out"

    #############################################
    # Split Large JSONL Files
    #############################################
    echo "# [${SLURM_JOB_ID}] Iteration $iteration: Splitting large JSONL files" >> "$out"
    
    # Process each language subdirectory in OUTPUT_FOLDER
    if [ -d "$FINAL_OUTPUT_FOLDER" ]; then
        for lang_dir in "$FINAL_OUTPUT_FOLDER"/*/ ; do
            if [ -d "$lang_dir" ]; then
                lang_name=$(basename "$lang_dir")
                
                # Skip hidden directories (starting with .)
                if [[ "$lang_name" == .* ]]; then
                    continue
                fi
                
                python3 -u "$workdir/llm-foundry/data/cc/splitter.py" \
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
    find "$WARC_EXTRACTION_OUTPUT" -mindepth 1 -delete 2>/dev/null || true
    find "$QUALITY_FILTER_OUTPUT" -mindepth 1 -delete 2>/dev/null || true
    
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
    iteration_out="$workdir/run_outputs/process-common-crawl-out.$SLURM_JOB_ID.iter_$iteration"
    iteration_err="$workdir/run_outputs/process-common-crawl-err.$SLURM_JOB_ID.iter_$iteration"
    
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
