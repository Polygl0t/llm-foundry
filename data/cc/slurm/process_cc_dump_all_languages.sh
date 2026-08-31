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
#
#
# This script overlaps download and compute.
# While the current batch of WARC files is being processed, the next batch is downloaded in the background.
# Trade-off: Peak disk usage roughly doubles compared to the sequential approach.
#
#############################################
#SBATCH --account=ag_bit_flek              # <-- Change to your SLURM account
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

# Set this to your workspace root (where you have the .venv and .modules.sh files).
workdir="/lustre/mlnvme/data/polyglot"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/process-cc-all-languages-out.$SLURM_JOB_ID"
err="$workdir/run_outputs/process-cc-all-languages-err.$SLURM_JOB_ID"

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

# ===== Alternatively, install with uv =====
# pip3 install --upgrade pip --no-cache-dir
# pip3 install uv
# uv pip install datatrove[io,processing] \
#    lxml[html_clean] \
#    stanza \
#    spacy \
#    pyyaml==6.0.2 \
#    --no-cache

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES nodes" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_CPUS_PER_TASK CPUs per task" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# [${SLURM_JOB_ID}] GLIBC version: $(ldd --version | head -n1)" >> "$out"
echo "# [${SLURM_JOB_ID}] Working directory: $workdir" >> "$out"
echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version)" >> "$out"

#############################################
# Job Time Management Functions
#############################################
duration_to_seconds() {
    local t="$1"
    local days=0
    local rest="$t"
    if [[ "$t" == *-* ]]; then
        days=${t%%-*}
        rest=${t#*-}
    fi
    local h=0 m=0 s=0
    IFS=: read -r h m s <<< "$rest"
    # 10# forces base-10; avoids "value too great for base" on zero-padded "08".
    echo $(( (10#$days)*86400 + (10#$h)*3600 + (10#$m)*60 + (10#$s) ))
}

get_remaining_seconds() {
    # SLURM_JOB_END_TIME: projected end as a UNIX timestamp (seconds).
    # Handles any --time format, including sub-day limits.
    if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
        echo $(( SLURM_JOB_END_TIME - $(date +%s) ))
        return
    fi

    # Fallback: parse squeue %l. SLURM emits "D-HH:MM:SS" only when >= 1 day,
    # and "HH:MM:SS" otherwise.
    local job_timelimit
    job_timelimit=$(squeue -j "$SLURM_JOB_ID" -h -o %l 2>/dev/null || echo "")
    if [[ -z "$job_timelimit" || "$job_timelimit" == "UNLIMITED" ]]; then
        echo 999999999
        return
    fi
    local total_seconds
    total_seconds=$(duration_to_seconds "$job_timelimit")
    echo $(( total_seconds - SECONDS ))
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
# CommonCrawl Paths & Configuration
#############################################
export DUMP="CC-MAIN-2025-51"                                                   # <-- Change to your desired CommonCrawl dump
export WARC_FILES_FOLDER="$workdir/common_crawl/$DUMP/warc_files"               # <-- Folder to store downloaded WARC files for this dump
export LOGS_FOLDER="$workdir/common_crawl/$DUMP/logs"                           # <-- Folder to store logs for this dump
export TEMP_OUTPUT_FOLDER="$workdir/common_crawl/$DUMP/language_filter_output"  # <-- Temporary folder for language filtering output before final processing
export OUTPUT_FOLDER="$workdir/common_crawl/$DUMP/all_languages"                # <-- Final output folder for processed data separated by language
export LANGUAGE_FILTER_BACKEND="ft176"                                          # <-- LID backend: ft176 AND glotlid
export LANGUAGE_THRESHOLD=0.65                                                  # <-- Language detection confidence threshold
export TOKENIZER_NAME_OR_PATH="Qwen/Qwen3-0.6B-Base"                            # <-- Good out-of-the-box tokenizer for many languages
export TOKENIZERS_PARALLELISM="false"                                           # <-- Disable parallelism to avoid issues with tokenizers
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK                                     # <-- Set OMP threads to match allocated CPUs
export HF_DATASETS_CACHE="$workdir/.cache/$SLURM_JOB_ID"                        # <-- Unique cache folder for this job to avoid conflicts with other jobs
export HUGGINGFACE_HUB_CACHE="$HF_DATASETS_CACHE"                               # <-- Use the same cache folder for Hugging Face Hub to avoid conflicts
export WARCS_PER_CICLE=1000                                                     # <-- Number of WARC files to process per iteration. Adjust based on available resources and expected processing time per WARC.

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" >> "$out"

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

# We handle two separate buffers to allow simultaneous downloading and processing of WARC files.
WARC_BUFFER_A="$WARC_FILES_FOLDER/buf_a"
WARC_BUFFER_B="$WARC_FILES_FOLDER/buf_b"
mkdir -p "$WARC_BUFFER_A" "$WARC_BUFFER_B"

active_dir="$WARC_BUFFER_A"
staging_dir="$WARC_BUFFER_B"

# Prime the first buffer before the loop.
bash "$workdir/warc_files_download.sh" "$WARCS_PER_CICLE" "$DUMP" \
    --remove-downloaded --download-dir "$active_dir" >/dev/null 2>&1

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
    echo "# [${SLURM_JOB_ID}] Iteration $iteration: Starting download in background" >> "$out"
    echo "# [${SLURM_JOB_ID}] Processing DUMP: $DUMP" >> "$out"

    # Download the next batch in the background while we compute.
    bash $workdir/warc_files_download.sh $WARCS_PER_CICLE $DUMP --remove-downloaded --download-dir $staging_dir >/dev/null 2>&1 &
    download_pid=$!

    #############################################
    # Language Filtering Processing
    #############################################
    echo "# [${SLURM_JOB_ID}] Iteration $iteration: Starting language filtering of warcs" >> "$out"

    # Process the current buffer in the foreground.
    python3 -u "$workdir/llm-foundry/data/cc/process_cc_dump_all_languages.py" \
        --warc_files_folder "$active_dir" \
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
    process_pid=$!

    #############################################
    # Swap Buffers
    #############################################

    # Wait for both downloading and processing before swapping, so the staging buffer is complete.
    wait "$download_pid" "$process_pid"

    echo "# [${SLURM_JOB_ID}] Iteration $iteration: Processing current warcs completed" >> "$out"
    echo "# [${SLURM_JOB_ID}] Iteration $iteration: Downloading warcs for next iteration completed" >> "$out"

    echo "# [${SLURM_JOB_ID}] Iteration $iteration: Swapping buffers" >> "$out"

    # Swapping buffers
    tmp="$active_dir"; active_dir="$staging_dir"; staging_dir="$tmp"

    # Clear the just-consumed buffer for its next reuse.
    find "$staging_dir" -mindepth 1 -delete 2>/dev/null || true

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

# Clear buffers
echo "# [${SLURM_JOB_ID}] Clearing buffers" >> "$out"
find "$WARC_BUFFER_A" -mindepth 1 -delete 2>/dev/null || true
find "$WARC_BUFFER_B" -mindepth 1 -delete 2>/dev/null || true


#############################################
# End of Script
#############################################
echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out"
