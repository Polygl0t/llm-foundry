#!/bin/bash -l

#############################################
# Download WARC Files from Common Crawl
#
# Purpose:
#   Downloads actual WARC (Web ARChive) files from Common Crawl
#   based on paths in the warc.paths file. WARC files contain
#   raw web data (HTML, HTTP headers, metadata) from crawled websites.
#
# Prerequisites:
#   Run warc_paths_get.sh first to download the warc.paths index file
#
# Usage:
#   bash warc_files_download.sh [number_of_files] [cc_dump] [OPTIONS]
#
# Arguments:
#   number_of_files  - How many WARC files to download (optional, default: all)
#   cc_dump          - Common Crawl dump identifier (optional, default: CC-MAIN-2025-30)
#
# Options:
#   --remove-downloaded  - Remove successfully downloaded paths from warc.paths file
#                          Useful for resuming downloads and tracking progress
#
# Examples:
#   bash warc_files_download.sh                          # Download all WARC files
#   bash warc_files_download.sh 10                       # Download first 10 files
#   bash warc_files_download.sh 100 CC-MAIN-2025-26      # Download 100 files from specific dump
#   bash warc_files_download.sh 50 CC-MAIN-2025-26 --remove-downloaded  # Track progress
#
# Background Execution (for long downloads):
#   nohup bash warc_files_download.sh 100 > download.log 2>&1 &
#   nohup bash warc_files_download.sh 1000 CC-MAIN-2025-26 --remove-downloaded > download.log 2>&1 &
#
# Monitor Progress:
#   tail -f download.log              # Watch log file in real-time
#   tail -f nohup.out                 # If no log file specified
#   watch -n 5 'ls -lh warc_files/'   # Watch directory size
#
# Check if Running:
#   ps aux | grep warc_files_download.sh
#   jobs                              # If started in same terminal session
#############################################

#############################################
# Argument Parsing
#############################################
# Initialize variables
REMOVE_DOWNLOADED=false                    # <-- Flag to remove downloaded paths from index
NUM_FILES=""                               # <-- Number of files to download (empty = all)
CC_DUMP_ARG=""                             # <-- Common Crawl dump identifier

# Parse command line arguments
# Loop through all arguments and categorize them
for arg in "$@"; do
    case $arg in
        --remove-downloaded)
            REMOVE_DOWNLOADED=true         # <-- Enable path removal feature
            ;;
        *)
            # Assign positional arguments in order
            if [ -z "$NUM_FILES" ]; then
                NUM_FILES="$arg"           # <-- First positional arg: number of files
            elif [ -z "$CC_DUMP_ARG" ]; then
                CC_DUMP_ARG="$arg"         # <-- Second positional arg: CC dump ID
            fi
            ;;
    esac
done

#############################################
# Configuration
#############################################
# Set Common Crawl dump (use argument or default)
CC_DUMP=${CC_DUMP_ARG:-"CC-MAIN-2025-30"}  # <-- Change default dump if desired

# Define directory structure
DOWNLOAD_DIR="./common_crawl/$CC_DUMP/warc_files"     # <-- Where WARC files are saved
WARC_PATHS_FILE="./common_crawl/$CC_DUMP/warc.paths"  # <-- Index file with WARC paths
TEMP_PATHS_FILE="./common_crawl/$CC_DUMP/warc.paths.tmp"  # <-- Temporary file for updates
BASE_URL="https://data.commoncrawl.org"    # <-- Common Crawl base URL

#############################################
# Initial Setup and Validation
#############################################
# Create directory for downloaded WARC files
mkdir -p "$DOWNLOAD_DIR"                   # <-- Creates nested directories if needed

# Display configuration summary
echo ""
echo "=== WARC Download Started at $(date) ==="
echo "CC_DUMP: $CC_DUMP"
if [ "$REMOVE_DOWNLOADED" = true ]; then
    echo "Remove downloaded paths: ENABLED (paths will be removed from warc.paths)"
else
    echo "Remove downloaded paths: DISABLED (warc.paths will remain unchanged)"
fi

# Verify that warc.paths file exists
if [ ! -f "$WARC_PATHS_FILE" ]; then
    echo ""
    echo "Error: WARC paths file not found at $WARC_PATHS_FILE"
    echo "Please run warc_paths_get.sh first to download the paths index."
    echo ""
    echo "Example: bash warc_paths_get.sh $CC_DUMP"
    exit 1
fi

#############################################
# Download Planning
#############################################
# Determine how many files to download
NUM_FILES=${NUM_FILES:-"all"}              # <-- Default to "all" if not specified
TOTAL_FILES=$(wc -l < "$WARC_PATHS_FILE")  # <-- Count total available files

if [ "$NUM_FILES" != "all" ]; then
    if ! [[ "$NUM_FILES" =~ ^[0-9]+$ ]] || [ "$NUM_FILES" -le 0 ]; then
        echo "Error: Invalid number of files. Please provide a positive integer or omit for all files."
        exit 1
    fi
    if [ "$NUM_FILES" -gt "$TOTAL_FILES" ]; then
        echo "Warning: Requested $NUM_FILES files, but only $TOTAL_FILES available. Downloading all."
        NUM_FILES=$TOTAL_FILES
    fi
    echo "Downloading first $NUM_FILES out of $TOTAL_FILES WARC files..."
else
    echo "Downloading all $TOTAL_FILES WARC files..."
    NUM_FILES=$TOTAL_FILES
fi

#############################################
# Main Download Loop
#############################################
echo ""
echo "Starting download to $DOWNLOAD_DIR..."
echo ""

# Safety measure: Re-download the most recent file
# This ensures the last download completed successfully
# (in case the script was interrupted mid-download)
if [ -d "$DOWNLOAD_DIR" ] && [ "$(ls -A "$DOWNLOAD_DIR" 2>/dev/null)" ]; then
    LATEST_FILE=$(ls -t "$DOWNLOAD_DIR"/*.warc.gz 2>/dev/null | head -n1)  # <-- Find most recent file
    if [ -n "$LATEST_FILE" ]; then
        LATEST_FILENAME=$(basename "$LATEST_FILE")
        echo "Safety check: Removing latest file for re-download: $LATEST_FILENAME"
        rm -f "$LATEST_FILE"           # <-- Remove potentially incomplete file
    fi
fi

# Initialize counters
DOWNLOADED=0                               # <-- Count of successfully downloaded files
FAILED=0                                   # <-- Count of failed downloads

# Array to track successfully downloaded paths (for removal from index)
declare -a DOWNLOADED_PATHS=()

# Read warc.paths line by line and download each file
# IFS= : Prevents leading/trailing whitespace trimming
# -r   : Prevents backslash interpretation
while IFS= read -r warc_path && [ $DOWNLOADED -lt $NUM_FILES ]; do
    FILENAME=$(basename "$warc_path")      # <-- Extract filename from path
    FULL_URL="$BASE_URL/$warc_path"       # <-- Construct full download URL
    OUTPUT_FILE="$DOWNLOAD_DIR/$FILENAME" # <-- Local file path
    
    # Skip if file already exists (allows resuming downloads)
    if [ -f "$OUTPUT_FILE" ]; then
        echo "[$((DOWNLOADED + 1))/$NUM_FILES] Skipping $FILENAME (already exists)"
        ((DOWNLOADED++))
        # Track as downloaded for potential removal from index
        if [ "$REMOVE_DOWNLOADED" = true ]; then
            DOWNLOADED_PATHS+=("$warc_path")
        fi
        continue
    fi
    
    echo "[$((DOWNLOADED + 1))/$NUM_FILES] Downloading $FILENAME..."
    
    # Download using wget with robust options
    # --continue      : Resume partial downloads
    # --progress=bar  : Show progress bar
    # --timeout=30    : Timeout for network operations (seconds)
    # --tries=3       : Retry up to 3 times on failure
    wget --continue --progress=bar --timeout=30 --tries=3 \
         "$FULL_URL" -O "$OUTPUT_FILE"
    
    # Check download status
    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded $FILENAME"
        ((DOWNLOADED++))
        # Track path for removal from index
        if [ "$REMOVE_DOWNLOADED" = true ]; then
            DOWNLOADED_PATHS+=("$warc_path")
        fi
    else
        echo "✗ Failed to download $FILENAME"
        ((FAILED++))
        # Clean up partial file to prevent corruption
        [ -f "$OUTPUT_FILE" ] && rm "$OUTPUT_FILE"
    fi
    
done < "$WARC_PATHS_FILE"                  # <-- Read from warc.paths file

#############################################
# Update WARC Paths Index (Optional)
#############################################
# If --remove-downloaded flag was set, remove successfully
# downloaded paths from warc.paths file. This is useful for:
# - Tracking download progress
# - Resuming interrupted downloads
# - Avoiding re-download of already processed files
#############################################
if [ "$REMOVE_DOWNLOADED" = true ] && [ ${#DOWNLOADED_PATHS[@]} -gt 0 ]; then
    echo ""
    echo "Updating warc.paths file..."
    echo "Removing ${#DOWNLOADED_PATHS[@]} downloaded paths from index..."
    
    # Create a copy of the original paths file
    cp "$WARC_PATHS_FILE" "$TEMP_PATHS_FILE"
    
    # Remove each downloaded path from the temp file
    for path in "${DOWNLOADED_PATHS[@]}"; do
        # grep -v "^$path$" : Find lines that DON'T match exactly
        # ^                 : Start of line anchor
        # $                 : End of line anchor
        grep -v "^$path$" "$TEMP_PATHS_FILE" > "$TEMP_PATHS_FILE.new" && \
            mv "$TEMP_PATHS_FILE.new" "$TEMP_PATHS_FILE"
    done
    
    # Replace original file with updated version
    mv "$TEMP_PATHS_FILE" "$WARC_PATHS_FILE"
    
    REMAINING=$(wc -l < "$WARC_PATHS_FILE")
    echo "✓ Updated warc.paths file. Remaining files to download: $REMAINING"
fi

#############################################
# Download Summary
#############################################
echo ""
echo "=== Download Summary ==="
echo "Successfully downloaded: $DOWNLOADED files"
if [ $FAILED -gt 0 ]; then
    echo "Failed downloads: $FAILED files"
    echo "Note: Failed files were cleaned up. Re-run script to retry."
fi
echo "Download directory: $DOWNLOAD_DIR"
echo ""

# Calculate total size of downloaded files
if [ -d "$DOWNLOAD_DIR" ] && [ "$(ls -A "$DOWNLOAD_DIR" 2>/dev/null)" ]; then
    TOTAL_SIZE=$(du -sh "$DOWNLOAD_DIR" | cut -f1)
    FILE_COUNT=$(ls -1 "$DOWNLOAD_DIR"/*.warc.gz 2>/dev/null | wc -l)
    echo "Total files in directory: $FILE_COUNT"
    echo "Total size: $TOTAL_SIZE"
fi

#############################################
# End of Script
#############################################
exit 0