#!/bin/bash -l

#############################################
# Download WARC paths from Common Crawl
#
# Purpose:
#   This script downloads the index file (warc.paths) 
#   that lists all available WARC files for a specific
#   Common Crawl dump. This index is needed before you
#   can download actual WARC files.
#
# Usage:
#   bash warc_paths_get.sh [CC_DUMP]
#
# Arguments:
#   CC_DUMP  - Common Crawl dump identifier (optional)
#              Defaults to "CC-MAIN-2025-30" if not provided
#              Find available dumps at: https://commoncrawl.org/get-started
#
# Examples:
#   bash warc_paths_get.sh                  # Use default dump (CC-MAIN-2025-30)
#   bash warc_paths_get.sh CC-MAIN-2025-51  # Download paths for December 2025 dump
#   bash warc_paths_get.sh CC-MAIN-2024-10  # Download paths for March 2024 dump
#
# Output:
#   Creates ./common_crawl/[CC_DUMP]/warc.paths
#   This file contains one WARC file path per line
#############################################

#############################################
# Configuration
#############################################
# Use first argument as CC_DUMP, or default to CC-MAIN-2025-30
CC_DUMP=${1:-"CC-MAIN-2025-30"}           # <-- Change default dump if desired
echo "Using CC_DUMP: $CC_DUMP"

# Set up directory structure for this dump
WARC_DIR="./common_crawl/$CC_DUMP"        # <-- All files for this dump go here
mkdir -p "$WARC_DIR"                       # <-- Create directory if it doesn't exist
#############################################
# Step 1: Download WARC Paths Index (Compressed)
#############################################
# The warc.paths.gz file contains a list of all WARC files
# available in this Common Crawl dump. Each line is a path
# to a WARC file that can be downloaded separately.
#############################################
echo "Fetching WARC paths for $CC_DUMP..."

# Download the compressed index file
# wget options:
#   -O : Specify output filename
wget https://data.commoncrawl.org/crawl-data/$CC_DUMP/warc.paths.gz \
     -O "$WARC_DIR/warc.paths.gz"

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to download WARC paths."
    echo "Please verify that dump '$CC_DUMP' exists at https://commoncrawl.org/"
    exit 1
fi

echo "✓ WARC paths downloaded successfully to $WARC_DIR/warc.paths.gz"
#############################################
# Step 2: Decompress WARC Paths Index
#############################################
# Decompress the .gz file to get a plain text file
# with one WARC path per line. This makes it easy
# to process with other scripts.
#############################################
echo "Unzipping WARC paths..."

# Decompress using gunzip (removes .gz file after extraction)
gunzip "$WARC_DIR/warc.paths.gz"

# Check if decompression was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to unzip WARC paths."
    exit 1
fi

echo "✓ WARC paths unzipped successfully to $WARC_DIR/warc.paths"

#############################################
# Summary
#############################################
TOTAL_WARCS=$(wc -l < "$WARC_DIR/warc.paths")
echo ""
echo "=== Summary ==="
echo "Total WARC files available: $TOTAL_WARCS"
echo "Paths file location: $WARC_DIR/warc.paths"

#############################################
# End of Script
#############################################
exit 0