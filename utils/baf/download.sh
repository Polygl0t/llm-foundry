#!/bin/bash
# Simple download script for HuggingFace datasets on BAF login node.
# Edit the variables below, then run:
#   bash download.sh
#
# Notes:
# - Run on the login node (desktop12.physik.uni-bonn.de), NOT as a cluster job.
# - Best run inside screen: screen -S download && bash scripts/download.sh
# - Point output_dir and cache_dir to $BUDDY, not to your home directory.
set -e

# ═══════════════════════════════════════════
#  EDIT THESE VARIABLES
# ═══════════════════════════════════════════
export repo_name="Polygl0t/gigalekh-v1"
export output_dir="$BUDDY/data"
export cache_dir="$BUDDY/.cache"
export token="$HF_TOKEN"
export repo_type="dataset"
export allow_patterns="default/*.parquet"
export foundry_dir="$HOME/llm-foundry"
export venv_dir="$HOME/.venv"
# ═══════════════════════════════════════════

# ---- Load modules ----
echo "===== Loading modules ====="
module load miniforge/24.7.1-0-py312

# ---- Create lightweight venv (comment out if you already have one) ----
python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"

# ---- Install huggingface_hub (comment out if you already have it) ----
pip3 install --upgrade pip -q
pip3 install huggingface_hub -q

# ---- Run download ----
echo "===== Downloading ====="
echo "  Foundry dir: $foundry_dir"
echo "  Venv dir:    $venv_dir"
echo "  Python:      $(which python3) : $(python3 --version)"
echo "  Repo:        $repo_name"
echo "  Output:      $output_dir"
echo "  Cache:       $cache_dir"

python3 "$foundry_dir/utils/download.py" \
    --repo_name "$repo_name" \
    --output_dir "$output_dir" \
    --cache_dir "$cache_dir" \
    --token "$token" \
    --repo_type "$repo_type" \
    --allow_patterns "$allow_patterns"

echo "===== Done ====="
deactivate
