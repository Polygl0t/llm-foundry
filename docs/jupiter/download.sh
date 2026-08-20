#!/bin/bash
# Simple download script for HuggingFace repos on the JSC JUPITER login node.
# Edit the variables below, then run:
#   bash download.sh
#
# Notes:
# - Run on a login node (jpbl-*), NOT as a cluster job (no sbatch/salloc).
# - Best run inside screen/tmux: screen -S download && bash download.sh
# - Point output_dir and cache_dir to your project directory
#   (e.g. /e/project1/polyglot), not to your home directory.
set -e

# ═══════════════════════════════════════════
#  EDIT THESE VARIABLES
# ═══════════════════════════════════════════
export workdir="/e/project1/polyglot"
export repo_name="Polygl0t/gigaverbo-v2"
export output_dir="$workdir/data/portuguese/text"
export cache_dir="$workdir/.cache"
export token="$HF_TOKEN"  # <-- Set this to your HuggingFace token (or leave empty for public repos)
export repo_type="dataset"
export allow_patterns="*"
export foundry_dir="$workdir/llm-foundry"
export venv_dir="$workdir/.venv"
export modules_script="$workdir/jupiter_modules_2026.sh"
# ═══════════════════════════════════════════

# ---- Clone the llm-foundry repo if it doesn't exist ----
if [ ! -d "$foundry_dir" ]; then
    echo "===== Cloning llm-foundry ====="
    git clone https://github.com/Polygl0t/llm-foundry.git "$foundry_dir"
fi

# ---- Load modules ----
echo "===== Loading modules ====="
source "$modules_script"

# ---- Load the virtual environment ----
source "$venv_dir/bin/activate"

# ---- Install huggingface_hub (comment out if you already have it) ----
# pip3 install --upgrade pip -q
# pip3 install huggingface_hub -q

# ---- Alternatively, install with uv (comment out if you already have it) ----
# pip3 install uv
# uv pip install huggingface_hub

# ---- Run download ----
echo "===== Downloading ====="
echo "  Foundry dir: $foundry_dir"
echo "  Venv dir:    $venv_dir"
echo "  Python:      $(which python3) : $(python3 --version)"
echo "  Repo:        $repo_name"
echo "  Output:      $output_dir"
echo "  Cache:       $cache_dir"

python3 "$foundry_dir/tools/download.py" \
    --repo_name "$repo_name" \
    --output_dir "$output_dir" \
    --cache_dir "$cache_dir" \
    --token "$token" \
    --repo_type "$repo_type" \
    --allow_patterns "$allow_patterns"

echo "===== Done ====="
deactivate
