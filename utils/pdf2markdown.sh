#!/bin/bash -l

#############################################
# NOTICE
#############################################
# There are several new tools for PDF to
# markdown conversion, that (alegedly) 
# outperform marker.
#
# Here is a table of comparisons: https://github.com/opendataloader-project/opendataloader-pdf
# - source: 
#
# | Engine                  | Overall | Reading Order | Table | Heading | Speed (s/page) |
# |-------------------------|---------|---------------|-------|---------|----------------|
# | opendataloader [hybrid] | 0.907   | 0.934         | 0.928 | 0.821   | 0.463          |
# | docling                 | 0.882   | 0.898         | 0.887 | 0.824   | 0.762          |
# | nutrient                | 0.880   | 0.924         | 0.662 | 0.811   | 0.230          |
# | marker                  | 0.861   | 0.890         | 0.808 | 0.796   | 53.932         |
# | unstructured [hi_res]   | 0.841   | 0.904         | 0.588 | 0.749   | 3.008          |
# | edgeparse               | 0.837   | 0.894         | 0.717 | 0.706   | 0.036          |
# | opendataloader          | 0.831   | 0.902         | 0.489 | 0.739   | 0.015          |
# | mineru                  | 0.831   | 0.857         | 0.873 | 0.743   | 5.962          |
# | pymupdf4llm             | 0.732   | 0.885         | 0.401 | 0.412   | 0.091          |
# | unstructured            | 0.686   | 0.882         | 0.000 | 0.388   | 0.077          |
# | markitdown              | 0.589   | 0.844         | 0.273 | 0.000   | 0.114          |
# | liteparse               | 0.576   | 0.866         | 0.000 | 0.000   | 1.061          |
#
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
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=mlgpu_long             # <-- Change to your partition
#SBATCH --job-name=pdf-to-markdown
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a40:8
#SBATCH --exclusive

#############################################
# Working Directory Setup
#############################################

# Set this to your workspace root (where you have the .venv and .modules.sh files).
workdir="/lustre/mlnvme/data/polyglot"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-pdf-to-markdown.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-pdf-to-markdown.$SLURM_JOB_ID"

#############################################
# Modules & Libraries Setup
#############################################

source "$workdir/.modules.sh" > "$out" 2>&1
# python3 -m venv "$workdir/.venv_amd_pdf"
source "$workdir/.venv_amd_pdf/bin/activate"

# ===== Install PDF to Markdown Tool =====
# pip3 install marker-pdf --no-cache-dir

#############################################
# Environment Setup
#############################################

export NUM_DEVICES=8                        # <-- Number of GPUs to use
export NUM_WORKERS=8                        # <-- Same as NUM_DEVICES          
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE="$workdir/.cache"
export HF_HUB_CACHE="$HF_DATASETS_CACHE"
export HF_ASSETS_CACHE="$HF_DATASETS_CACHE"
export EXTRACT_IMAGES=False                 # <-- Set to True if you want to extract images from PDFs
export PDF_DIR="$workdir/pdfs"              # <-- Directory containing PDF files
export OUTPUT_DIR="$workdir/markdown_files" # <-- Directory to save converted markdown files
export CLEAN_CACHE="1"  # Set to "1" to clean cache after job completion
mkdir -p "$OUTPUT_DIR"

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NNODES node(s)" >> "$out"
echo "# [${SLURM_JOB_ID}] Using $SLURM_NTASKS GPUs in total ($SLURM_NTASKS_PER_NODE per node)" >> "$out"
echo "# [${SLURM_JOB_ID}] Running on nodes: $(scontrol show hostnames "$SLURM_NODELIST" | tr '\n' ' ')" >> "$out"
echo "# [${SLURM_JOB_ID}] GLIBC version: $(ldd --version | head -n1)" >> "$out"
echo "# [${SLURM_JOB_ID}] Working directory: $workdir" >> "$out"
echo "# [${SLURM_JOB_ID}] Python executable: $(which python3) — $(python3 --version) — $(python3 --version)" >> "$out"

#############################################
# Main Job Execution
#############################################
marker_chunk_convert "$PDF_DIR" "$OUTPUT_DIR" 1>>"$out" 2>>"$err"

#############################################
# End of Script
#############################################
# Clean HF_DATASETS_CACHE folder if requested
if [ "$CLEAN_CACHE" = "1" ]; then
    echo "# [${SLURM_JOB_ID}] Cleaning HF_DATASETS_CACHE" >> "$out"
    if [ -d "$HF_DATASETS_CACHE" ]; then
        find "$HF_DATASETS_CACHE" -mindepth 1 -delete 2>/dev/null || true
    fi
else
    echo "# [${SLURM_JOB_ID}] Skipping cache cleanup (CLEAN_CACHE=$CLEAN_CACHE)" >> "$out"
fi

echo "# [${SLURM_JOB_ID}] Job finished at: $(date)" >> "$out"
