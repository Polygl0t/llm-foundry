#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=sgpu_short             # <-- Change to your partition
#SBATCH --job-name=installation_amd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

#############################################
# Working Directory Setup
#############################################
# Dynamically resolve workdir from the script's own location.
# This works because installation.sh copies this script into workdir
# and submits it via: sbatch $workdir/install_amd.sh
workdir="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-installation-amd.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-installation-amd.$SLURM_JOB_ID"
log="$workdir/run_outputs/log-installation-amd.$SLURM_JOB_ID"

# Initialize log file
echo "=================================================" | tee "$log"
echo "AMD Installation Log - Job ID: $SLURM_JOB_ID" | tee -a "$log"
echo "Started at: $(date)" | tee -a "$log"
echo "=================================================" | tee -a "$log"
echo "" | tee -a "$log"

#############################################
# Helper Functions
#############################################
log_step() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ SUCCESS: $1" | tee -a "$log"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ ERROR: $1" | tee -a "$log"
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ WARNING: $1" | tee -a "$log"
}

#############################################
# Pre-flight Checks
#############################################
log_step "Running pre-flight checks..."

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not available. GPU may not be accessible."
    exit 1
fi
log_success "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"

# Check CUDA
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0
    log_warning "CUDA_VISIBLE_DEVICES not set, defaulting to 0"
fi
log_success "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

#############################################
# Environment Setup
#############################################
log_step "Setting up environment..."

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
log_success "OMP_NUM_THREADS=$OMP_NUM_THREADS"

source "$workdir/.modules_amd.sh"
log_success "Loaded AMD modules"

if [ ! -d "$workdir/.venv_amd" ]; then
    log_error "Virtual environment not found at $workdir/.venv_amd"
    log_error "Please run ws_installation.sh first to create the virtual environment"
    exit 1
fi

source "$workdir/.venv_amd/bin/activate"
log_success "Activated virtual environment: $(which python3)"

#############################################
# Package Installation with Error Handling
#############################################
log_step "Starting package installations..."
echo "" | tee -a "$log"

# Track installation status
declare -A install_status
packages_failed=()

install_package() {
    local package_name="$1"
    local pip_args="${2:-}"
    
    log_step "Installing $package_name..."
    
    if pip3 install $pip_args $package_name 2>&1 | tee -a "$log"; then
        install_status["$package_name"]="SUCCESS"
        log_success "$package_name installed"
        return 0
    else
        install_status["$package_name"]="FAILED"
        packages_failed+=("$package_name")
        log_error "$package_name installation failed"
        return 1
    fi
}

# Upgrade pip first
log_step "Upgrading pip..."
if pip3 install --upgrade pip 2>&1 | tee -a "$log"; then
    log_success "pip upgraded"
else
    log_error "pip upgrade failed"
    exit 1
fi

# Install base packages
install_package "wheel" "--no-cache-dir" || exit 1
install_package "packaging" "--no-cache-dir" || exit 1

# Install PyTorch
log_step "Installing PyTorch ecosystem..."
if pip3 install torch torchvision torchaudio --no-cache-dir 2>&1 | tee -a "$log"; then
    install_status["torch"]="SUCCESS"
    log_success "PyTorch installed"
else
    install_status["torch"]="FAILED"
    packages_failed+=("torch")
    log_error "PyTorch installation failed"
    exit 1
fi

# Install flash-attn (can be problematic)
log_step "Installing flash-attn (this may take a while)..."
if pip3 install flash-attn --no-build-isolation --no-cache-dir 2>&1 | tee -a "$log"; then
    install_status["flash-attn"]="SUCCESS"
    log_success "flash-attn installed"
else
    install_status["flash-attn"]="FAILED"
    packages_failed+=("flash-attn")
    log_warning "flash-attn installation failed - continuing anyway"
fi

# Install vLLM
install_package "vllm" "--upgrade --no-cache-dir" || log_warning "vLLM installation failed - continuing anyway"

# Install core ML packages
log_step "Installing core ML packages..."
core_packages="numpy pandas datasets transformers trl evaluate huggingface_hub scikit-learn matplotlib sentencepiece protobuf pyyaml codecarbon wandb"
for pkg in $core_packages; do
    install_package "$pkg" "--no-cache-dir"
done

# Install additional packages
install_package "accelerate" "--no-cache-dir"
install_package "liger-kernel" "--no-cache-dir"
install_package "datatrove[io,processing]" "--no-cache-dir"
install_package "stanza" "--no-cache-dir"
install_package "spacy" "--no-cache-dir"
install_package "json-repair" "--no-cache-dir"

echo "" | tee -a "$log"
log_step "Package installation phase completed"
echo "" | tee -a "$log"

#############################################
# Validation Phase
#############################################
log_step "Starting validation phase..."
echo "" | tee -a "$log"

# Run validation script
python3 "$workdir/install_amd.py" 2>&1 | tee -a "$log"
validation_exit_code=$?

if [ $validation_exit_code -eq 0 ]; then
    log_success "Validation completed successfully"
else
    log_error "Validation failed with exit code $validation_exit_code"
fi

#############################################
# Final Summary Report
#############################################
echo "" | tee -a "$log"
echo "=================================================" | tee -a "$log"
echo "INSTALLATION SUMMARY" | tee -a "$log"
echo "=================================================" | tee -a "$log"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$log"
echo "Completed at: $(date)" | tee -a "$log"
echo "" | tee -a "$log"

# Count successes and failures
success_count=0
failure_count=0
for pkg in "${!install_status[@]}"; do
    if [ "${install_status[$pkg]}" = "SUCCESS" ]; then
        ((success_count++))
    else
        ((failure_count++))
    fi
done

echo "Packages attempted: $((success_count + failure_count))" | tee -a "$log"
echo "Successful installations: $success_count" | tee -a "$log"
echo "Failed installations: $failure_count" | tee -a "$log"

if [ ${#packages_failed[@]} -gt 0 ]; then
    echo "" | tee -a "$log"
    echo "Failed packages:" | tee -a "$log"
    for pkg in "${packages_failed[@]}"; do
        echo "  - $pkg" | tee -a "$log"
    done
fi

echo "" | tee -a "$log"
echo "Full logs available at:" | tee -a "$log"
echo "  - Main log: $log" | tee -a "$log"
echo "  - stdout: $out" | tee -a "$log"
echo "  - stderr: $err" | tee -a "$log"
echo "=================================================" | tee -a "$log"

#############################################
# End of Script
#############################################
# Exit with appropriate code
if [ $validation_exit_code -ne 0 ] || [ $failure_count -gt 2 ]; then
    log_error "Installation completed with errors"
    exit 1
else
    log_success "Installation completed successfully"
    exit 0
fi
