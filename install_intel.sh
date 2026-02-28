#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# Learn more about SLURM options at:
# - https://slurm.schedmd.com/sbatch.html
#############################################
#SBATCH --account=ag_cst_gabriel           # <-- Change to your SLURM account
#SBATCH --partition=lm_devel               # <-- Change to your partition
#SBATCH --job-name=installation_intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --mem=1900G
#SBATCH --exclusive

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

#############################################
# Working Directory Setup
#############################################
username="nklugeco_hpc"                    # <-- Change to the corresponding username that created the workspace
file_system="mlnvme"                       # <-- Change to your filesystem
workspace_name="polyglot"                  # <-- Change to your workspace/project name

workdir="/lustre/$file_system/data/$username-$workspace_name"
mkdir -p "$workdir/run_outputs"
cd "$workdir"
ulimit -c 0

out="$workdir/run_outputs/out-installation-intel.$SLURM_JOB_ID"
err="$workdir/run_outputs/err-installation-intel.$SLURM_JOB_ID"
log="$workdir/run_outputs/log-installation-intel.$SLURM_JOB_ID"

# Initialize log file
echo "=================================================" | tee "$log"
echo "Intel Installation Log - Job ID: $SLURM_JOB_ID" | tee -a "$log"
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

# Check CPU info
cpu_model=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
log_success "CPU detected: $cpu_model"

# Check memory
total_mem=$(free -h | awk '/^Mem:/ {print $2}')
log_success "Total memory: $total_mem"

#############################################
# Environment Setup
#############################################
log_step "Setting up environment..."

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
log_success "OMP_NUM_THREADS=$OMP_NUM_THREADS"

source "$workdir/.modules_intel.sh"
log_success "Loaded Intel modules"

if [ ! -d "$workdir/.venv_intel" ]; then
    log_error "Virtual environment not found at $workdir/.venv_intel"
    log_error "Please run ws_installation.sh first to create the virtual environment"
    exit 1
fi

source "$workdir/.venv_intel/bin/activate"
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

# Install core packages (no GPU dependencies for Intel)
log_step "Installing core packages..."
core_packages="numpy pandas datasets transformers evaluate huggingface_hub scikit-learn matplotlib sentencepiece protobuf pyyaml"
for pkg in $core_packages; do
    install_package "$pkg" "--no-cache-dir" || exit 1
done

# Install data processing packages
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
python3 "$workdir/install_intel.py" 2>&1 | tee -a "$log"
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
if [ $validation_exit_code -ne 0 ] || [ $failure_count -gt 0 ]; then
    log_error "Installation completed with errors"
    exit 1
else
    log_success "Installation completed successfully"
    exit 0
fi
