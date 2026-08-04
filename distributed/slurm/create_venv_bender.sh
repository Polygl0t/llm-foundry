#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# One-time setup: submit this job to create a ready-to-use venv for
# distributed training on Bender.
#
# Usage:
#   sbatch distributed/slurm/create_venv_bender.sh
#
# Stack selection (set LLM_FOUNDRY_STACK below):
#   - A40 partition  -> Intel stack
#   - A100 partition -> AMD stack
#
# Why separate venvs?  The module stacks provide different compiled
# toolchains and libraries that may be incompatible.  Keeping two
# venvs avoids subtle ABI issues.
#############################################
#SBATCH --partition=A40devel    # <-- Change to A100devel for AMD stack
#SBATCH --job-name=create-venv
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --gpus=1

set -e
#############################################
# Configuration — tweak these to your needs
#############################################

# Workspace root (your $HOME).
workdir="/home/nklugeco"

# Venv directory name.
venv_name=".venv_intel"

# Path to the .modules.sh file.
modules_file="$workdir/.modules.sh"

# GPU architectures for flash-attn kernel compilation.
#   8.0 = NVIDIA A100 (Ampere)
#   8.6 = NVIDIA A40  (Ampere)
flash_attn_cuda_archs="8.0;8.6"

# Stack override (remove or change if needed).
export LLM_FOUNDRY_STACK=intel

# Log files.
mkdir -p "$workdir/run_outputs"
out="$workdir/run_outputs/create-venv-bender-out.$SLURM_JOB_ID"
err="$workdir/run_outputs/create-venv-bender-err.$SLURM_JOB_ID"

cd "$workdir"
ulimit -c 0

echo "# [${SLURM_JOB_ID}] Job started at: $(date)" | tee -a "$out"
echo "# [${SLURM_JOB_ID}] Hostname: $(hostname)" | tee -a "$out"
echo "# [${SLURM_JOB_ID}] Workdir: $workdir" | tee -a "$out"
echo "# [${SLURM_JOB_ID}] Venv: $workdir/$venv_name" | tee -a "$out"
echo "# [${SLURM_JOB_ID}] Stack: $LLM_FOUNDRY_STACK" | tee -a "$out"

#############################################
# Modules Setup
#############################################

echo "===== Setting up modules =====" | tee -a "$out"
source "$modules_file" 2>&1 | tee -a "$out"

echo "===== Environment Info =====" | tee -a "$out"
echo "  Python: $(which python3) — $(python3 --version)" | tee -a "$out"
echo "  CUDA_HOME: ${CUDA_HOME:-not set}" | tee -a "$out"
echo "  nvcc: $(which nvcc 2>/dev/null || echo 'not found') — $(nvcc --version 2>/dev/null | head -n1 || echo 'N/A')" | tee -a "$out"

#############################################
# Create venv
#############################################

venv_dir="$workdir/$venv_name"
echo "===== Creating venv at $venv_dir =====" | tee -a "$out"
python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"

echo "===== Upgrading pip =====" | tee -a "$out"
pip3 install --upgrade pip 2>&1 | tee -a "$out"

echo "===== Installing wheel + packaging =====" | tee -a "$out"
pip3 install wheel==0.45.1 packaging==25.0 --no-cache-dir 2>&1 | tee -a "$out"

echo "===== Installing PyTorch 2.6.0+cu124 =====" | tee -a "$out"
pip3 install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --no-cache-dir 2>&1 | tee -a "$out"

# Pin torch via a constraints file for every install that follows.
torch_constraints="/tmp/torch-constraints-$SLURM_JOB_ID.txt"
cat > "$torch_constraints" << 'EOF'
torch==2.6.0+cu124
EOF

echo "===== Installing llm-foundry [distributed] =====" | tee -a "$out"
pip3 install -e "$workdir/llm-foundry/.[distributed]" \
    --no-cache-dir -c "$torch_constraints" 2>&1 | tee -a "$out"

echo "===== Installing flash-attn 2.8.3 (prebuilt wheel) =====" | tee -a "$out"
# Prebuilt wheel for: flash-attn 2.8.3, CUDA 12.4, torch 2.6, Python 3.12.
# Find other wheels at: https://mjunya.com/flash-attention-prebuild-wheels/
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip3 install \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu124torch2.6-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl \
    --no-cache-dir 2>&1 | tee -a "$out"

# Alternative: build from source (uncomment below and comment out the wheel above).
#
#   FLASH_ATTENTION_FORCE_BUILD=TRUE \
#       MAX_JOBS=4 \
#       FLASH_ATTN_CUDA_ARCHS="$flash_attn_cuda_archs" \
#       pip3 install flash-attn==2.8.3 \
#       --no-binary :flash-attn: \
#       --no-build-isolation \
#       --no-cache-dir \
#       -c "$torch_constraints"

#############################################
# NOTES — packages NOT installed
#############################################
# flash-linear-attention:
#   Requires PyTorch >= 2.7.0.  Bender's newest CUDA is 12.4, which is
#   incompatible with official PyTorch 2.7+ release wheels.  Once a
#   compatible combination is available, add:
#
#       pip3 install flash-linear-attention --no-cache-dir
#
# causal-conv1d:
#
#       pip3 install ninja causal-conv1d --no-build-isolation --no-cache-dir

#############################################
# Verification
#############################################

echo "===== Verifying installation =====" | tee -a "$out"
python3 - <<'PY' 2>&1 | tee -a "$out"
from importlib.metadata import version
import torch

packages = [
    "torch", "torchvision", "torchaudio",
    "transformers", "datasets", "accelerate", "sentencepiece",
    "wandb", "pyyaml", "kernels", "liger-kernel",
    "flash-attn", "codecarbon", "trackio",
]
for pkg in packages:
    try:
        ver = version(pkg)
        tag = f" (runtime cuda={torch.version.cuda})" if pkg == "torch" else ""
        print(f"  {pkg}=={ver}{tag}")
    except Exception as e:
        print(f"  {pkg}: NOT FOUND ({e})")
PY

echo "===== Verifying flash-attn imports =====" | tee -a "$out"
python3 -c "from flash_attn import flash_attn_func; print('  flash_attn OK')" 2>&1 | tee -a "$out"

echo "===== Verifying GPU =====" | tee -a "$out"
python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0)}')" 2>&1 | tee -a "$out"

echo "===== Done =====" | tee -a "$out"
echo "Venv ready: $venv_dir" | tee -a "$out"
echo "Activate with: source $venv_dir/bin/activate" | tee -a "$out"
