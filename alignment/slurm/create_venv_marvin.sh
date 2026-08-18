#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# One-time setup: submit this job to create a ready-to-use venv for
# distributed training on Marvin.
#
# Usage:
#   sbatch distributed/slurm/create_venv_marvin.sh
#
#############################################
#SBATCH --account=ag_bit_flek              # <-- Change to your SLURM account
#SBATCH --partition=mlgpu_devel            # <-- Change to your partition
#SBATCH --job-name=create-venv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --oversubscribe

set -e
#############################################
# Configuration — tweak these to your needs
#############################################

# Workspace root.
workdir="/lustre/mlnvme/data/polyglot"

# Name of the venv directory.
venv_name=".venv_trl"

# Path to the .modules file.
modules_file="$workdir/llm-foundry/.modules.sh"

# GPU architectures for flash-attn kernel compilation.
#   8.0 = NVIDIA A100 (Ampere)
#   8.6 = NVIDIA A40  (Ampere)
flash_attn_cuda_archs="8.0;8.6"

# Log file
mkdir -p "$workdir/run_outputs"
out="$workdir/run_outputs/create-venv-out.$SLURM_JOB_ID"

cd "$workdir"
ulimit -c 0

# Redirect ALL output (stdout + stderr) for the rest of the script to $out.
exec > "$out" 2>&1

echo "# [${SLURM_JOB_ID}] Job started at: $(date)"
echo "# [${SLURM_JOB_ID}] Hostname: $(hostname)"
echo "# [${SLURM_JOB_ID}] Workdir: $workdir"
echo "# [${SLURM_JOB_ID}] Venv: $workdir/$venv_name"

#############################################
# Modules Setup
#############################################

echo "===== Setting up modules ====="
export LLM_FOUNDRY_STACK=amd # On Marvin, GPU nodes require the AMD stack.
source "$modules_file"

echo "===== Environment Info ====="
echo "  Python: $(which python3) — $(python3 --version)"
echo "  CUDA_HOME: ${CUDA_HOME:-not set}"
echo "  nvcc: $(which nvcc 2>/dev/null || echo 'not found') — $(nvcc --version 2>/dev/null | head -n1 || echo 'N/A')"

#############################################
# Create venv
#############################################

venv_dir="$workdir/$venv_name"
echo "===== Creating venv at $venv_dir ====="
python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"

echo "===== Upgrading pip ====="
pip3 install --upgrade pip

echo "===== Installing uv ====="
pip3 install uv

echo "===== Installing wheel + packaging ====="
uv pip install wheel==0.45.1 packaging==25.0 --no-cache

echo "===== Installing PyTorch 2.13.0+cu126 (pinned) ====="
uv pip install --no-cache \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.13.0+cu126

# Pin torch via a constraints file for every install that follows.
torch_constraints="/tmp/torch-constraints-$SLURM_JOB_ID.txt"
cat > "$torch_constraints" << 'EOF'
torch==2.13.0+cu126
EOF

echo "===== Installing llm-foundry [distributed] ====="
uv pip install -e "$workdir/llm-foundry/.[distributed]" \
    --no-cache -c "$torch_constraints"

echo "===== Installing flash-attn 2.8.3 (prebuilt wheel) ====="
# Prebuilt wheel for: flash-attn 2.8.3, CUDA 12.6, torch 2.13, Python 3.12.
# Find other wheels at: https://mjunya.com/flash-attention-prebuild-wheels/
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE uv pip install \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3+cu126torch2.13-cp312-cp312-linux_x86_64.whl \
    --no-cache \
    -c "$torch_constraints"

# Alternative: build from source (uncomment below and comment out the wheel above).
#
#   FLASH_ATTENTION_FORCE_BUILD=TRUE \
#       MAX_JOBS=4 \
#       FLASH_ATTN_CUDA_ARCHS="$flash_attn_cuda_archs" \
#       uv pip install flash-attn==2.8.3 \
#       --no-binary :flash-attn: \
#       --no-build-isolation \
#       --no-cache \
#       -c "$torch_constraints"

echo "===== Installing flash-linear-attention ====="
uv pip install flash-linear-attention --no-cache \
    -c "$torch_constraints"

echo "===== Installing causal-conv1d ====="
uv pip install causal-conv1d --no-build-isolation --no-cache \
    -c "$torch_constraints"

rm -f "$torch_constraints"

#############################################
# Verification
#############################################

echo "===== Verifying installation ====="
python3 - <<'PY'
from importlib.metadata import version
import torch

packages = [
    "torch", "transformers", "datasets", "accelerate", "sentencepiece",
    "wandb", "pyyaml", "kernels", "liger-kernel",
    "flash-attn", "causal-conv1d", "flash-linear-attention",
    "codecarbon", "trackio",
]
for pkg in packages:
    try:
        ver = version(pkg)
        tag = f" (runtime cuda={torch.version.cuda})" if pkg == "torch" else ""
        print(f"  {pkg}=={ver}{tag}")
    except Exception as e:
        print(f"  {pkg}: NOT FOUND ({e})")
PY

echo "===== Verifying flash-attn imports ====="
python3 -c "from flash_attn import flash_attn_func; print('  flash_attn OK')"

echo "===== Verifying GPU ====="
python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0)}')"

echo "===== Done ====="
echo "Venv ready: $venv_dir"
echo "Activate with: source $venv_dir/bin/activate"
