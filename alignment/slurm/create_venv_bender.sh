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

# Post-training stack (installed further down):
#   torch 2.6.0+cu124 (fixed by this cluster's CUDA 12.4)  transformers 5.14.0
#   trl 1.13.0   NO vLLM (see the TRL install step: vLLM >=0.19.1 requires torch >=2.11,
#   while CUDA 12.4 caps torch at 2.6.0)

# Stack override (remove or change if needed).
export LLM_FOUNDRY_STACK=intel

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
echo "# [${SLURM_JOB_ID}] Stack: $LLM_FOUNDRY_STACK"

#############################################
# Modules Setup
#############################################

echo "===== Setting up modules ====="
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

echo "===== Installing PyTorch 2.6.0+cu124 ====="
uv pip install \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --no-cache

# Pin torch via a constraints file for every install that follows.
torch_constraints="/tmp/torch-constraints-$SLURM_JOB_ID.txt"
cat > "$torch_constraints" << 'EOF'
torch==2.6.0+cu124
EOF

echo "===== Installing flash-attn 2.8.3 (prebuilt wheel) ====="
# Prebuilt wheel for: flash-attn 2.8.3, CUDA 12.4, torch 2.6, Python 3.12.
# Find other wheels at: https://mjunya.com/flash-attention-prebuild-wheels/
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE uv pip install \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu124torch2.6-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl \
    --no-cache

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

#############################################
# NOTES — packages NOT installed
#############################################
# flash-linear-attention:
#   Requires PyTorch >= 2.7.0.  Bender's newest CUDA is 12.4, which is
#   incompatible with official PyTorch 2.7+ release wheels.  Once a
#   compatible combination is available, add:
#
#       uv pip install flash-linear-attention --no-cache
#
# causal-conv1d:
#
#       uv pip install ninja causal-conv1d --no-build-isolation --no-cache

echo "===== Installing logging/reporting extras ====="
uv pip install wandb trackio codecarbon --no-cache \
    -c "$torch_constraints"

echo "===== Installing Liger-Kernel ====="
# Fused kernels for faster, memory-efficient training.
# trl 1.13.0 requires liger-kernel>=0.8.2 for `use_liger_kernel`.
uv pip install liger-kernel==0.8.3 --no-cache \
    -c "$torch_constraints"

echo "===== Installing TRL (newest that runs on torch 2.6.0) ====="
# vLLM is NOT INSTALLABLE on this stack:
#   * trl's `[vllm]` extra requires vllm>=0.19.1
#   * every vllm>=0.19.1 pins torch>=2.11 (0.19.1 -> 2.11.0, 0.20-0.26 -> 2.11.0,
#     0.27-0.30 -> 2.13.0)
#   * CUDA 12.4 caps torch at 2.6.0, whose newest vllm is 0.8.5 -- below trl's floor
# Only `grpo_trainer.py --use_vllm` needs vLLM; SFT/DPO/reward run without it.
uv pip install --no-cache -c "$torch_constraints" \
    "trl==1.13.0" \
    "transformers==5.14.0" \
    "datasets==5.0.1" \
    "accelerate==1.13.0"

echo "===== Installing alignment/gym dependencies ====="
uv pip install --no-cache -c "$torch_constraints" \
    nltk==3.10.3 \
    langdetect==1.0.9 \
    immutabledict==4.3.1

echo "===== Downloading the NLTK data the gym verifiers need ====="
export NLTK_DATA="${NLTK_DATA:-$HOME/nltk_data}"
mkdir -p "$NLTK_DATA"
python3 - <<'PY'
import nltk

print("  nltk.download('punkt_tab') ->", nltk.download("punkt_tab"))
PY

rm -f "$torch_constraints"

#############################################
# Verification
#############################################

echo "===== Verifying installation ====="
python3 - <<'PY'
from importlib.metadata import version
import torch

packages = [
    "torch", "trl", "transformers", "datasets", "accelerate",
    "peft", "sentencepiece", "wandb", "pyyaml", "liger-kernel",
    "flash-attn", "codecarbon", "trackio",
    # alignment/gym dependencies
    "nltk", "langdetect", "immutabledict",
]
for pkg in packages:
    try:
        ver = version(pkg)
        tag = f" (runtime cuda={torch.version.cuda})" if pkg == "torch" else ""
        print(f"  {pkg}=={ver}{tag}")
    except Exception as e:
        print(f"  {pkg}: NOT FOUND ({e})")
PY

echo "===== Verifying trl import ====="
python3 -c "import trl; print(f'  trl {trl.__version__} OK')"

echo "===== Verifying flash-attn import ====="
python3 -c "from flash_attn import flash_attn_func; print('  flash_attn OK')"

echo "===== Verifying the alignment gym stack (GRPO reward functions) ====="
(cd "$workdir/llm-foundry/alignment" && python3 -c "
import nltk.data
nltk.data.find('tokenizers/punkt_tab')
import gym.verifier
from gym import utils
assert utils.count_sentences('Uma frase. E outra!') == 2
print('  gym.verifier OK, and the Portuguese punkt_tab data resolves offline')
")

echo "===== Verifying GPU ====="
python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0)}')"

echo "===== Done ====="
echo "Venv ready: $venv_dir"
echo "Activate with: source $venv_dir/bin/activate"
