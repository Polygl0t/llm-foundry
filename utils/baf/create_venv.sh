#!/bin/bash
# One-time setup: create venv tarball for distributed training.
# - Step 1: start an interactive job (adjust the resources as needed):
# >> condor_submit -interactive -append '+ContainerOS = "Rocky9"' -append '+CephFS_IO = "low"' -append '+MaxRuntimeHours=1' -append 'Request_gpus = 1' -append 'requirements = (CUDADeviceName == "NVIDIA H200")' -append 'Request_cpus = 1' -append 'Request_memory = 16000 MB'
# - Step 2: Inside the the container, run this script
# >> bash $BUDDY/create_venv.sh
# Note: You CANNOT access files in your home directory (`~`) from inside the container.
#       You can only access files in your CephFS directory (`$BUDDY`). Hence, you should make
#       sure that the `llm-foundry`, or any other thing you need to run in the container,
#       is in your CephFS directory (`$BUDDY`). Symlinks will not work.
set -e

workdir="$BUDDY"
venv_name=".venv"
venv_dir="/jwd/$venv_name"
tarball="$workdir/$venv_name.tar.gz"

echo "===== Setting up modules ====="
source "$workdir/llm-foundry/.modules.sh"

echo "===== Creating venv ====="
python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"

echo "===== Upgrading pip ====="
pip3 install --upgrade pip

echo "===== Installing PyTorch 2.13.0+cu126 (pinned) ====="
# Install torch first and explicitly so the rest of the stack binds to this
# version. Using cu126 (CUDA 12.6) because torch 2.12+ naturally requires
# cuda-bindings >=13.0.3, which is the same major version that the
# nvidia-cutlass-dsl-libs-cu13 stack expects.
pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.13.0+cu126

# Pin torch via a constraints file used by every install that follows.
# Without this, pip's resolver is free to silently upgrade torch to satisfy
# an unrelated package's dependency.
torch_constraints="/tmp/torch-constraints.txt"
cat > "$torch_constraints" << 'EOF'
torch==2.13.0+cu126
EOF

echo "===== Installing distributed dependencies from the foundry's pyproject.toml ====="
pip3 install --no-cache-dir -c "$torch_constraints" \
    "wheel==0.45.1" \
    "packaging==25.0" \
    "numpy==2.3.2" \
    "transformers==5.14.0" \
    "datasets==4.0.0" \
    "sentencepiece==0.2.0" \
    "accelerate==1.9.0" \
    "codecarbon==3.2.9" \
    "wandb==0.27.2" \
    "trackio==0.32.2" \
    "pyyaml==6.0.2" \
    "liger-kernel==0.8.0" \
    "kernels==0.13.0"

echo "===== Installing causal-conv1d ====="
export CUDA_HOME="/usr/local/cuda-12"
export TORCH_CUDA_ARCH_LIST="9.0"
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip3 install ninja causal-conv1d --no-build-isolation --no-cache-dir -c "$torch_constraints" || \
    echo "WARNING: causal-conv1d failed (non-fatal)."

echo "===== Installing flash-attn-4 (cutlass DSL stack) ====="
pip3 install \
    "nvidia-cutlass-dsl==4.5.2" \
    "nvidia-cutlass-dsl-libs-cu13==4.5.2" \
    --no-cache-dir \
    -c "$torch_constraints"
pip3 install "quack-kernels==0.5.0" --no-cache-dir -c "$torch_constraints"
cat > /tmp/cutlass.txt << 'EOF'
nvidia-cutlass-dsl==4.5.2
nvidia-cutlass-dsl-libs-cu13==4.5.2
nvidia-cutlass-dsl-libs-base==4.5.2
EOF
pip3 install flash-attn-4 --pre --no-cache-dir -c /tmp/cutlass.txt -c "$torch_constraints"
rm -f /tmp/cutlass.txt

echo "===== Installing flash-linear-attention ====="
pip3 install flash-linear-attention --no-cache-dir -c "$torch_constraints"

echo "===== Installing torchao ====="
# torchao powers the optional `fp8: true` mixed precision path in the
# distributed trainers. The H100 / H200 GPUs on this cluster have hardware
# fp8 support (compute capability 9.0). Pull from the PyTorch cu126 index so
# the prebuilt kernels match the CUDA version of the installed torch wheel,
# falling back to PyPI if that index has no matching wheel.
pip3 install torchao --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir -c "$torch_constraints" || \
    pip3 install torchao --no-cache-dir -c "$torch_constraints" || \
    echo "WARNING: torchao failed (non-fatal). fp8 training will be unavailable."

rm -f "$torch_constraints"

echo "===== Verifying installation ====="
python3 - <<'PY'
from importlib.metadata import version
import torch

packages = [
    "torch", "transformers", "datasets", "accelerate", "sentencepiece",
    "wandb", "pyyaml", "kernels", "liger-kernel", "flash-attn-4",
    "nvidia-cutlass-dsl", "quack-kernels",
    "flash-linear-attention", "causal-conv1d", "codecarbon", "torchao",
    "trackio",
]
for pkg in packages:
    try:
        ver = version(pkg)
        tag = f" (runtime cuda={torch.version.cuda})" if pkg == "torch" else ""
        print(f"  {pkg}=={ver}{tag}")
    except Exception as e:
        print(f"  {pkg}: NOT FOUND ({e})")
PY

echo "===== Verifying flash-attn-4 imports ====="
python3 -c "from flash_attn.cute import flash_attn_func, flash_attn_varlen_func; print('  flash_attn.cute OK')"

echo "===== Creating tarball ====="
cd /jwd
tar czf "$tarball" "$venv_name"

echo "===== Done ====="
echo "Tarball: $tarball"
ls -lh "$tarball"

deactivate 2>/dev/null || true
rm -rf "$venv_dir"
echo "Installation complete. The venv has been compressed into a tarball and saved to $tarball."
