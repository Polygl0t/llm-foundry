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

echo "===== Installing PyTorch 2.12.0+cu128 (pinned) ====="
# Install torch first and explicitly so the rest of the stack binds to this
# version (torchao requires torch >= 2.11.0). The H200 nodes use CUDA 12, so
# pull from the cu128 index.
pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.12.0+cu128

echo "===== Installing LLM Foundry ====="
pip3 install -e "$workdir/llm-foundry/.[distributed]" --no-cache-dir

echo "===== Installing causal-conv1d ====="
export CUDA_HOME="/usr/local/cuda-12"
export TORCH_CUDA_ARCH_LIST="9.0"
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip3 install ninja causal-conv1d --no-build-isolation --no-cache-dir || \
    echo "WARNING: causal-conv1d failed (non-fatal)."

echo "===== Installing flash-attn-4  ====="
pip3 install \
    "nvidia-cutlass-dsl==4.5.2" \
    "nvidia-cutlass-dsl-libs-cu13==4.5.2" \
    --no-cache-dir
pip3 install "quack-kernels==0.5.0" --no-cache-dir
cat > /tmp/cutlass.txt << 'EOF'
nvidia-cutlass-dsl==4.5.2
nvidia-cutlass-dsl-libs-cu13==4.5.2
nvidia-cutlass-dsl-libs-base==4.5.2
EOF
pip3 install flash-attn-4 --pre --no-cache-dir -c /tmp/cutlass.txt
rm -f /tmp/cutlass.txt

echo "===== Installing flash-linear-attention ====="
pip3 install flash-linear-attention --no-cache-dir

echo "===== Installing torchao ====="
# torchao powers the optional `fp8: true` mixed precision path in the
# distributed trainers. The H100 / H200 GPUs on this cluster have hardware
# fp8 support (compute capability 9.0). Pull from the PyTorch cu128 index so
# the prebuilt kernels match the CUDA version of the installed torch wheel,
# falling back to PyPI if that index has no matching wheel.
pip3 install torchao --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir || \
    pip3 install torchao --no-cache-dir || \
    echo "WARNING: torchao failed (non-fatal). fp8 training will be unavailable."

echo "===== Verifying installation ====="
python3 - <<'PY'
from importlib.metadata import version
import torch

packages = [
    "torch", "transformers", "datasets", "accelerate", "sentencepiece",
    "wandb", "pyyaml", "kernels", "liger-kernel", "flash-attn-4",
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

echo "===== Creating tarball ====="
cd /jwd
tar czf "$tarball" "$venv_name"

echo "===== Done ====="
echo "Tarball: $tarball"
ls -lh "$tarball"

deactivate 2>/dev/null || true
rm -rf "$venv_dir"
echo "Installation complete. The venv has been compressed into a tarball and saved to $tarball."
