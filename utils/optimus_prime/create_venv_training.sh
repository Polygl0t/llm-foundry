#!/bin/bash
# One-time setup: create venv tarball for distributed training.
# - Step 1: start an interactive job (adjust the resources as needed):
# >> condor_submit -interactive -append '+ContainerOS = "Rocky9"' -append '+CephFS_IO = "low"' -append '+MaxRuntimeHours=1' -append 'Request_gpus = 1' -append 'requirements = (CUDADeviceName == "NVIDIA H200")' -append 'Request_cpus = 1' -append 'Request_memory = 16000 MB'
# Step 2: Inside the the container,run this script
# >> bash /cephfs/user/<user-id>/llm-foundry/utils/optimus_prime/create_venv_training.sh


# If you want to create a venv on your $BUDDY or outside containers, then 
# 1. Change the paths below to point to your local $BUDDY and a local venv directory.
# 2. remove the code for creating a tarball and the cleanup code at the end of this script.
set -e

workdir="${BUDDY}"
venv_dir="/jwd/venv_ddp"
tarball="$workdir/venv_ddp.tar.gz"

echo "===== Setting up modules ====="
source "$workdir/llm-foundry/.modules.sh"

echo "===== Creating venv ====="
python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"

echo "===== Upgrading pip ====="
pip3 install --upgrade pip

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

echo "===== Verifying installation ====="
python3 - <<'PY'
from importlib.metadata import version
import torch

packages = [
    "torch", "transformers", "datasets", "accelerate", "sentencepiece",
    "wandb", "pyyaml", "kernels", "liger-kernel", "flash-attn-4",
    "flash-linear-attention", "causal-conv1d", "codecarbon",
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
tar czf "$tarball" venv_ddp

echo "===== Done ====="
echo "Tarball: $tarball"
ls -lh "$tarball"

deactivate 2>/dev/null || true
rm -rf "$venv_dir"
echo "Installation complete. The venv has been compressed into a tarball and saved to $tarball."