#!/usr/bin/env bash

#############################################
# Installation Script for JSC JUPITER (Stages/2026)
#############################################
# Run on a LOGIN node (jpbl-*). Do NOT submit as an sbatch job.
# This installer uses the JSC Stages/2026 toolchain:
#   Python 3.13, CUDA/13 nvcc, cuDNN/NCCL from the module stack.
#
# The JSC PyTorch module is currently torch 2.9.1+cu128 while nvcc is CUDA 13.
# We unload that module and build local PyTorch 2.9.1 against CUDA 13 so
# source-built extensions such as causal-conv1d see a matching CUDA pair.
# FlashAttention-4 with the CUDA 13 extra is used for H100.
#
# Usage:
#   bash jupiter_installation_2026.sh
#
# Optional knobs:
#   JUPITER_BUILD_PYTORCH=0                         # fail instead of building torch
#   MAX_JOBS=16 CMAKE_BUILD_PARALLEL_LEVEL=16       # PyTorch build parallelism
#   FLASH_ATTN_4_VERSION=4.0.0b13
#############################################

set -euo pipefail

workdir="/e/project1/polyglot/COMMON"
venv_dir="$workdir/.venv_distributed_2026"
modules_script="jupiter_modules_2026.sh"
pytorch_src_dir="$workdir/.src/pytorch-v2.9.1-cuda13"
flash_attn_4_version="${FLASH_ATTN_4_VERSION:-4.0.0b13}"

cd "$workdir"

require_login_node() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "ERROR: do NOT run this under sbatch/salloc." >&2
        echo "Run it directly on a login node: bash jupiter_installation_2026.sh" >&2
        exit 1
    fi
}

create_fresh_venv() {
    rm -rf "$venv_dir"
    python3 -m venv "$venv_dir"
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
    pip install --upgrade pip wheel setuptools
}

use_2026_cuda_toolchain_without_module_torch() {
    source "$workdir/$modules_script"
    module unload PyTorch || true
    module load CMake Ninja
}

get_nvcc_cuda() {
    nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1
}

torch_is_cuda13_build() {
    python3 - <<'PY'
import sys
import subprocess

try:
    import torch
except Exception:
    sys.exit(1)

version = torch.__version__.split("+")[0]
nvcc_output = subprocess.check_output(["nvcc", "--version"], text=True)
nvcc_major = "13" if "release 13." in nvcc_output else ""
torch_major = (torch.version.cuda or "").split(".", 1)[0]

if version.startswith("2.9.") and torch_major == nvcc_major and torch._C._GLIBCXX_USE_CXX11_ABI:
    print(f"OK torch={torch.__version__} cuda={torch.version.cuda}")
else:
    sys.exit(1)
PY
}

build_pytorch_291_cuda13() {
    if [[ "${JUPITER_BUILD_PYTORCH:-1}" != "1" ]]; then
        echo "ERROR: no CUDA 13 PyTorch 2.9 build is installed in $venv_dir." >&2
        echo "Set JUPITER_BUILD_PYTORCH=1 or install a suitable torch build manually." >&2
        exit 1
    fi

    mkdir -p "$(dirname "$pytorch_src_dir")"

    if [[ ! -d "$pytorch_src_dir/.git" ]]; then
        git clone --recursive --branch v2.9.1 --depth 1 https://github.com/pytorch/pytorch.git "$pytorch_src_dir"
    else
        git -C "$pytorch_src_dir" fetch --depth 1 origin v2.9.1
        git -C "$pytorch_src_dir" checkout v2.9.1
        git -C "$pytorch_src_dir" submodule sync --recursive
        git -C "$pytorch_src_dir" submodule update --init --recursive --depth 1
    fi

    pip install -r "$pytorch_src_dir/requirements.txt"

    pushd "$pytorch_src_dir" >/dev/null
    rm -rf dist build

    MAX_JOBS="${MAX_JOBS:-16}" \
    CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}" \
    USE_CUDA=1 \
    USE_CUDNN=1 \
    USE_NCCL=1 \
    USE_SYSTEM_NCCL=1 \
    USE_CXX11_ABI=1 \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" \
    PYTORCH_BUILD_VERSION=2.9.1 \
    PYTORCH_BUILD_NUMBER=1 \
        python3 setup.py bdist_wheel

    pip install --force-reinstall dist/torch-2.9.1-*.whl
    popd >/dev/null
}

ensure_pytorch() {
    if torch_is_cuda13_build; then
        return 0
    fi

    echo "No suitable CUDA 13 PyTorch 2.9 build found; building PyTorch locally."
    build_pytorch_291_cuda13
    torch_is_cuda13_build
}

install_distributed_requirements_without_torch() {
    local requirements_file
    requirements_file="$(mktemp)"

    python3 - <<'PY' > "$requirements_file"
import pathlib
import tomllib

config = tomllib.loads(pathlib.Path("llm-foundry/pyproject.toml").read_text())
skip = {"torch", "torchaudio", "torchvision"}

for requirement in config["project"]["optional-dependencies"]["distributed"]:
    name = requirement.split("[", 1)[0]
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        name = name.split(separator, 1)[0]
    normalized_name = name.strip().lower().replace("_", "-")
    if normalized_name not in skip:
        print(requirement)
PY

    pip install -r "$requirements_file"
    rm -f "$requirements_file"
    pip install -e "$workdir/llm-foundry" --no-deps
}

get_torch_cuda() {
    python3 - <<'PY'
import torch
print(torch.version.cuda or "")
PY
}

major_versions_match() {
    local left="$1"
    local right="$2"
    [[ -n "$left" && -n "$right" && "${left%%.*}" == "${right%%.*}" ]]
}

install_attention_stack() {
    local torch_cuda
    local nvcc_cuda

    torch_cuda="$(get_torch_cuda)"
    nvcc_cuda="$(get_nvcc_cuda)"

    pip install ninja packaging psutil
    pip install "flash-attn-4[cu13]==$flash_attn_4_version"
    pip install flash-linear-attention

    if major_versions_match "$torch_cuda" "$nvcc_cuda"; then
        MAX_JOBS=4 NVCC_THREADS=2 CAUSAL_CONV1D_FORCE_BUILD=TRUE \
            pip install causal-conv1d --no-build-isolation -v
    else
        echo "WARNING: skipping causal-conv1d." >&2
        echo "  torch CUDA: ${torch_cuda:-unknown}; nvcc CUDA: ${nvcc_cuda:-not found}." >&2
    fi
}

print_final_status() {
    python3 - <<'PY'
import importlib.util
import torch

checks = {
    "fla": "fla",
    "flash_attn_4": "flash_attn.cute",
    "causal_conv1d": "causal_conv1d",
}

print("=== Installation complete ===")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
for label, module_name in checks.items():
    status = "OK" if importlib.util.find_spec(module_name) else "not installed"
    print(f"{label}: {status}")
PY
}

require_login_node
use_2026_cuda_toolchain_without_module_torch
create_fresh_venv

# 1. PyTorch: build or reuse a local CUDA 13 build that matches nvcc.
ensure_pytorch

# 2. Project dependencies: keep home-cluster torch pins in pyproject.toml untouched.
install_distributed_requirements_without_torch

# 3. H100 attention stack for the 2026 module environment.
install_attention_stack

# And we are done!
print_final_status