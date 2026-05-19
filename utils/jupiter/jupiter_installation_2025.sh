#!/usr/bin/env bash

#############################################
# Installation Script for JSC JUPITER (Stages/2025)
#############################################
# Run on a LOGIN node (jpbl-*). Do NOT submit as an sbatch job:
#   - login nodes have internet
#   - login nodes have nvcc via the CUDA module
#   - a GPU is not needed to compile PyTorch or CUDA extensions
#
# Usage:
#   bash jupiter_installation_2025.sh
#
# Optional knobs:
#   JUPITER_MODULES_SCRIPT=jupiter_modules_2025.sh  # default
#   JUPITER_BUILD_PYTORCH=0                         # fail instead of building torch
#   MAX_JOBS=16 CMAKE_BUILD_PARALLEL_LEVEL=16       # PyTorch build parallelism
#############################################

set -euo pipefail

workdir="/e/project1/polyglot/COMMON"
venv_dir="$workdir/.venv_distributed_2025"
modules_script="${JUPITER_MODULES_SCRIPT:-jupiter_modules_2025.sh}"
pytorch_src_dir="$workdir/.src/pytorch-v2.9.1"
flash_attn_aarch64_wheel="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_aarch64.whl"

cd "$workdir"

require_login_node() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "ERROR: do NOT run this under sbatch/salloc." >&2
        echo "Run it directly on a login node: bash jupiter_installation_2025.sh" >&2
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

torch_is_jupiter_cuda_build() {
    python3 - <<'PY'
import sys

try:
    import torch
except Exception:
    sys.exit(1)

version = torch.__version__.split("+")[0]
if version.startswith("2.9.") and torch.version.cuda and torch._C._GLIBCXX_USE_CXX11_ABI:
    print(f"OK torch={torch.__version__} cuda={torch.version.cuda}")
else:
    sys.exit(1)
PY
}

build_pytorch_291() {
    if [[ "${JUPITER_BUILD_PYTORCH:-1}" != "1" ]]; then
        echo "ERROR: no CUDA-enabled PyTorch 2.9 build is installed in $venv_dir." >&2
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
    if torch_is_jupiter_cuda_build; then
        return 0
    fi

    echo "No suitable CUDA PyTorch 2.9 build found; building PyTorch locally."
    build_pytorch_291
    torch_is_jupiter_cuda_build
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

get_nvcc_cuda() {
    nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n 1
}

major_versions_match() {
    local left="$1"
    local right="$2"
    [[ -n "$left" && -n "$right" && "${left%%.*}" == "${right%%.*}" ]]
}

can_use_flash_attn_wheel() {
    python3 - <<'PY'
import platform
import sys
import torch

python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
torch_version = ".".join(torch.__version__.split("+")[0].split(".")[:2])
cxx11_abi = torch._C._GLIBCXX_USE_CXX11_ABI

if platform.machine() == "aarch64" and python_tag == "cp312" and torch_version == "2.9" and cxx11_abi:
    sys.exit(0)
sys.exit(1)
PY
}

install_cuda_extensions() {
    local torch_cuda
    local nvcc_cuda

    torch_cuda="$(get_torch_cuda)"
    nvcc_cuda="$(get_nvcc_cuda)"

    pip install ninja

    if can_use_flash_attn_wheel; then
        pip install "$flash_attn_aarch64_wheel" --no-cache-dir
    elif major_versions_match "$torch_cuda" "$nvcc_cuda"; then
        MAX_JOBS=4 NVCC_THREADS=2 FLASH_ATTENTION_FORCE_BUILD=TRUE \
            pip install flash-attn==2.8.3 --no-build-isolation -v
    else
        echo "WARNING: skipping flash-attn." >&2
        echo "  torch CUDA: ${torch_cuda:-unknown}; nvcc CUDA: ${nvcc_cuda:-not found}." >&2
    fi

    pip install flash-linear-attention

    if major_versions_match "$torch_cuda" "$nvcc_cuda"; then
        MAX_JOBS=4 NVCC_THREADS=2 CAUSAL_CONV1D_FORCE_BUILD=TRUE \
            pip install causal-conv1d --no-build-isolation -v
    else
        echo "WARNING: skipping causal-conv1d because nvcc does not match PyTorch CUDA." >&2
    fi
}

print_final_status() {
    python3 - <<'PY'
import importlib.util
import torch

print("=== Installation complete ===")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
for module_name in ("fla", "flash_attn", "causal_conv1d"):
    status = "OK" if importlib.util.find_spec(module_name) else "not installed"
    print(f"{module_name}: {status}")
PY
}

require_login_node
source "$workdir/$modules_script"
create_fresh_venv

# 1. PyTorch: use an existing JUPITER CUDA build or build PyTorch 2.9.1 locally.
ensure_pytorch

# 2. Project dependencies: keep home-cluster torch pins in pyproject.toml untouched.
install_distributed_requirements_without_torch

# 3. Attention/convolution packages selected for the active JUPITER stack.
install_cuda_extensions

# And we are done!
print_final_status
