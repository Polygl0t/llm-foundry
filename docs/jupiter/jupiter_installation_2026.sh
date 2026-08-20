#!/usr/bin/env bash

#############################################
# Installation Script for JSC JUPITER Stages/2026
#############################################
# Run on a LOGIN node (jpbl-*). Do NOT submit as an sbatch job.
#
# Uses the Stages/2026 module stack:
# - GCCcore/.14.3.0
# - Python/3.13.5
# - CUDA/13
# - CMake
# - Ninja
#
# Usage:
#    bash jupiter_installation_2026.sh
#############################################

set -euo pipefail

workdir="/e/project1/polyglot/COMMON"
venv_dir="$workdir/.venv_distributed"
modules_script="setup/jupiter_modules_2026.sh"


cd "$workdir"

# Suppress uv hardlink-fallback warnings: the venv and uv's cache live on
# different filesystems here, so hardlinking is not possible.
export UV_LINK_MODE=copy


require_login_node() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "ERROR: do NOT run this under sbatch/salloc."
        echo "Run it directly on a login node. E.g., bash jupiter_installation_2026.sh"
        exit 1
    fi
}


load_modules() {
    source "$workdir/$modules_script"
    echo "[$(date)] Loaded modules for JSC JUPITER Stages/2026:"
    module list
}


create_fresh_venv() {
    echo "[$(date)] Setting up Python virtual environment at $venv_dir..."
    rm -rf "$venv_dir"
    python3 -m venv "$venv_dir"
    source "$venv_dir/bin/activate"
    pip install --upgrade pip
    pip install uv
    uv pip install wheel packaging --no-cache

    # Ensure venv packages take precedence over system modules set via PYTHONPATH.
    _venv_site="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
    export PYTHONPATH="${_venv_site}${PYTHONPATH:+:$PYTHONPATH}"

    # Install pbr in the venv; the system pbr (6.1.1) still imports
    # 'pkg_resources' which was removed in Python 3.13.
    # We need this for causal-conv1d
    uv pip install --reinstall pbr --no-cache
}


install_core_packages() {
    echo "[$(date)] Installing PyTorch 2.13.0+cu130 (pinned)..."
    # Install torch first and explicitly so the rest of the stack binds to this version.
    uv pip install --reinstall --no-cache \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0+cu130

    echo "[$(date)] Installing core packages..."
    # --reinstall: force a fresh install into the venv, regardless of any
    # versions visible from the read-only cluster module paths.
    uv pip install --reinstall \
        "fsspec[http]==2025.3.0" \
        numpy==2.3.2 \
        transformers==5.14.0 \
        datasets==4.0.0 \
        sentencepiece==0.2.1 \
        accelerate==1.13.0 \
        codecarbon==3.2.9 \
        wandb==0.27.2 \
        trackio==0.32.2 \
        pyyaml==6.0.2 \
        liger-kernel==0.8.0 \
        kernels==0.13.0 \
        --no-cache
}


install_attention_stack() {
    echo "[$(date)] Installing attention stack (flash-attn-4, flash-linear-attention, causal-conv1d)..."
    uv pip install \
        "flash-attn-4[cu13]==4.0.0b15" \
        --pre \
        --no-cache

    uv pip install \
        flash-linear-attention \
        --no-cache

    # causal-conv1d needs nvcc + CUDA headers matching PyTorch's CUDA version.
    # The Stages/2026 `CUDA/13` module (loaded via jupiter_modules_2026.sh)
    # provides nvcc 13.0, which matches PyTorch's cu130 build.
    if ! command -v nvcc &>/dev/null; then
        echo "WARNING: nvcc not found on PATH; skipping causal-conv1d." >&2
        echo "         Ensure 'module load CUDA/13' succeeded." >&2
        return 0
    fi

    # CUDA_HOME is what torch.utils.cpp_extension consults; derive it from nvcc.
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

    nvcc_ver="$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')"
    torch_cuda="$(python3 -c 'import torch; print(torch.version.cuda)')"
    echo "[$(date)] nvcc=${nvcc_ver}  torch.version.cuda=${torch_cuda}  CUDA_HOME=${CUDA_HOME}"
    if [[ "${nvcc_ver%%.*}" != "${torch_cuda%%.*}" ]]; then
        echo "WARNING: CUDA major mismatch (nvcc=${nvcc_ver}, torch=${torch_cuda})." >&2
        echo "         causal-conv1d build will likely fail." >&2
    fi

    # causal-conv1d's setup.py hard-codes a long `cc_flag` list and ignores
    # TORCH_CUDA_ARCH_LIST, so it tries to build for ~9 architectures. On a
    # login node that gets OOM-killed.
    src_dir="$workdir/.build_causal_conv1d"
    rm -rf "$src_dir" && mkdir -p "$src_dir"
    pip download causal-conv1d \
        --no-binary=:all: --no-deps --no-build-isolation \
        -d "$src_dir"
    tar -xzf "$src_dir"/causal_conv1d-*.tar.gz -C "$src_dir"
    src="$(find "$src_dir" -maxdepth 1 -type d -name 'causal_conv1d-*' | head -n1)"
    [[ -n "$src" ]] || { echo "ERROR: causal-conv1d source not found in $src_dir" >&2; return 1; }

    # Workaround: patch out all cc_flag.append("-gencode") / cc_flag.append("arch=...") pairs
    # where the arch is NOT compute_90, replacing them with `pass` at the same indentation level.
    python3 - "$src/setup.py" <<'PY'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
# Replace each pair of cc_flag.append("-gencode") / cc_flag.append("arch=...")
# with `pass` (at the same indentation) unless the arch is compute_90.
# Using `pass` (not deletion) preserves the parent `if` block's body, since
# some pairs are the sole body of `if bare_metal_version >= Version("X"):`.
def _sub(m):
    indent = m.group(1)
    return f"\n{indent}pass"
new = re.sub(
    r'\n([ \t]+)cc_flag\.append\("-gencode"\)\s*\n[ \t]+cc_flag\.append\("arch=compute_(?!90,)[^"]+"\)',
    _sub,
    s,
)
if new == s:
    print("WARNING: arch patch did not match anything; setup.py layout may have changed.", file=sys.stderr)
p.write_text(new)
PY

    MAX_JOBS=2 NVCC_THREADS=1 CAUSAL_CONV1D_FORCE_BUILD=TRUE \
        uv pip install "$src" --no-build-isolation --no-cache -v || \
        echo "WARNING: causal-conv1d build failed (non-fatal)." >&2
}


install_torchao() {
    echo "[$(date)] Installing torchao..."
    # torchao powers the optional `fp8: true` mixed precision path in the
    # distributed trainers. The GH200 (Grace Hopper, compute capability 9.0)
    # nodes on JUPITER have hardware fp8 support.
    # Prefer the PyTorch cu130 index so the prebuilt kernels match the CUDA
    # version of the pinned torch wheel; fall back to PyPI otherwise.
    uv pip install --reinstall --no-cache \
        --index-url https://download.pytorch.org/whl/cu130 \
        torchao \
    || uv pip install --reinstall --no-cache torchao \
    || echo "WARNING: torchao install failed (non-fatal); fp8 training will be unavailable." >&2
}


print_final_status() {
    python3 - <<'PY'
import importlib.util
import torch

checks = {
    "transformers": "transformers",
    "datasets": "datasets",
    "sentencepiece": "sentencepiece",
    "accelerate": "accelerate",
    "codecarbon": "codecarbon",
    "wandb": "wandb",
    "liger-kernel": "liger_kernel",
    "flash_attn_4": "flash_attn.cute",
    "flash_linear_attention": "fla",
    "causal_conv1d": "causal_conv1d",
    "torchao": "torchao",
    "trackio": "trackio",
}

print("=== Installation complete ===")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
for label, module_name in checks.items():
    status = "OK" if importlib.util.find_spec(module_name) else "not installed"
    print(f"{label}: {status}")
PY
}


# ===== Main Installation Steps =====
echo "[$(date)] Starting installation..."
require_login_node
load_modules
create_fresh_venv


# 1. Core ML/data packages with pinned versions.
install_core_packages

# 2. Attention stack for GH200.
install_attention_stack

# 3. torchao for fp8 mixed precision training on GH200.
install_torchao

# And we are done!
print_final_status
echo "[$(date)] Installation finished."
