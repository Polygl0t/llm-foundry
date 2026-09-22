#!/usr/bin/env bash

#############################################
# Post-training (TRL) venv for JSC JUPITER Stages/2026
#############################################
# Creates .venv_trl: the environment that runs the POST-TRAINING part of the foundry.
#
#     alignment/sft_trainer.py      supervised fine-tuning         (TRL SFTTrainer)
#     alignment/dpo_trainer.py      direct preference optimisation (TRL DPOTrainer)
#     alignment/reward_trainer.py   reward modelling               (TRL RewardTrainer)
#     alignment/grpo_trainer.py     verifier-reward RL             (TRL GRPOTrainer)
#
# Run on a LOGIN node (jpbl-*).
#
# Usage:
#     bash jupiter_installation_trl_2026.sh
#     SKIP_INSTALL=1  bash jupiter_installation_trl_2026.sh   # re-verify only
#     RECREATE_VENV=0 bash jupiter_installation_trl_2026.sh   # keep the venv
#
# Environment overrides:
#     RECREATE_VENV=1        delete and rebuild .venv_trl                    (default 1)
#     SKIP_INSTALL=0         skip venv/packages, run the checks only         (default 0)
#     SKIP_VERIFY=0          skip the final verification block               (default 0)
#     WITH_PEFT=1            peft (LoRA plumbing; unused by the argparse)    (default 1)
#     WITH_FLASH_ATTN=1      flash-attn-4, for --attn_implementation
#                            flash_attention_4 (FA2/FA3 need other pkgs)     (default 1)
#     WITH_VLLM=0            vLLM, for grpo_trainer.py --use_vllm;
#                            CONFLICTS with WITH_FLASH_ATTN=1                (default 0)
#     WITH_TRACKIO=1         trackio, for --report_to trackio                (default 1)
#     WITH_HYBRID=1          flash-linear-attention + causal-conv1d, for
#                            post-training HYBRID (linear-attention / SSM)
#                            models such as Qwen3-Next or Falcon-H1         (default 1)
#     VENV_DIR=<path>        build somewhere other than $workdir/.venv_trl
#
# Environment variable defaults for the installation script.
#     TORCH_VERSION=2.13.0
#     TORCH_CUDA_TAG=cu130
#     TRANSFORMERS_VERSION=5.14.0
#     TRL_VERSION=1.13.0
#     VLLM_VERSION=0.28.0    (newest usable - see below)
#     DATASETS_VERSION=5.0.1
#     ACCELERATE_VERSION=1.13.0
#     PEFT_VERSION=0.21.0
#     LIGER_VERSION=0.8.3
#     TRACKIO_VERSION=0.38.1
#     FLASH_ATTN_VERSION=4.0.0b27
#     FLA_VERSION=0.5.2
#     CAUSAL_CONV1D_VERSION=1.7.0
#
#
# CAVEATS / THINGS TO KNOW
# ====================
#   * flash-attn-4 (`WITH_FLASH_ATTN=1`, THE DEFAULT) provides
#     `--attn_implementation flash_attention_4`. However, this is
#     INCOMPATIBLE with vLLM: flash-attn-4 requires
#     `apache-tvm-ffi>=0.1.12,<0.2` while every vLLM release (0.28.0, 0.29.0,
#     0.30.0) pins `apache-tvm-ffi==0.1.11`.
#
#   * vLLM (`WITH_VLLM=0`, NOT installed by default) is only needed by
#     `grpo_trainer.py --use_vllm`. For this, build a separate
#     venv with `WITH_VLLM=1 WITH_FLASH_ATTN=0`.
#
#   * trackio (`WITH_TRACKIO=1`, the default) gives `--report_to trackio`. Necessary
#     for experiment tracking on JSC (compute nodes have no internet access).
#
#   * flash-linear-attention + causal-conv1d (`WITH_HYBRID=1`, THE DEFAULT) add
#     post-training support for HYBRID models -- anything mixing gated
#     linear-attention and/or short convolutions with full attention.
#
#     NEITHER package touches `apache-tvm-ffi`, so this coexists with flash-attn-4
#     AND with vLLM. That pin comes only from `tilelang`, which is an OPTIONAL
#     extra of FLA -- so do NOT request `flash-linear-attention[tilelang]`. If this
#     build fails, rerun with `WITH_HYBRID=0`.
#
#   * peft (`WITH_PEFT=1`): the alignment argparse exposes no LoRA flags, so this
#     is plumbing only, installed because TRL expects it to be available.
#
#############################################

set -euo pipefail

#############################################
# Configurations (tweak these to your needs)
#############################################

workdir="/e/project1/polyglot/COMMON"
venv_dir="${VENV_DIR:-$workdir/.venv_trl}"
modules_script="setup/jupiter_modules_2026.sh"
alignment_dir="$workdir/llm-foundry/alignment"

# --- PyTorch and Transformers version control ---
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu130}"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.14.0}"
# vLLM pins the vision/audio pair exactly; keep the family one matched build.
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.28.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"

# --- TRL, vLLM, and related packages ---
TRL_VERSION="${TRL_VERSION:-1.13.0}"
VLLM_VERSION="${VLLM_VERSION:-0.28.0}"
DATASETS_VERSION="${DATASETS_VERSION:-5.0.1}"
PEFT_VERSION="${PEFT_VERSION:-0.21.0}"
LIGER_VERSION="${LIGER_VERSION:-0.8.3}"
TRACKIO_VERSION="${TRACKIO_VERSION:-0.38.1}"

# --- Accelerate and Flash Attention ---
ACCELERATE_VERSION="${ACCELERATE_VERSION:-1.13.0}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-4.0.0b27}"

# --- Hybrid (linear-attention / SSM) models ---
FLA_VERSION="${FLA_VERSION:-0.5.2}"
CAUSAL_CONV1D_VERSION="${CAUSAL_CONV1D_VERSION:-1.7.0}"

# --- Behaviour flags ---
RECREATE_VENV="${RECREATE_VENV:-1}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"
WITH_PEFT="${WITH_PEFT:-1}"
WITH_FLASH_ATTN="${WITH_FLASH_ATTN:-1}"
WITH_VLLM="${WITH_VLLM:-0}"
WITH_TRACKIO="${WITH_TRACKIO:-1}"
WITH_HYBRID="${WITH_HYBRID:-1}"

# The venv and uv's cache live on different filesystems here, so hardlinking is
# not possible; suppress uv's hardlink-fallback warnings.
export UV_LINK_MODE=copy

#############################################
# Helpers
#############################################

require_login_node() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "ERROR: do NOT run this under sbatch/salloc." >&2
        echo "Run it directly on a login node. E.g., bash jupiter_installation_trl_2026.sh" >&2
        exit 1
    fi
}

load_modules() {
    source "$workdir/$modules_script"
    echo "[$(date)] Loaded modules for JSC JUPITER Stages/2026:"
    module list 2>&1 | sed 's/^/    /'
}

drop_active_venv() {
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        echo "[$(date)] Deactivating the currently active venv: $VIRTUAL_ENV"
        deactivate 2>/dev/null || true
    fi
}

uvpip() {
    uv pip install --no-cache \
        --extra-index-url "$TORCH_INDEX" \
        --index-strategy unsafe-best-match \
        "$@"
}

#############################################
# venv
#############################################

create_fresh_venv() {
    drop_active_venv

    if [[ "$RECREATE_VENV" == "1" ]]; then
        echo "[$(date)] Recreating the Python virtual environment at $venv_dir..."
        rm -rf "$venv_dir"
    elif [[ -d "$venv_dir" ]]; then
        echo "[$(date)] Reusing the existing venv at $venv_dir (RECREATE_VENV=0)"
    else
        echo "[$(date)] RECREATE_VENV=0 but $venv_dir does not exist; creating it..."
    fi

    if [[ ! -d "$venv_dir" ]]; then
        python3 -m venv "$venv_dir"
    fi
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"

    pip install --upgrade pip
    pip install uv
    uv pip install wheel packaging --no-cache

    # Ensure venv packages take precedence over system modules set via PYTHONPATH.
    _venv_site="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
    export PYTHONPATH="${_venv_site}${PYTHONPATH:+:$PYTHONPATH}"

    uv pip install --reinstall pbr --no-cache

    echo "[$(date)] PYTHONPATH for job scripts: $PYTHONPATH"
}

#############################################
# PyTorch
#############################################

install_torch() {
    echo "[$(date)] Installing the torch family (torch ${TORCH_VERSION}+${TORCH_CUDA_TAG}, "
    echo "    torchvision ${TORCHVISION_VERSION}+${TORCH_CUDA_TAG}, torchaudio ${TORCHAUDIO_VERSION}+${TORCH_CUDA_TAG})..."
    uvpip \
        "torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}" \
        "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_TAG}" \
        "torchaudio==${TORCHAUDIO_VERSION}+${TORCH_CUDA_TAG}"
}

#############################################
# Core Installation
#############################################

install_core_packages() {
    echo "[$(date)] Installing the TRL post-training stack..."
    echo "    transformers==${TRANSFORMERS_VERSION}  trl==${TRL_VERSION}  datasets==${DATASETS_VERSION}"
    echo "    accelerate==${ACCELERATE_VERSION}      peft==${PEFT_VERSION} (WITH_PEFT=${WITH_PEFT})"

    uvpip \
        "fsspec[http]==2025.3.0" \
        numpy==2.3.2 \
        "transformers==${TRANSFORMERS_VERSION}" \
        "datasets==${DATASETS_VERSION}" \
        sentencepiece==0.2.1 \
        "accelerate==${ACCELERATE_VERSION}" \
        pyyaml==6.0.2 \
        "trl==${TRL_VERSION}" \
        "liger-kernel==${LIGER_VERSION}" \
        scikit-learn==1.9.1 \
        joblib==1.6.0

    if [[ "$WITH_PEFT" == "1" ]]; then
        uvpip "peft==${PEFT_VERSION}"
    fi
}

#############################################
# Reporting backends + the gym/verifier helpers
#############################################

install_optional_packages() {
    echo "[$(date)] Installing reporting backends and alignment/gym helpers..."

    uvpip codecarbon==3.2.9 wandb==0.27.2

    if [[ "$WITH_TRACKIO" == "1" ]]; then
        uvpip "trackio==${TRACKIO_VERSION}"
    else
        echo "[$(date)] Skipping trackio (WITH_TRACKIO=0) -> --report_to trackio unavailable"
    fi

    uvpip \
        pandas==3.0.5 \
        nltk==3.10.3 \
        langdetect==1.0.9 \
        immutabledict==4.3.1
}

#############################################
# vLLM (only consumed by grpo_trainer.py --use_vllm)
#############################################

install_vllm() {
    if [[ "$WITH_VLLM" != "1" ]]; then
        echo "[$(date)] Skipping vLLM (WITH_VLLM=$WITH_VLLM) -> grpo_trainer.py --use_vllm will not work"
        return 0
    fi

    echo "[$(date)] Installing vLLM ${VLLM_VERSION} (for grpo_trainer.py --use_vllm)..."
    uvpip "vllm==${VLLM_VERSION}"
}

#############################################
# flash-attn-4 — optional, and a vLLM trade-off
#############################################

install_flash_attn() {
    if [[ "$WITH_FLASH_ATTN" != "1" ]]; then
        echo "[$(date)] Skipping flash-attn-4 (WITH_FLASH_ATTN=$WITH_FLASH_ATTN)"
        return 0
    fi

    echo "[$(date)] Installing flash-attn-4 ${FLASH_ATTN_VERSION} (for --attn_implementation flash_attention_*)..."
    uvpip "flash-attn-4[cu13]==${FLASH_ATTN_VERSION}" --pre
}

#############################################
# Hybrid (linear-attention / SSM) models
#############################################

install_hybrid_models() {
    if [[ "$WITH_HYBRID" != "1" ]]; then
        echo "[$(date)] Skipping the hybrid stack (WITH_HYBRID=$WITH_HYBRID) -> linear-attention/SSM models will not run"
        return 0
    fi

    echo "[$(date)] Installing flash-linear-attention ${FLA_VERSION}..."
    uvpip "flash-linear-attention==${FLA_VERSION}"

    if ! command -v nvcc &>/dev/null; then
        echo "WARNING: nvcc not found on PATH; skipping causal-conv1d." >&2
        echo "         Ensure 'module load CUDA/13' succeeded." >&2
        return 0
    fi

    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

    local nvcc_ver torch_cuda
    nvcc_ver="$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')"
    torch_cuda="$(python3 -c 'import torch; print(torch.version.cuda)')"
    echo "[$(date)] nvcc=${nvcc_ver}  torch.version.cuda=${torch_cuda}  CUDA_HOME=${CUDA_HOME}"
    if [[ "${nvcc_ver%%.*}" != "${torch_cuda%%.*}" ]]; then
        echo "WARNING: CUDA major mismatch (nvcc=${nvcc_ver}, torch=${torch_cuda})." >&2
        echo "         The causal-conv1d build will likely fail." >&2
    fi

    local src_dir="$workdir/.build_causal_conv1d_trl"
    local src=""
    rm -rf "$src_dir" && mkdir -p "$src_dir"

    echo "[$(date)] Fetching the causal-conv1d ${CAUSAL_CONV1D_VERSION} sdist..."
    if ! pip download "causal-conv1d==${CAUSAL_CONV1D_VERSION}" \
            --no-binary=:all: --no-deps --no-build-isolation \
            -d "$src_dir"; then
        echo "WARNING: could not download the causal-conv1d sdist (non-fatal)." >&2
        rm -rf "$src_dir"
        return 0
    fi
    tar -xzf "$src_dir"/causal_conv1d-*.tar.gz -C "$src_dir"
    src="$(find "$src_dir" -maxdepth 1 -type d -name 'causal_conv1d-*' | head -n1)"
    if [[ -z "$src" ]]; then
        echo "WARNING: causal-conv1d source not found in $src_dir (non-fatal)." >&2
        rm -rf "$src_dir"
        return 0
    fi

    python3 - "$src/setup.py" <<'PY'
import pathlib, re, sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()


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

    echo "[$(date)] Building causal-conv1d ${CAUSAL_CONV1D_VERSION} for compute_90 only..."
    MAX_JOBS=2 NVCC_THREADS=1 CAUSAL_CONV1D_FORCE_BUILD=TRUE \
        uv pip install "$src" --no-build-isolation --no-cache -v || \
        echo "WARNING: causal-conv1d build FAILED (verification will report it)." >&2

    rm -rf "$src_dir"
}

#############################################
# Re-assert the torch build
#############################################

reassert_torch() {
    echo "[$(date)] Re-asserting the torch family..."
    uvpip \
        "torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}" \
        "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_TAG}" \
        "torchaudio==${TORCHAUDIO_VERSION}+${TORCH_CUDA_TAG}"
}

#############################################
# Verification
#############################################

verify_install() {
    echo "[$(date)] Verifying the post-training stack..."

    python3 - \
        "torch=${TORCH_VERSION}+${TORCH_CUDA_TAG}" \
        "transformers=${TRANSFORMERS_VERSION}" \
        "trl=${TRL_VERSION}" \
        "datasets=${DATASETS_VERSION}" \
        "vllm=${WITH_VLLM}" \
        "trackio=${WITH_TRACKIO}" \
        "peft=${WITH_PEFT}" \
        "flash_attn=${WITH_FLASH_ATTN}" \
        "hybrid=${WITH_HYBRID}" <<'PY'
import importlib.metadata as md
import importlib.util
import sys

import torch

opts = dict(a.split("=", 1) for a in sys.argv[1:])
bad = []

print(f"  cuda           {torch.version.cuda}")
actual = {
    "torch": torch.__version__,
    "transformers": md.version("transformers"),
    "trl": md.version("trl"),
    "datasets": md.version("datasets"),
}
for name in ("torch", "transformers", "trl", "datasets"):
    want, got = opts[name], actual[name]
    print(f"  {name:14s} {got:<18s} {'OK' if got == want else 'EXPECTED ' + want}")
    if got != want:
        bad.append(f"{name} is {got}, expected {want}")

required = {
    "accelerate": "accelerate",
    "sentencepiece": "sentencepiece",
    "liger_kernel": "liger_kernel",
    "sklearn": "sklearn",
    "joblib": "joblib",
    "pandas": "pandas",
    "nltk": "nltk",
    "langdetect": "langdetect",
    "immutabledict": "immutabledict",
    "codecarbon": "codecarbon",
    "wandb": "wandb",
}

if opts["vllm"] == "1":
    required["vllm"] = "vllm"
if opts["trackio"] == "1":
    required["trackio"] = "trackio"
if opts["peft"] == "1":
    required["peft"] = "peft"
if opts["flash_attn"] == "1":
    required["flash_attn"] = "flash_attn"
if opts["hybrid"] == "1":
    required["fla"] = "fla"
    required["causal_conv1d"] = "causal_conv1d"

for label, mod in required.items():
    ok = importlib.util.find_spec(mod) is not None
    print(f"  {label:14s} {'OK' if ok else 'MISSING'}")
    if not ok:
        hint = ""
        if mod == "causal_conv1d":
            hint = (
                " -- it is a source build: check nvcc / CUDA_HOME / the build log above,"
                " or rerun with WITH_HYBRID=0"
            )
        bad.append(f"required module {mod!r} is not importable{hint}")

for label, mod in {"flash_attn": "flash_attn", "vllm": "vllm",
                   "trackio": "trackio", "fla": "fla",
                   "causal_conv1d": "causal_conv1d"}.items():
    if label not in required:
        state = "OK" if importlib.util.find_spec(mod) else "not installed (by flag)"
        print(f"  {label:14s} {state}")

if bad:
    print("\n=== VERIFICATION FAILED ===", file=sys.stderr)
    for b in bad:
        print(f"    - {b}", file=sys.stderr)
    sys.exit(1)
print("  versions + modules: OK")
PY


    python3 - "$workdir" <<'PY'
import ast
import dataclasses
import inspect
import pathlib
import sys

workdir = pathlib.Path(sys.argv[1])
align = workdir / "llm-foundry" / "alignment"

import trl

print(f"  trl            {trl.__version__}")
print(f"  transformers   {__import__('transformers').__version__}")
print(f"  datasets       {__import__('datasets').__version__}")

targets = {
    "sft_trainer.py": ("SFTConfig", "SFTTrainer"),
    "dpo_trainer.py": ("DPOConfig", "DPOTrainer"),
}

failures = []

for script, (config_name, trainer_name) in targets.items():
    config_cls = getattr(trl, config_name, None)
    if config_cls is None:
        failures.append(f"trl.{config_name} does not exist in trl {trl.__version__}")
        continue
    if getattr(trl, trainer_name, None) is None:
        failures.append(f"trl.{trainer_name} does not exist in trl {trl.__version__}")

    # Accepted kwargs = dataclass fields, else the __init__ signature.
    if dataclasses.is_dataclass(config_cls):
        accepted = {f.name for f in dataclasses.fields(config_cls)}
        source = "dataclass fields"
    else:
        accepted = set(inspect.signature(config_cls).parameters)
        source = "signature"

    passed = set()
    path = align / script
    if not path.exists():
        failures.append(f"{path} not found")
        continue
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == config_name
        ):
            passed |= {kw.arg for kw in node.keywords if kw.arg}

    unknown = sorted(passed - accepted)
    status = "OK" if not unknown else f"UNKNOWN {unknown}"
    print(f"  {script}: {len(passed)} kwargs vs {config_name} ({source}) -> {status}")
    for kw in unknown:
        failures.append(f"{script} passes {kw!r}, not a {config_name} field in trl {trl.__version__}")

if failures:
    print("\n  === CONTRACT CHECK FAILED ===")
    for f in failures:
        print(f"    - {f}")
    print("    The trainers and the pinned TRL disagree - see MODERN STACK in the script header.")
    sys.exit(1)
print("  contract check: OK")
PY

    echo "[$(date)] Importing the trainer modules..."
    ( cd "$alignment_dir" && python3 -c "
import dpo_trainer, sft_trainer
print('  dpo_trainer.py + sft_trainer.py import cleanly')
" )

    echo "[$(date)] Verification passed."
}

#############################################
# Main
#############################################

echo "[$(date)] Starting the TRL post-training venv setup..."

require_login_node
load_modules

if [[ "$WITH_FLASH_ATTN" == "1" && "$WITH_VLLM" == "1" ]]; then
    cat >&2 <<'EOF'
ERROR: WITH_FLASH_ATTN=1 conflicts with WITH_VLLM=1.

    flash-attn-4 requires   apache-tvm-ffi>=0.1.12,<0.2
    every vLLM release pins apache-tvm-ffi==0.1.11

Pick one:

  * flash-attn-4, no vLLM   (the default)
        bash jupiter_installation_trl_2026.sh
      DPO/SFT use --attn_implementation flash_attention_4 --bf16;
      GRPO can then not use --use_vllm.

  * vLLM, no flash-attn-4
        WITH_VLLM=1 WITH_FLASH_ATTN=0 bash jupiter_installation_trl_2026.sh
      GRPO can use --use_vllm; DPO/SFT fall back to
      --attn_implementation sdpa (no extra package needed).
EOF
    exit 1
fi

if [[ "$SKIP_INSTALL" == "1" ]]; then
    echo "[$(date)] SKIP_INSTALL=1 -> activating the existing venv, skipping installs."
    drop_active_venv
    if [[ ! -d "$venv_dir" ]]; then
        echo "ERROR: SKIP_INSTALL=1 but $venv_dir does not exist." >&2
        exit 1
    fi
    source "$venv_dir/bin/activate"
    _venv_site="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
    export PYTHONPATH="${_venv_site}${PYTHONPATH:+:$PYTHONPATH}"
else
    create_fresh_venv
    install_torch
    install_core_packages
    install_optional_packages
    install_vllm
    install_flash_attn
    install_hybrid_models
    reassert_torch
fi

if [[ "$SKIP_VERIFY" == "1" ]]; then
    echo "[$(date)] SKIP_VERIFY=1 -> skipping the verification block."
else
    verify_install
fi

#############################################
# Done
#############################################

cat <<EOF

===== Done =====
Venv ready: $venv_dir
Activate:   source $venv_dir/bin/activate
PYTHONPATH: $PYTHONPATH

Run a post-training job with:

    source jupiter_modules_2026.sh
    source .venv_trl/bin/activate
    export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
    accelerate launch --config_file llm-foundry/alignment/configs/.ddp_config.yaml \\
        llm-foundry/alignment/dpo_trainer.py <args...>

Pinned: torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}  transformers==${TRANSFORMERS_VERSION}
        trl==${TRL_VERSION}  datasets==${DATASETS_VERSION}  accelerate==${ACCELERATE_VERSION}
        peft==${PEFT_VERSION}  liger==${LIGER_VERSION}
Hybrid: flash-linear-attention $([[ "$WITH_HYBRID" == 1 ]] && echo "${FLA_VERSION}" || echo "skipped (WITH_HYBRID=0)") + causal-conv1d $([[ "$WITH_HYBRID" == 1 ]] && echo "${CAUSAL_CONV1D_VERSION} (built for compute_90)" || echo "skipped (WITH_HYBRID=0)")
Reporting: wandb + codecarbon; trackio $([[ "$WITH_TRACKIO" == 1 ]] && echo "${TRACKIO_VERSION}" || echo "skipped (WITH_TRACKIO=0)")

Attention: use --attn_implementation sdpa (built into torch) or, with --bf16,
  --attn_implementation flash_attention_4. flash-attn-4 is \
$([[ "$WITH_FLASH_ATTN" == 1 ]] && echo "INSTALLED" || echo "NOT installed (WITH_FLASH_ATTN=0)"); it is a REWRITE with
  no v2 API, so flash_attention_2 / flash_attention_3 are NOT available (and the
  argparse help listing them is generic upstream text).
vLLM: $([[ "$WITH_VLLM" == 1 ]] && echo "installed (grpo_trainer.py --use_vllm works)" || echo "NOT installed (WITH_VLLM=0), so grpo_trainer.py --use_vllm will not work")
  vLLM and flash-attn-4 are mutually exclusive (apache-tvm-ffi); rebuild with
  WITH_VLLM=1 WITH_FLASH_ATTN=0 to swap.
EOF
echo "[$(date)] Setup finished."
