#!/usr/bin/env bash

#############################################
# lm-evaluation-harness setup for JSC JUPITER Stages/2026:
# install the harness AND pre-cache every task's dataset.
#############################################
# Run on a LOGIN node (jpbl-*)
#
# Usage:
#     bash jupiter_installation_eval_harness_2026.sh   # full install + pre-cache every task
#     SKIP_INSTALL=1 bash ...                          # re-pre-cache everything, no reinstall
#     bash ... mmlu_poly_pt calame_pt                  # pre-cache ONLY these (implies SKIP_INSTALL)
#     ENSURE_EXTRAS=1 bash ...                         # add the ruler/ifeval/math extras
#     SIMULATE_OFFLINE=1 bash ...                      # prove the caches work with no network
#
# It creates three things:
#
#     .venv_eval/                    Python environment for the eval harness
#     lm_evaluation_harness/         the harness source (editable install)
#     .cache/eval_harness/           HF caches, NLTK data, RULER haystack
#
#
# Environment overrides:
#     RECREATE_VENV=1       delete and rebuild .venv_eval             (default 0)
#     FORCE_CLONE=1         delete and re-clone the harness source    (default 0)
#     WITH_VLLM=0           skip the optional vLLM backend            (default 1)
#     SKIP_INSTALL=1        skip clone/venv/install, only pre-cache    (default 0)
#     SKIP_PRECACHE=1       skip the dataset pre-cache + smoke test   (default 0)
#     LIMIT=1               docs per task in the smoke test           (default 1)
#     GEN_KWARGS="max_gen_toks=16"
#                           cap generated tokens; "" = task defaults
#     DEVICE=cpu            login nodes have no GPU
#     MODEL=...             local HF folder, used only to instantiate a model
#     ENSURE_EXTRAS=1       also install the ruler/ifeval/math extras
#     SIMULATE_OFFLINE=1    blackhole outbound HTTP(S) during the smoke test
#############################################

set -euo pipefail

#############################################
# Environment setup
#############################################

workdir="/e/project1/polyglot/COMMON"
venv_dir="$workdir/.venv_eval"
venv_py="$venv_dir/bin/python"
harness_dir="$workdir/lm_evaluation_harness"
cache_dir="$workdir/.cache/eval_harness"

modules_script="setup/jupiter_modules_2026.sh"

repo_url="https://github.com/Polygl0t/lm-evaluation-harness.git"
repo_branch="polyglot_harness_portuguese"

# Datasets pre-cached when no task names are given on the command line.
precache_tasks=(
    arc_challenge_poly_pt
    mmlu_poly_pt
    hellaswag_poly_pt
    lambada_poly_pt
    calame_pt
    belebele_por_Latn
    global_piqa_nonparallel_cloze_por_latn_braz
    global_piqa_parallel_cloze_por_latn_braz
    bluex_cloze
    enem_cloze
    oab_exams_cloze
    ifeval_pt
    gsm8k_pt
    ruler_pt
)

# Task names on the command line mean "pre-cache only": adding a dataset must
# never drag the clone/venv/reinstall steps along with it.
if [[ "$#" -gt 0 ]]; then
    tasks=("$@")
    SKIP_INSTALL=1
else
    tasks=("${precache_tasks[@]}")
fi
SKIP_INSTALL="${SKIP_INSTALL:-0}"

RECREATE_VENV="${RECREATE_VENV:-0}"
FORCE_CLONE="${FORCE_CLONE:-0}"
WITH_VLLM="${WITH_VLLM:-1}"
SKIP_PRECACHE="${SKIP_PRECACHE:-0}"
ENSURE_EXTRAS="${ENSURE_EXTRAS:-0}"
SIMULATE_OFFLINE="${SIMULATE_OFFLINE:-0}"

# Smoke-test knobs.
LIMIT="${LIMIT:-${PRECACHE_LIMIT:-1}}"
DEVICE="${DEVICE:-${PRECACHE_DEVICE:-cpu}}"
MODEL="${MODEL:-$workdir/checkpoints/portuguese/Tucano2-qwen-0.5B-Base}"
GEN_KWARGS="${GEN_KWARGS-${PRECACHE_GEN_KWARGS-max_gen_toks=16}}"

export UV_LINK_MODE=copy
export HF_DATASETS_CACHE="$cache_dir/datasets"
export HF_HUB_CACHE="$cache_dir/models"

# NLTK corpora and the RULER haystack live in their own directories, NOT in the
# HF cache, so the eval job must export these too.
# ruler_patch.sh is a shared helper: it downloads the RULER
# haystack and patches the harness to read it instead of fetching it over HTTP
# at dataset-generation time.
source "$workdir/setup/ruler_patch.sh"
export NLTK_DATA="$cache_dir/nltk_data"
export RULER_HAYSTACK_DIR="$cache_dir/$RULER_HAYSTACK_SUBDIR"


#############################################
# Helpers
#############################################

require_login_node() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "ERROR: do NOT run this under sbatch/salloc." >&2
        echo "Run it directly on a login node. E.g., bash jupiter_installation_eval_harness_2026.sh" >&2
        exit 1
    fi
}

uvpip() {
    uv pip install --python "$venv_dir/bin/python" "$@"
}

load_modules() {
    # shellcheck disable=SC1091
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


#############################################
# Step 1 -- clone (or update) the harness
#############################################

clone_harness() {
    if [[ -d "$harness_dir/.git" ]]; then
        if [[ "$FORCE_CLONE" == "1" ]]; then
            echo "[$(date)] FORCE_CLONE=1 -> removing $harness_dir"
            rm -rf "$harness_dir"
        else
            echo "[$(date)] Harness already cloned; updating to origin/$repo_branch ..."
            git -C "$harness_dir" fetch --depth 1 origin "$repo_branch"
            git -C "$harness_dir" checkout --quiet "$repo_branch"
            git -C "$harness_dir" reset --hard --quiet "origin/$repo_branch"
            echo "[$(date)] Harness at $(git -C "$harness_dir" rev-parse --short HEAD) ($repo_branch)"
            return 0
        fi
    elif [[ -e "$harness_dir" ]]; then
        # Present but not a git checkout -- refuse to delete it silently.
        echo "ERROR: $harness_dir exists but is not a git checkout." >&2
        echo "       Move it away, or re-run with FORCE_CLONE=1 to delete it." >&2
        exit 1
    fi

    echo "[$(date)] Cloning $repo_url ($repo_branch) -> $harness_dir"
    git clone --branch "$repo_branch" --depth 1 "$repo_url" "$harness_dir"
    echo "[$(date)] Harness at $(git -C "$harness_dir" rev-parse --short HEAD) ($repo_branch)"
}


#############################################
# Step 2 -- create the virtual environment
#############################################

create_venv() {
    if [[ -d "$venv_dir" && "$RECREATE_VENV" != "1" ]]; then
        echo "[$(date)] Reusing existing venv: $venv_dir"
        echo "             (RECREATE_VENV=1 to delete and rebuild)"
    else
        echo "[$(date)] Creating venv: $venv_dir"
        rm -rf "$venv_dir"
        python3 -m venv "$venv_dir"
    fi

    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
    echo "[$(date)] Interpreter: $(python3 -c 'import sys; print(sys.executable)')"

    pip install --quiet --upgrade pip
    pip install --quiet uv
}


#############################################
# Step 3 -- install the harness
#############################################

install_harness() {
    echo "[$(date)] Installing PyTorch 2.13.0+cu130 (pinned)..."
    uvpip --reinstall --no-cache \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0+cu130

    echo "[$(date)] Installing lm_eval[hf,ifeval] + pyyaml (editable, from $harness_dir)..."
    uvpip -e "$harness_dir[hf,ifeval]" --no-cache
    uvpip pyyaml --no-cache

    local _venv_site
    _venv_site="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
    export PYTHONPATH="${_venv_site}${PYTHONPATH:+:$PYTHONPATH}"
}

maybe_install_vllm() {
    if [[ "$WITH_VLLM" != "1" ]]; then
        echo "[$(date)] WITH_VLLM=$WITH_VLLM -> skipping the vLLM backend."
        return 0
    fi

    echo "[$(date)] Installing the vLLM backend (optional; failure is non-fatal)..."
    if uvpip "vllm>=0.18" --no-cache; then
        echo "[$(date)] vLLM installed. The eval script can now use MODE=\"vllm\"."
    else
        echo "WARNING: vLLM install failed -- non-fatal." >&2
        echo "         Keep MODE=\"hf\" in scripts/portuguese/eval_harness_pt_booster.sh." >&2
    fi
}


#############################################
# Step 4 -- NLTK + pre-cache datasets + smoke test
#############################################

ensure_nltk() {
    echo "[$(date)] Ensuring punkt_tab is in $NLTK_DATA"
    "$venv_py" - <<'PY' || echo "WARNING: NLTK setup failed" >&2
import os
import nltk

target = os.environ["NLTK_DATA"]
for pkg in ("punkt_tab",):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
        print(f"  {pkg}: already present")
    except LookupError:
        print(f"  {pkg}: downloading into {target} ...", end=" ", flush=True)
        print("OK" if nltk.download(pkg, download_dir=target, quiet=True) else "FAILED")
PY
}

precache_datasets() {
    if [[ "$SKIP_PRECACHE" == "1" ]]; then
        echo "[$(date)] SKIP_PRECACHE=1 -> skipping dataset pre-cache and smoke test."
        echo "             NOTE: the eval job runs offline; uncached tasks WILL fail."
        return 0
    fi

    if [[ ! -d "$MODEL" ]]; then
        echo "ERROR: smoke-test model not found: $MODEL" >&2
        echo "       Set MODEL to any local HF-format folder." >&2
        exit 1
    fi

    mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$NLTK_DATA" "$RULER_HAYSTACK_DIR"

    apply_ruler_haystack_patch "$harness_dir" || echo "WARNING: RULER patch failed" >&2
    ensure_ruler_haystack_file "$cache_dir" "$harness_dir" || echo "WARNING: RULER haystack download failed" >&2
    ensure_nltk

    if [[ "$ENSURE_EXTRAS" == "1" ]]; then
        echo "[$(date)] Installing the ruler/ifeval/math extras into the fork..."
        uvpip -e "$harness_dir[ruler,ifeval,math]" --no-cache
    fi

    local gen_args=()
    if [[ -n "$GEN_KWARGS" ]]; then
        gen_args=(--gen_kwargs "$GEN_KWARGS")
    fi

    local tmp
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/harness-precache-XXXXXX")"
    local failures=() ok=0

    echo ""
    echo "=============================================="
    echo "Dataset pre-cache + smoke test"
    echo "  tasks      : ${#tasks[@]}"
    echo "  model      : $MODEL"
    echo "  device     : $DEVICE (login nodes have no GPU)"
    echo "  limit      : $LIMIT docs per task"
    echo "  gen_kwargs : ${GEN_KWARGS:-<task defaults>}"
    echo "  datasets   : $HF_DATASETS_CACHE"
    echo "=============================================="

    if [[ "$SIMULATE_OFFLINE" == "1" ]]; then
        echo "[offline] blackholing outbound HTTP(S) for this smoke test"
        export http_proxy="http://127.0.0.1:9" https_proxy="http://127.0.0.1:9"
        export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
        export no_proxy="" NO_PROXY=""
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
    fi

    for t in "${tasks[@]}"; do
        rm -rf "$tmp/$t"; mkdir -p "$tmp/$t"
        printf '  %-46s ' "$t"
        if "$venv_py" -m lm_eval \
                --model hf \
                --model_args "pretrained=$MODEL" \
                --tasks "$t" \
                --num_fewshot 0 \
                --batch_size 4 \
                --limit "$LIMIT" \
                --device "$DEVICE" \
                "${gen_args[@]}" \
                --output_path "$tmp/$t" >"$tmp/$t.log" 2>&1; then
            echo "OK"
            ok=$((ok + 1))
        else
            echo "FAIL"
            failures+=("$t")
        fi
    done

    echo ""
    echo "  smoke test: $ok/${#tasks[@]} task(s) OK"

    PRECACHE_FAILED=0
    if (( ${#failures[@]} > 0 )); then
        PRECACHE_FAILED=1
        echo "" >&2
        echo "Failed task(s): ${failures[*]}" >&2
        for t in "${failures[@]}"; do
            echo "" >&2
            echo "--- $t: last 15 lines of $tmp/$t.log ---" >&2
            tail -15 "$tmp/$t.log" >&2
        done
        echo "" >&2
        echo "Logs kept in: $tmp" >&2
        echo "If it is a missing import, re-run with ENSURE_EXTRAS=1." >&2
    else
        rm -rf "$tmp"
    fi
}


#############################################
# Step 5 -- report
#############################################

print_env_status() {
    echo ""
    echo "=============================================="
    echo "Harness environment"
    echo "=============================================="

    "$venv_py" - <<'PY' || true
import importlib.util
import platform

print(f"  python           : {platform.python_version()}")
for name in ("lm_eval", "torch", "transformers", "datasets", "accelerate", "peft", "vllm"):
    if importlib.util.find_spec(name) is None:
        print(f"  {name:<16} : NOT INSTALLED")
        continue
    mod = __import__(name)
    print(f"  {name:<16} : {getattr(mod, '__version__', 'unknown')}")

try:
    import torch
    print(f"  torch cuda       : {torch.version.cuda} (available={torch.cuda.is_available()})")
except Exception as exc:
    print(f"  torch cuda       : unavailable ({type(exc).__name__})")
PY

    echo "  venv             : $venv_dir"
    echo "  harness          : $harness_dir ($(git -C "$harness_dir" rev-parse --short HEAD 2>/dev/null || echo '?'), $repo_branch)"
}

print_cache_status() {
    echo ""
    echo "=============================================="
    echo "Cached datasets ($(du -sh "$HF_DATASETS_CACHE" 2>/dev/null | cut -f1) total)"
    echo "=============================================="
    du -sh "$HF_DATASETS_CACHE"/* 2>/dev/null | sort -h | tail -20 || true
    echo ""
    echo "  NLTK data  : $NLTK_DATA ($(du -sh "$NLTK_DATA" 2>/dev/null | cut -f1))"
    echo "  RULER hay  : $RULER_HAYSTACK_DIR"
    echo ""
    echo "The eval job runs offline and reads exactly these paths, so it must export"
    echo "the same HF_DATASETS_CACHE, HF_HUB_CACHE, NLTK_DATA and RULER_HAYSTACK_DIR."
    echo ""
    echo "Next: sbatch $workdir/scripts/portuguese/eval_harness_pt_booster.sh"
    echo "=============================================="
}


#############################################
# Main
#############################################

cd "$workdir"

require_login_node
load_modules

if [[ "$SKIP_INSTALL" == "1" ]]; then
    echo "[$(date)] SKIP_INSTALL=1 -> pre-cache only, nothing is installed."
    if [[ ! -x "$venv_py" ]]; then
        echo "ERROR: $venv_py not found." >&2
        echo "       Run this script once with no arguments to do the full install." >&2
        exit 1
    fi
else
    drop_active_venv
    clone_harness
    apply_ruler_haystack_patch "$harness_dir" || echo "WARNING: RULER patch failed" >&2
    create_venv
    install_harness
    maybe_install_vllm
fi

PRECACHE_FAILED=0
precache_datasets

if [[ "$SKIP_INSTALL" != "1" ]]; then
    print_env_status
fi
print_cache_status

if (( PRECACHE_FAILED )); then
    exit 1
fi

deactivate 2>/dev/null || true
