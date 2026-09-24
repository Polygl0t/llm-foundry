#!/bin/bash -l

#############################################
# Run the LM-evaluation-harness on one JUPITER booster node
# (4 GH200 -> 4 evals in parallel).
#
# Requires : bash jupiter_installation_eval_harness_2026.sh (login node)
#############################################
#SBATCH --account=polyglot
#SBATCH --partition=booster
#SBATCH --job-name=evals
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#############################################

workdir="/e/project1/polyglot"
cd "$workdir"
ulimit -c 0

mkdir -p "$workdir/logs"
out="$workdir/logs/eval-pt-out.${SLURM_JOB_ID:-local}"

source "$workdir/setup/jupiter_modules_2026.sh" > "$out" 2>&1
venv_dir="$workdir/.venv_eval"
venv_py="$venv_dir/bin/python"

if [[ ! -x "$venv_py" ]]; then
    echo "ERROR: $venv_py not found." | tee -a "$out" >&2
    echo "       Run 'bash jupiter_installation_eval_harness_2026.sh' on a login node first." | tee -a "$out" >&2
    exit 1
fi

# --- Environment setup --------------------------------------------------------
export VIRTUAL_ENV="$venv_dir"
export PATH="$venv_dir/bin:$PATH"
export HF_DATASETS_CACHE="$workdir/.cache/eval_harness/datasets"
export HF_HUB_CACHE="$workdir/.cache/eval_harness/models"
export NLTK_DATA="$workdir/.cache/eval_harness/nltk_data"
export RULER_HAYSTACK_DIR="$workdir/.cache/eval_harness/ruler_haystack"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="8"
export MKL_NUM_THREADS="8"
export TRITON_CACHE_DIR="$workdir/.cache/eval_harness/triton_cache/${SLURM_JOB_ID:-local}"
export TORCHINDUCTOR_CACHE_DIR="$workdir/.cache/eval_harness/inductor_cache/${SLURM_JOB_ID:-local}"
export VLLM_CACHE_ROOT="$workdir/.cache/eval_harness/vllm_cache"
export VLLM_NO_USAGE_STATS=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

# --- What to evaluate --------------------------------------------------------
MODELS=(
    "$workdir/checkpoints/Curio-edu-1.1b"
    "$workdir/checkpoints/Tucano2-0.6B-Base"
    "$workdir/checkpoints/manaca-1b-base"
    "$workdir/checkpoints/Tucano-2b4"
)

# --- Tasks to evaluate ---------------------------------------------------------
TASKS="arc_challenge_poly_pt,\
mmlu_poly_pt,\
hellaswag_poly_pt,\
lambada_poly_pt,\
calame_pt,\
belebele_por_Latn,\
global_piqa_nonparallel_cloze_por_latn_braz,\
global_piqa_parallel_cloze_por_latn_braz,\
bluex_cloze,\
enem_cloze,\
oab_exams_cloze"

# --- Chat tasks inclusion ------------------------------------------------------
ADD_CHAT_TASKS="0" # <-- Set to 1 to include chat tasks in the evaluation
CHAT_TASKS="ifeval_pt,gsm8k_pt,ruler_pt,humaneval,humaneval_instruct"
if [[ "$ADD_CHAT_TASKS" == "1" ]]; then
    TASKS="$TASKS,$CHAT_TASKS"
fi

# --- Evaluation settings ------------------------------------------------------
NUM_FEWSHOT=0
BATCH_SIZE="auto"
MODE="vllm" # <-- hf or vllm
MAX_MODEL_LEN=""
LIMIT=""

# --- Chat-model flags --------------------------------------------------------
#
# lm-eval knobs -> variables:
#   --apply_chat_template                  CHAT_MODEL=1
#   --fewshot_as_multiturn <bool>          MULTITURN_FEWSHOT=1|0 (auto-on with CHAT_MODEL)
#   --model_args=...,enable_thinking=True  ENABLE_THINKING=1    (generation only)
#   --model_args=...,think_end_token='</think>'  THINK_END_TOKEN
#   --model_args=...,max_length=32768      MAX_MODEL_LEN
#   --metadata '{"max_seq_lengths":[...]}' RULER_SEQ_LENGTHS

CHAT_MODEL="0"                  # 1 -> --apply_chat_template
MULTITURN_FEWSHOT="auto"        # 1 forces it on, 0 forces it off, auto leaves it alone.
ENABLE_THINKING="0"             # 1 -> model_args enable_thinking=True (needs CHAT_MODEL=1)
THINK_END_TOKEN="</think>"      # required by enable_thinking; Qwen3 uses </think>

# --- RULER (long context) ----------------------------------------------------
# RULER's context lengths live outside its task YAML: each `niah_pt_*` sub-task is
# generated per length and reports one metric per length ("4096", "8192", ...),
# supplied at runtime as --metadata '{"max_seq_lengths":[...]}'. Use the ladder
# (4096 8192 16384 ...) for a long-context stage, never a length the model was not
# trained for, and keep MAX_MODEL_LEN >= the largest entry.
RULER_SEQ_LENGTHS=(4096)

# --- Scheduling --------------------------------------------------------------
MAX_PARALLEL=4        # one eval per GH200

# --- Output ------------------------------------------------------------------
RESULTS_TAG="${RESULTS_TAG:-$([[ "$CHAT_MODEL" == "1" ]] && echo chat)}"
RESULTS_DIR="$workdir/checkpoints/.evals${RESULTS_TAG:+-$RESULTS_TAG}"
LOGS_DIR="$workdir/logs/eval-pt.${SLURM_JOB_ID:-local}"
TMP_ROOT="$LOGS_DIR/.tmp"

set -uo pipefail

# Build the --metadata JSON, e.g. {"max_seq_lengths":[4096,8192]}.
RULER_METADATA=""
RULER_MAX_LEN=0
if (( ${#RULER_SEQ_LENGTHS[@]} > 0 )); then
    _lengths=""
    for _len in "${RULER_SEQ_LENGTHS[@]}"; do
        _lengths+="${_lengths:+,}${_len}"
        (( _len > RULER_MAX_LEN )) && RULER_MAX_LEN=$_len
    done
    RULER_METADATA="{\"max_seq_lengths\":[${_lengths}]}"
fi

if [[ -n "$RULER_METADATA" && -n "$MAX_MODEL_LEN" ]] && (( RULER_MAX_LEN > MAX_MODEL_LEN )); then
    echo "WARN: RULER asks for $RULER_MAX_LEN tokens but MAX_MODEL_LEN=$MAX_MODEL_LEN."
    echo "      Prompts will be truncated; lengths above $MAX_MODEL_LEN will score ~0."
fi

if [[ "$ENABLE_THINKING" == "1" && "$CHAT_MODEL" != "1" ]]; then
    echo "WARN: ENABLE_THINKING=1 has no effect without CHAT_MODEL=1 (no chat template is applied)."
fi

if [[ "$ENABLE_THINKING" == "1" ]]; then
    echo "WARN: lm_eval rejects enable_thinking=True for loglikelihood tasks, so every"
    echo "      cloze task in TASKS will fail. Use thinking only with generation tasks."
fi

# humaneval/humaneval_instruct (and mbpp, cruxeval, etc.) are marked `unsafe_code: true`
# in lm-evaluation-harness: they execute model-generated code to check test cases, and
# lm_eval refuses to run them unless this is explicitly confirmed.
CONFIRM_UNSAFE_CODE="0"
if [[ ",$TASKS," == *",humaneval,"* || ",$TASKS," == *",humaneval_instruct,"* ]]; then
    CONFIRM_UNSAFE_CODE="1"
fi

mkdir -p "$LOGS_DIR" "$RESULTS_DIR" "$TMP_ROOT"

echo "========================================="
echo "Models listed   : ${#MODELS[@]}"
echo "Results         : $RESULTS_DIR"
echo "Logs            : $LOGS_DIR"
echo "Backend         : $MODE    few-shot: $NUM_FEWSHOT"
echo "Parallel        : $MAX_PARALLEL (of ${SLURM_NTASKS_PER_NODE:-$MAX_PARALLEL} allocated)"
echo "Max model len   : ${MAX_MODEL_LEN:-<model default>}"
echo "Limit           : ${LIMIT:-<none>}"
if [[ "$CHAT_MODEL" == "1" ]]; then
    echo "Chat template   : yes (multiturn fewshot: $MULTITURN_FEWSHOT, thinking: $ENABLE_THINKING)"
else
    echo "Chat template   : no (base model)"
fi
if [[ -n "$RULER_METADATA" ]]; then
    echo "RULER lengths   : ${RULER_SEQ_LENGTHS[*]}"
else
    echo "RULER lengths   : <ruler_pt default: 4096>"
fi
echo "Unsafe code run : $([[ "$CONFIRM_UNSAFE_CODE" == "1" ]] && echo "yes (humaneval)" || echo "no")"
echo "========================================="

#############################################
# Phase 1 -- build the work queue
#############################################

declare -a QUEUE=()        # model directories, in the order listed in MODELS
declare -a QUEUE_NAME=()   # output file stem for each entry
declare -A NAME_TAKEN=()   # guards against duplicate stems
UNIQUE_NAME=""             # result of unique_name()

unique_name() {
    local src="$1" label="$2" parent candidate
    parent="$(basename "$(dirname "$src")")"
    candidate="$label"
    if [[ -n "${NAME_TAKEN[$candidate]:-}" ]]; then
        candidate="${parent}-${label}"
    fi
    local n=2
    while [[ -n "${NAME_TAKEN[$candidate]:-}" ]]; do
        candidate="${parent}-${label}-${n}"
        n=$((n + 1))
    done
    NAME_TAKEN["$candidate"]=1
    UNIQUE_NAME="$candidate"
}

add_item() {
    # $1 = raw entry, either "/path/to/model" or "/path/to/model|label"
    local entry="$1" src label
    src="${entry%%|*}"
    if [[ "$entry" == *"|"* ]]; then
        label="${entry##*|}"
    else
        label="$(basename "$src")"
    fi

    if [[ ! -d "$src" ]]; then
        echo "WARN: skipped (not a directory): $src"
        return 0
    fi

    unique_name "$src" "$label"
    local name="$UNIQUE_NAME"

    if [[ -f "$RESULTS_DIR/$name.yaml" ]]; then
        echo "SKIP: $name already evaluated ($RESULTS_DIR/$name.yaml)"
        return 0
    fi
    QUEUE+=("$src")
    QUEUE_NAME+=("$name")
}

for entry in "${MODELS[@]}"; do
    add_item "$entry"
done

TOTAL=${#QUEUE[@]}
if (( TOTAL == 0 )); then
    echo "Nothing to evaluate: MODELS is empty, or every entry was skipped."
    echo "Done."
    exit 0
fi

echo ""
echo "$TOTAL model(s) to evaluate:"
for i in "${!QUEUE[@]}"; do
    printf '  %2d. %-22s %s\n' "$((i + 1))" "${QUEUE_NAME[$i]}" "${QUEUE[$i]}"
done
echo ""

#############################################
# Phase 2 -- run the queue, one eval per GPU
#############################################

declare -a GPU_PID=()      # PID occupying each GPU ("" = free)
declare -a GPU_NAME=()     # name of the job occupying that GPU

FINISHED=0
FAILED=0

launch_eval() {
    # $1 = source dir, $2 = name, $3 = gpu index
    local src="$1" name="$2" gpu="$3"
    local job_out="$LOGS_DIR/out_${name}.log"
    local job_err="$LOGS_DIR/err_${name}.log"
    local tmp_dir="$TMP_ROOT/$name"
    rm -rf "$tmp_dir"; mkdir -p "$tmp_dir"

    # --model_args is shared by both backends except for the length-cap key.
    local margs="pretrained=$src"
    if [[ -n "$MAX_MODEL_LEN" ]]; then
        if [[ "$MODE" == "vllm" ]]; then
            margs="$margs,max_model_len=$MAX_MODEL_LEN"
        else
            margs="$margs,max_length=$MAX_MODEL_LEN"
        fi
    fi
    # Thinking needs think_end_token and is rejected by loglikelihood tasks.
    if [[ "$ENABLE_THINKING" == "1" && "$CHAT_MODEL" == "1" ]]; then
        margs="$margs,enable_thinking=True,think_end_token=$THINK_END_TOKEN"
    fi

    # Chat-model flags are not passed at all for base checkpoints.
    local extra=()
    [[ -n "$LIMIT" ]] && extra+=(--limit "$LIMIT")
    [[ "$CHAT_MODEL" == "1" ]] && extra+=(--apply_chat_template)
    [[ "$CONFIRM_UNSAFE_CODE" == "1" ]] && extra+=(--confirm_run_unsafe_code)
    case "$MULTITURN_FEWSHOT" in
        1) extra+=(--fewshot_as_multiturn true) ;;
        0) extra+=(--fewshot_as_multiturn false) ;;
    esac
    # RULER context lengths. Must stay LAST: --metadata is `nargs="+"`, so it
    # swallows every following token that is not an option.
    [[ -n "$RULER_METADATA" ]] && extra+=(--metadata "$RULER_METADATA")

    echo "  [gpu $gpu] launch  $name"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        "$venv_py" -m lm_eval \
            --model "$MODE" \
            --model_args "$margs" \
            --tasks "$TASKS" \
            --num_fewshot "$NUM_FEWSHOT" \
            --batch_size "$BATCH_SIZE" \
            --device cuda \
            --output_path "$tmp_dir" \
            "${extra[@]}"
    ) >"$job_out" 2>"$job_err" &
    # The caller reads $! immediately; this function must not background
    # anything else, or the wrong PID would be recorded.
}

# Block until GPU $1 is free, then report how the job that was using it ended.
drain_gpu() {
    local gpu="$1"
    if [[ -n "${GPU_PID[$gpu]:-}" ]]; then
        if wait "${GPU_PID[$gpu]}"; then
            echo "  [gpu $gpu] OK    ${GPU_NAME[$gpu]}"
        else
            echo "  [gpu $gpu] FAIL  ${GPU_NAME[$gpu]}  (see $LOGS_DIR/err_${GPU_NAME[$gpu]}.log)"
            FAILED=$((FAILED + 1))
        fi
        GPU_PID[$gpu]=""
        FINISHED=$((FINISHED + 1))
    fi
}

# Item i always uses GPU (i % MAX_PARALLEL) and we wait for the previous job on
# that GPU before reusing it, so exactly four evals run concurrently.
for idx in "${!QUEUE[@]}"; do
    gpu=$(( idx % MAX_PARALLEL ))
    drain_gpu "$gpu"
    launch_eval "${QUEUE[$idx]}" "${QUEUE_NAME[$idx]}" "$gpu"
    GPU_PID[$gpu]=$!
    GPU_NAME[$gpu]="${QUEUE_NAME[$idx]}"
done

for gpu in $(seq 0 $((MAX_PARALLEL - 1))); do
    drain_gpu "$gpu"
done

echo ""
echo "All evaluations finished: $((FINISHED - FAILED))/$FINISHED OK, $FAILED failed."

#############################################
# Phase 3 -- JSON -> YAML
#############################################

echo ""
echo "========================================="
echo "Post-processing JSON -> YAML"
echo "========================================="

cat > "$LOGS_DIR/json_to_yaml.py" <<'PYEOF'
"""Flatten one lm-eval results JSON into a small YAML file.

The JSON lands in <output_path>/<model>/results_<timestamp>.json; the newest file
wins, so a resumed run cannot pick up a stale result from an earlier attempt.
"""
import json
import os
import re
import sys

import yaml

tmp_dir, yaml_path, model_name, model_path = sys.argv[1:5]

candidates = [
    os.path.join(root, f)
    for root, _, files in os.walk(tmp_dir)
    for f in files
    if f.endswith(".json")
]
if not candidates:
    print(f"  !! {model_name}: no JSON results found in {tmp_dir}")
    sys.exit(1)

filepath = max(candidates, key=os.path.getmtime)
with open(filepath) as fh:
    data = json.load(fh)

results = data.get("results", data)

flat = {}
if isinstance(results, dict):
    for key, value in results.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat[f"{key}_{subkey.replace(',none', '')}"] = subvalue
        else:
            flat[key] = value

# The stem is not always a bare `step_XXXXX` (collisions get a parent prefix),
# so look for the step anywhere in it.
match = re.search(r"step_(\d+)", model_name)
step = int(match.group(1)) if match else None

out = {
    "model_name": model_name,
    "model_path": model_path,
    "checkpoint_step": step,
    "results": flat,
}

with open(yaml_path, "w") as fh:
    yaml.dump(out, fh, default_flow_style=False, sort_keys=True)

print(f"  -> {os.path.basename(yaml_path)}  ({len(flat)} metric(s))")
PYEOF

PROCESSED=0
for i in "${!QUEUE_NAME[@]}"; do
    name="${QUEUE_NAME[$i]}"
    tmp_dir="$TMP_ROOT/$name"
    yaml_out="$RESULTS_DIR/$name.yaml"

    if [[ ! -d "$tmp_dir" ]]; then
        continue
    fi
    if "$venv_py" "$LOGS_DIR/json_to_yaml.py" "$tmp_dir" "$yaml_out" "$name" "${QUEUE[$i]}"; then
        PROCESSED=$((PROCESSED + 1))
        rm -rf "$tmp_dir"
    else
        # Keep the temp dir for inspection if it produced nothing.
        echo "  !! $name: post-processing failed (raw output kept in $tmp_dir)"
    fi
done

echo ""
echo "Wrote $PROCESSED YAML file(s) to $RESULTS_DIR"

#############################################
# Cleanup
#############################################

rmdir "$TMP_ROOT" 2>/dev/null || true

echo ""
echo "========================================="
echo "Done. $PROCESSED model(s) evaluated, $FAILED failure(s)."
echo "Results: $RESULTS_DIR"
echo "Logs   : $LOGS_DIR"
echo "========================================="

if (( FAILED > 0 )); then
    exit 1
fi
