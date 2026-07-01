#!/bin/bash -l

#############################################
# TEST SCRIPT: .modules.sh stack detection
#############################################
# Covers all partition/cluster combinations for both AMD and Intel stacks.
# Run with:  bash test_modules.sh
#############################################

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_pass=0
_fail=0

_run_case() {
    local label="$1" partition="$2" cluster="$3" expected_stack="$4" expected_path_kw="$5"

    result="$(SLURM_JOB_PARTITION="${partition}" SLURM_CLUSTER_NAME="${cluster}" \
        bash --norc -c 'source "'"${repo_root}"'/.modules.sh" 2>/dev/null; echo "${LLM_FOUNDRY_STACK}|${LLM_FOUNDRY_MODULEPATH}"')"

    local result_stack="${result%%|*}"
    local result_path="${result#*|}"
    local ok=1

    if [[ "${result_stack}" != "${expected_stack}" ]]; then
        echo "  FAIL  ${label}  (partition=${partition}, cluster=${cluster}) -> stack: expected '${expected_stack}', got '${result_stack}'"
        ok=0
    fi

    if [[ -n "${expected_path_kw}" && "${result_path}" != *"${expected_path_kw}"* ]]; then
        echo "  FAIL  ${label}  (partition=${partition}, cluster=${cluster}) -> modulepath: expected to contain '${expected_path_kw}'"
        ok=0
    fi

    if [[ "${ok}" -eq 1 ]]; then
        echo "  PASS  ${label}  (partition=${partition}, cluster=${cluster}) -> stack=${result_stack}"
        (( _pass++ ))
    else
        (( _fail++ ))
    fi
}

echo "================================================================"
echo " TEST SUITE: AMD stack"
echo "================================================================"

_run_case "Marvin GPU (sgpu_long)" "sgpu_long"  "marvin" "amd"   "easybuild-AMD/modules"
_run_case "Marvin GPU (mlgpu)"     "mlgpu"      "marvin" "amd"   "easybuild-AMD/modules"
_run_case "Bender A100short"       "A100short"  "bender" "amd"   "easybuild-AMD_A100"
_run_case "Bender A100medium"      "A100medium" "bender" "amd"   "easybuild-AMD_A100"

echo ""
echo "================================================================"
echo " TEST SUITE: Intel stack"
echo "================================================================"

_run_case "Marvin CPU (lm_medium)" "lm_medium" "marvin" "intel" "easybuild-INTEL/modules"
_run_case "Marvin CPU (lm_long)"   "lm_long"   "marvin" "intel" "easybuild-INTEL/modules"
_run_case "Bender a40 (A40short)"  "A40short"  "bender" "intel" "easybuild-INTEL_A40"
_run_case "Bender a40 (A40medium)" "A40medium" "bender" "intel" "easybuild-INTEL_A40"

echo ""
echo "================================================================"
echo " TEST SUITE: BAF (HTCondor)"
echo "================================================================"

# BAF detection relies on hostname containing "baf" (fallback in _detect_cluster).
# We mock hostname since we're not running on a BAF node.
_fake_bin="$(mktemp -d)"
trap 'rm -rf "${_fake_bin}"' EXIT
cat > "${_fake_bin}/hostname" << 'EOF'
#!/bin/bash
echo "node01.baf.htcondor"
EOF
chmod +x "${_fake_bin}/hostname"

result="$(PATH="${_fake_bin}:${PATH}" bash --norc -c 'source "'"${repo_root}"'/.modules.sh" 2>/dev/null; echo "CUDA=${CUDA_HOME:-<unset>}|STACK=${LLM_FOUNDRY_STACK:-<unset>}"')"
if [[ "${result}" == *"CUDA=/usr/local/cuda-12"* ]]; then
    echo "  PASS  BAF hostname detection -> ${result}"
    (( _pass++ ))
else
    echo "  FAIL  BAF hostname detection -> ${result}"
    (( _fail++ ))
fi

# Override via LLM_FOUNDRY_STACK=baf (with hostname mock)
result="$(PATH="${_fake_bin}:${PATH}" LLM_FOUNDRY_STACK=baf bash --norc -c 'source "'"${repo_root}"'/.modules.sh" 2>/dev/null; echo "CUDA=${CUDA_HOME:-<unset>}|STACK=${LLM_FOUNDRY_STACK:-<unset>}"')"
if [[ "${result}" == *"CUDA=/usr/local/cuda-12"* ]]; then
    echo "  PASS  BAF override (LLM_FOUNDRY_STACK=baf) -> ${result}"
    (( _pass++ ))
else
    echo "  FAIL  BAF override (LLM_FOUNDRY_STACK=baf) -> ${result}"
    (( _fail++ ))
fi

echo ""
echo "----------------------------------------------------------------"
echo "Results: ${_pass} passed, ${_fail} failed"
[[ "${_fail}" -eq 0 ]] && echo "OVERALL: PASS" || echo "OVERALL: FAIL"
