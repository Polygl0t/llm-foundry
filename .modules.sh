# Auto-detecting module loader for the Marvin/Bender dual-stack (AMD / Intel).
#
# Source this file from any sbatch script:
#
#     source "$workdir/.modules.sh"
#
# Stack selection rules per cluster:
#
#   Marvin:
#     - Partitions with "gpu" in the name (e.g. sgpu, mlgpu)  -> AMD stack, CUDA 12.6
#     - All other partitions                                  -> Intel stack
#
#   Bender:
#     - Partition "a100"                                      -> AMD stack, CUDA 12.4
#     - Partition "a40"                                       -> Intel stack, CUDA 12.4
#
# Override: export LLM_FOUNDRY_STACK=amd|intel before submitting to bypass auto-detection.
# Learn about Marvin|Bender dual software stacks at:
# - https://wiki.hpc.uni-bonn.de/en/dualstacks

_stack=""
_cluster=""
_cuda_module=""

# Marvin paths to easybuild modulefiles
_easybuild_modulepath_amd_marvin="/opt/software/easybuild-AMD/modules/all:/etc/modulefiles:/usr/share/modulefiles:/opt/software/modulefiles:/usr/share/modulefiles/Linux:/usr/share/modulefiles/Core:/usr/share/lmod/lmod/modulefiles/Core"
_easybuild_modulepath_intel_marvin="/opt/software/easybuild-INTEL/modules/all:/etc/modulefiles:/usr/share/modulefiles:/opt/software/modulefiles:/usr/share/modulefiles/Linux:/usr/share/modulefiles/Core:/usr/share/lmod/lmod/modulefiles/Core"

# Bender paths to easybuild modulefiles
_easybuild_modulepath_amd_bender="/software/easybuild-AMD_A100/modules/all:/etc/modulefiles:/usr/share/modulefiles:/software/easybuild-AMD_A100/modules/all:/usr/share/modulefiles/Linux:/usr/share/modulefiles/Core:/usr/share/lmod/lmod/modulefiles/Core"
_easybuild_modulepath_intel_bender="/software/easybuild-INTEL_A40/modules/all:/etc/modulefiles:/usr/share/modulefiles:/software/easybuild-INTEL_A40/modules/all:/usr/share/modulefiles/Linux:/usr/share/modulefiles/Core:/usr/share/lmod/lmod/modulefiles/Core"


# Detect the cluster using hardcoded partition-name rules (primary), then
# SLURM_CLUSTER_NAME and hostname as fallbacks.
#
# Partition contains "gpu" (sgpu, mlgpu, ...) -> marvin
# Partition is "a100" or "a40"                -> bender
_detect_cluster() {
    local partition_lower="${1,,}"
    local hostname_fqdn

    # Primary: infer cluster from well-known partition names.
    if [[ "${partition_lower}" == *gpu* ]]; then
        printf '%s\n' "marvin"
        return 0
    fi

    if [[ "${partition_lower}" == *a100* || "${partition_lower}" == *a40* ]]; then
        printf '%s\n' "bender"
        return 0
    fi

    # Fallback: SLURM cluster name, then hostname.
    if [[ -n "${SLURM_CLUSTER_NAME:-}" ]]; then
        printf '%s\n' "${SLURM_CLUSTER_NAME,,}"
        return 0
    fi

    hostname_fqdn="$(hostname -f 2>/dev/null || hostname 2>/dev/null || true)"
    hostname_fqdn="${hostname_fqdn,,}"

    if [[ "${hostname_fqdn}" == *marvin* ]]; then
        printf '%s\n' "marvin"
        return 0
    fi

    if [[ "${hostname_fqdn}" == *bender* ]]; then
        printf '%s\n' "bender"
        return 0
    fi

    return 1
}

# Build MODULEPATH from a colon-separated list, keeping only entries that exist.
_apply_existing_modulepath_entries() {
    local requested_path="$1"
    local rebuilt_path=""
    local entry

    IFS=':' read -r -a _modulepath_entries <<< "${requested_path}"

    for entry in "${_modulepath_entries[@]}"; do
        [[ -d "${entry}" ]] || continue
        [[ -n "${rebuilt_path}" ]] && rebuilt_path+=":"
        rebuilt_path+="${entry}"
    done

    if [[ -n "${rebuilt_path}" ]]; then
        export MODULEPATH="${rebuilt_path}"
        echo "[.modules.sh] MODULEPATH: ${MODULEPATH}" >&2
    else
        echo "[.modules.sh] WARNING: none of the requested module paths exist; leaving MODULEPATH unchanged" >&2
    fi

    unset _modulepath_entries
}

# Step 1: Detect cluster
_partition="${SLURM_JOB_PARTITION:-}"
_partition_lower="${_partition,,}"

if ! _cluster="$(_detect_cluster "${_partition_lower}")"; then
    _cluster="unknown"
fi

echo "[.modules.sh] Cluster: ${_cluster}  Partition: ${_partition:-<none>}" >&2

# Step 2: Determine stack (LLM_FOUNDRY_STACK overrides auto-detection)
if [[ -n "${LLM_FOUNDRY_STACK:-}" ]]; then
    _stack="${LLM_FOUNDRY_STACK,,}"
    echo "[.modules.sh] Stack: ${_stack}  (override via LLM_FOUNDRY_STACK)" >&2
else
    case "${_cluster}" in
        marvin)
            # Partitions with "gpu" in the name (sgpu, mlgpu, ...) use the AMD stack.
            if [[ "${_partition_lower}" == *gpu* ]]; then
                _stack="amd"
            else
                _stack="intel"
            fi
            ;;
        bender)
            # a100 nodes use the AMD stack; a40 nodes use the Intel stack.
            if [[ "${_partition_lower}" == *a100* ]]; then
                _stack="amd"
            else
                _stack="intel"
            fi
            ;;
        *)
            _stack="intel"
            echo "[.modules.sh] WARNING: unknown cluster '${_cluster}'; defaulting to Intel stack (set LLM_FOUNDRY_STACK to override)" >&2
            ;;
    esac
    echo "[.modules.sh] Stack: ${_stack}  (auto-detected)" >&2
fi

# Step 3: Select module path based on cluster and stack
case "${_cluster}_${_stack}" in
    marvin_amd)   _easybuild_modulepath="${_easybuild_modulepath_amd_marvin}" ;;
    marvin_intel) _easybuild_modulepath="${_easybuild_modulepath_intel_marvin}" ;;
    bender_amd)   _easybuild_modulepath="${_easybuild_modulepath_amd_bender}" ;;
    bender_intel) _easybuild_modulepath="${_easybuild_modulepath_intel_bender}" ;;
    *)
        _easybuild_modulepath=""
        echo "[.modules.sh] WARNING: no module path defined for cluster '${_cluster}', stack '${_stack}'" >&2
        ;;
esac
echo "[.modules.sh] Module path: ${_easybuild_modulepath%%:*}" >&2

# Step 4: Determine CUDA version
# Marvin AMD: CUDA 12.6   Marvin Intel: no CUDA
# Bender AMD: CUDA 12.4   Bender Intel: CUDA 12.4 (both stacks require it)

case "${_cluster}" in
    marvin)
        [[ "${_stack}" == "amd" ]] && _cuda_module="CUDA/12.6.0" || _cuda_module=""
        ;;
    bender)
        _cuda_module="CUDA/12.4.0"
        ;;
    *)
        _cuda_module=""
        ;;
esac

# Step 5: Load modules
case "${_stack}" in
    amd)
        _apply_existing_modulepath_entries "${_easybuild_modulepath}"
        module purge
        if [[ -z "${_cuda_module}" ]]; then
            echo "[.modules.sh] ERROR: no CUDA module defined for cluster '${_cluster}'" >&2
            unset _stack _cluster _cuda_module _partition _partition_lower
            return 1 2>/dev/null || exit 1
        fi
        echo "[.modules.sh] Loading: ${_cuda_module} + Python/3.12.3-GCCcore-13.3.0" >&2
        module load "${_cuda_module}" Python/3.12.3-GCCcore-13.3.0
        ;;
    intel)
        _apply_existing_modulepath_entries "${_easybuild_modulepath}"
        module --force purge
        if [[ -n "${_cuda_module}" ]]; then
            echo "[.modules.sh] Loading: ${_cuda_module} + Python/3.12.3-GCCcore-13.3.0" >&2
            module load "${_cuda_module}" Python/3.12.3-GCCcore-13.3.0
        else
            echo "[.modules.sh] Loading: Python/3.12.3-GCCcore-13.3.0" >&2
            module load Python/3.12.3-GCCcore-13.3.0
        fi
        ;;
    *)
        echo "[.modules.sh] ERROR: unknown stack '${_stack}' (expected 'amd' or 'intel')" >&2
        unset _stack _cluster _cuda_module _partition _partition_lower
        return 1 2>/dev/null || exit 1
        ;;
esac

# Step 6: Report loaded modules
if command -v module >/dev/null 2>&1; then
    echo "[.modules.sh] Loaded modules:" >&2
    module list 2>&1 | sed 's/^/    /' >&2
fi

export LLM_FOUNDRY_STACK="${_stack}"
export LLM_FOUNDRY_MODULEPATH="${_easybuild_modulepath}"
unset -f _detect_cluster
unset -f _apply_existing_modulepath_entries
unset _stack _cluster _cuda_module _partition _partition_lower _easybuild_modulepath \
    _easybuild_modulepath_amd_marvin _easybuild_modulepath_intel_marvin \
    _easybuild_modulepath_amd_bender _easybuild_modulepath_intel_bender