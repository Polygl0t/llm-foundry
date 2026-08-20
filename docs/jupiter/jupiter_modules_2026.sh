#########################################################
# 2026 Modules for JSC JUPITER.
#########################################################

echo "[jupiter_modules_2026.sh] Setting up JUPITER 2026 software stack" >&2

echo "[jupiter_modules_2026.sh] Purging all loaded modules" >&2
module --force purge

echo "[jupiter_modules_2026.sh] Loading Stages/2026" >&2
module load Stages/2026

echo "[jupiter_modules_2026.sh] Loading GCCcore/.14.3.0" >&2
module load GCCcore/.14.3.0

echo "[jupiter_modules_2026.sh] Loading Python/3.13.5" >&2
module load Python/3.13.5

echo "[jupiter_modules_2026.sh] Loading CUDA/13" >&2
module load CUDA/13

echo "[jupiter_modules_2026.sh] Loading CMake Ninja" >&2
module load CMake Ninja

# Report loaded modules
if command -v module >/dev/null 2>&1; then
    echo "[jupiter_modules_2026.sh] Loaded modules:" >&2
    module list 2>&1 | sed 's/^/    /' >&2
fi

echo "[jupiter_modules_2026.sh] JUPITER 2026 stack ready" >&2
