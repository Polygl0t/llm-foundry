#!/bin/bash -l

#############################################
# SLURM Job Configuration
#############################################
# One-time setup: submit this job to create a ready-to-use venv
# for the agents pipeline.
#
# Usage:
#   sbatch synthetic/agents/slurm/create_venv_agents.sh
#
#############################################
#SBATCH --account=ag_bit_flek              # <-- Change to your SLURM account
#SBATCH --partition=mlgpu_devel            # <-- Change to your partition
#SBATCH --job-name=create-venv-agents
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=16G
#SBATCH --oversubscribe

set -e
#############################################
# Configuration — tweak these to your needs
#############################################

# Workspace root.
workdir="/lustre/mlnvme/data/polyglot"

# Name of the venv directory.
venv_name=".venv_agents"

# Path to the .modules file.
modules_file="$workdir/llm-foundry/.modules.sh"

# Log file
mkdir -p "$workdir/run_outputs"
out="$workdir/run_outputs/create-venv-agents-out.$SLURM_JOB_ID"

cd "$workdir"
ulimit -c 0

# Redirect ALL output (stdout + stderr) for the rest of the script to $out.
exec > "$out" 2>&1

echo "# [${SLURM_JOB_ID}] Job started at: $(date)"
echo "# [${SLURM_JOB_ID}] Hostname: $(hostname)"
echo "# [${SLURM_JOB_ID}] Workdir: $workdir"
echo "# [${SLURM_JOB_ID}] Venv dir: $workdir/$venv_name"

#############################################
# Modules Setup
#############################################

echo "===== Setting up modules ====="
export LLM_FOUNDRY_STACK=amd # On Marvin, GPU nodes require the AMD stack.
source "$modules_file"

echo "===== Environment Info ====="
echo "  Python: $(which python3) — $(python3 --version)"
echo "  CUDA_HOME: ${CUDA_HOME:-not set}"
echo "  nvcc: $(which nvcc 2>/dev/null || echo 'not found') — $(nvcc --version 2>/dev/null | head -n1 || echo 'N/A')"

#############################################
# Create venv
#############################################

venv_dir="$workdir/$venv_name"
echo "===== Creating venv at $venv_dir ====="
python3 -m venv "$venv_dir"
source "$venv_dir/bin/activate"

echo "===== Upgrading pip ====="
pip3 install --upgrade pip

echo "===== Installing uv ====="
pip3 install uv

echo "===== Installing PyTorch 2.11.0 (pinned to vLLM's required version) ====="
uv pip install --no-cache torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# Pin torch via a constraints file used by every install that follows.
torch_constraints="/tmp/torch-constraints-$SLURM_JOB_ID.txt"
cat > "$torch_constraints" << 'EOF'
torch==2.11.0
torchvision==0.26.0
torchaudio==2.11.0
EOF

echo "===== Installing vLLM 0.23.0 + flashinfer-python 0.6.12 (pinned) ====="
uv pip install --no-cache -c "$torch_constraints" vllm==0.23.0 flashinfer-python==0.6.12

echo "===== Installing CUDA 13 nvcc (pip) for flashinfer/DeepGEMM JIT ====="
uv pip install --no-cache "cuda-toolkit[nvcc]==13.0.2"

echo "===== Verifying nvcc ====="
_nvcc="$(find "$venv_dir/lib" -name nvcc -type f | head -1)"
if [[ -n "$_nvcc" ]]; then
    echo "  nvcc: $_nvcc"
    "$_nvcc" --version | tail -1
else
    echo "  nvcc: NOT FOUND (JIT compile will fall back to the system nvcc, which may not be available)"
fi

echo "===== Installing agents dependencies from the foundry's pyproject.toml ====="
uv pip install --no-cache -c "$torch_constraints" \
    "accelerate" \
    "transformers" \
    "smolagents[litellm]>=1.26.0" \
    "datasets>=3.0" \
    "python-dotenv>=1.0" \
    "pyyaml>=6.0" \
    "sympy>=1.13" \
    "ddgs>=9.0.0" \
    "wikipedia-api==0.15.0" \
    "markdownify>=0.14.1"

rm -f "$torch_constraints"

echo "===== Verifying installation ====="
python3 - <<'PY'
from importlib.metadata import version
import torch

packages = [
    "torch", "vllm", "transformers", "accelerate", "datasets",
    "smolagents", "litellm", "pyyaml", "sympy", "ddgs",
    "wikipedia-api", "markdownify", "python-dotenv",
]
for pkg in packages:
    try:
        ver = version(pkg)
        tag = f" (runtime cuda={torch.version.cuda})" if pkg == "torch" else ""
        print(f"  {pkg}=={ver}{tag}")
    except Exception as e:
        print(f"  {pkg}: NOT FOUND ({e})")
PY

echo "===== Verifying vLLM import ====="
python3 -c "from vllm import LLM; print('  vllm.LLM OK')"

echo "===== Done ====="
echo "Venv ready: $venv_dir"
echo "Activate with: source $venv_dir/bin/activate"
