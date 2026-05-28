#!/bin/bash -l

#############################################
# Workspace Setup Script for Marvin HPC Cluster
#############################################
# This script allocates a workspace on the Marvin HPC cluster,
# clones the repository, and prepares the directory structure.
#
# Learn about Marvin workspaces at:
# https://wiki.hpc.uni-bonn.de/en/marvin/workspaces
#############################################

# ----------- User Customization Section -----------
username="nklugeco_hpc"        # <-- Change to your username
file_system="mlnvme"           # <-- Change to your filesystem
work_group="ag_bit_flek"       # <-- Change to your work group
email="kluge@uni-bonn.de"      # <-- Change to your email
remainder=7                    # <-- Change as needed
num_days=90                    # <-- Change as needed
workspace_name="polyglot"      # <-- Change to your workspace/project name

# Workspace directory (constructed from the above variables)
workdir="/lustre/$file_system/data/$username-$workspace_name"

#############################################
# Allocate Workspace
# User Guide to HPC Workspaces:
# https://github.com/holgerBerger/hpc-workspace/blob/1.4.0/user-guide.md
#############################################
ws_allocate -F $file_system -G $work_group -m $email -r $remainder -d $num_days -n $workspace_name
echo "Workspace allocated!"
echo "Username: $username"
echo "File System: $file_system"
echo "Work Group: $work_group"
echo "Workspace Name: $workspace_name"
echo "Workspace Directory: $workdir"

#############################################
# Clone Repository
#############################################
cd $workdir
git clone --branch main https://github.com/Polygl0t/llm-foundry.git
echo "Workspace ready."

#############################################
# Stack Sourcing (.modules.sh)
#############################################
# Marvin|Bender has a dual software stack (AMD and Intel). Instead of two
# separate module files, this repo ships a single auto-detecting
# loader at the repository root: `.modules.sh`.
#
# The script first detects the cluster from SLURM_JOB_PARTITION (falling
# back to SLURM_CLUSTER_NAME, then hostname), then selects the stack:
#
#   Marvin:
#     - Partition contains "gpu" (e.g. sgpu, mlgpu) -> AMD stack, CUDA 12.6
#     - Any other partition                         -> Intel stack, no CUDA
#
#   Bender:
#     - Partition "a100"                            -> AMD stack, CUDA 12.4
#     - Partition "a40"                             -> Intel stack, CUDA 12.4
#
#   Unknown cluster                                 -> Intel stack + warning
#
# Override: set LLM_FOUNDRY_STACK=amd|intel before sourcing to bypass
# auto-detection. Required on login nodes (no SLURM context):
#
#   LLM_FOUNDRY_STACK=amd   source $workdir/.modules.sh
#   LLM_FOUNDRY_STACK=intel source $workdir/.modules.sh
#
# Inside a SLURM job, SLURM_JOB_PARTITION is set automatically from your
# #SBATCH --partition directive, so the right stack is selected without
# any extra configuration:
#
#   source $workdir/.modules.sh
#
# When sourced, the script logs the chosen cluster, stack, module path,
# and a `module list` so the resolved environment is visible in your job log.

#############################################
# Installing Dependencies
#############################################
# Dependencies are managed via pyproject.toml with optional groups:
#
#   data         - Data processing (datatrove, spacy, stanza, etc.)
#   distributed  - Distributed training (torch, accelerate, flash_attn, etc.)
#   synth        - Synthetic data generation (torch, vllm, etc.)
#   trl          - Fine-tuning with TRL (trl, vllm, flash_attn, etc.)
#
# Each config must be installed on the matching node type so that
# hardware-specific packages (CUDA wheels, flash-attn, etc.) resolve
# correctly. The `.modules.sh` loader handles stack selection for you;
# you just need to submit the install job to the correct partition.
#
# --- Step 1: Create a virtual environment (on the login node) ---
#
# On the login node there is no SLURM context, so force the stack
# explicitly via LLM_FOUNDRY_STACK before creating the venv:
#
#   LLM_FOUNDRY_STACK=amd source $workdir/.modules.sh   # or =intel for "data"
#   python3 -m venv $workdir/.venv_<config>
#   source $workdir/.venv_<config>/bin/activate
#   pip install --upgrade pip
#   deactivate
#   module purge
#
# --- Step 2: Install packages (as a SLURM job on the correct node) ---
#
# Inside the job, SLURM_JOB_PARTITION is set automatically from your
# #SBATCH --partition directive, so `.modules.sh` resolves the stack
# automatically -- no LLM_FOUNDRY_STACK override needed.
#
#   sbatch --export=ALL <<'EOF'
#   #!/bin/bash -l
#   #SBATCH --account=<your-account>
#   #SBATCH --partition=<partition>        # see table above
#   #SBATCH --job-name=install-<config>
#   #SBATCH --output=$workdir/run_outputs/install-<config>-%j.out
#   #SBATCH --time=01:00:00
#   #SBATCH --nodes=1
#   #SBATCH --ntasks-per-node=1
#   #SBATCH --threads-per-core=1
#   #SBATCH --mem=500G
#   #SBATCH --oversubscribe
#
#   source $workdir/.modules.sh        # auto-detects the stack inside the job
#   source $workdir/.venv_<config>/bin/activate
#   pip install -e /path/to/llm-foundry/.[<config>]
#   EOF
#
# Example — installing the "distributed" config on Marvin:
#
#   LLM_FOUNDRY_STACK=amd source $workdir/.modules.sh
#   python3 -m venv $workdir/.venv_distributed
#   source $workdir/.venv_distributed/bin/activate
#   pip install --upgrade pip
#   deactivate
#   module purge
#
#   sbatch --export=ALL <<'EOF'
#   #!/bin/bash -l
#   #SBATCH --account=<your-account>
#   #SBATCH --partition=mlgpu_short
#   #SBATCH --job-name=install-distributed
#   #SBATCH --output=$workdir/run_outputs/install-distributed-%j.out
#   #SBATCH --time=01:00:00
#   #SBATCH --nodes=1
#   #SBATCH --ntasks-per-node=1
#   #SBATCH --threads-per-core=1
#   #SBATCH --mem=500G
#   #SBATCH --oversubscribe
#
#   source $workdir/.modules.sh
#   source $workdir/.venv_distributed/bin/activate
#   pip install -e /path/to/llm-foundry/.[distributed]
#   EOF
#
# Example — installing the "distributed" config on Bender:
#
#   LLM_FOUNDRY_STACK=amd source $workdir/.modules.sh
#   python3 -m venv $workdir/.venv_distributed
#   source $workdir/.venv_distributed/bin/activate
#   pip install --upgrade pip
#   deactivate
#   module purge
#
#   sbatch --export=ALL <<'EOF'
#   #!/bin/bash -l
#   #SBATCH --partition=A100short
#   #SBATCH --job-name=install-distributed
#   #SBATCH --output=$workdir/run_outputs/install-distributed-%j.out
#   #SBATCH --time=01:00:00
#   #SBATCH --ntasks-per-node=1
#   #SBATCH --gpus=1
#
#   source $workdir/.modules.sh
#   source $workdir/.venv_distributed/bin/activate
#   pip install -e /path/to/llm-foundry/.[distributed]
#   EOF
# - Note: Bender does not use some of the SLURM directives used on Marvin 
#   (e.g., --account, --nodes, etc.) since it has a different resource 
#   management setup.
#############################################
