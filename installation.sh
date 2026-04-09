#!/bin/bash -l

#############################################
# Workspace Setup Script for HPC Usage
#############################################
# This script allocates a workspace on the Marvin HPC cluster,
# clones the repository, and prepares the directory structure.
#
# Dual stack documentation:
# https://wiki.hpc.uni-bonn.de/en/dualstacks
#############################################

# ----------- User Customization Section -----------
username="nklugeco_hpc"        # <-- Change to your username
file_system="mlnvme"           # <-- Change to your filesystem
work_group="ag_bit_flek"       # <-- Change to your work group
email="kluge@uni-bonn.de"      # <-- Change to your email
remainder=7                    # <-- Change as needed
num_days=90                    # <-- Change as needed
workspace_name="polyglot"      # <-- Change to your workspace/project name
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
# Installing Dependencies
#############################################
# Dependencies are managed via pyproject.toml with optional groups:
#
#   data         - Data processing (datatrove, spacy, stanza, etc.)
#   distributed  - Distributed training (torch, accelerate, flash_attn, etc.)
#   synth        - Synthetic data generation (torch, vllm, etc.)
#   trl          - Fine-tuning with TRL (trl, vllm, flash_attn, etc.)
#
# Marvin uses a dual software stack (AMD and Intel). Each config
# needs the correct modules and must be installed on the matching
# node type so that hardware-specific packages resolve correctly.
#
#   Config        Modules file         SLURM partition (recommended)
#   ------        ------------         ---------------
#   data          .modules_intel.sh    lm_short (Intel nodes)
#   distributed   .modules_amd.sh     mlgpu_short (AMD/GPU nodes)
#   synth         .modules_amd.sh     mlgpu_short (AMD/GPU nodes)
#   trl           .modules_amd.sh     mlgpu_short (AMD/GPU nodes)
#
# --- Step 1: Create a virtual environment (on the login node) ---
#
#   source $workdir/.modules_amd.sh          # or .modules_intel.sh for "data"
#   python3 -m venv $workdir/.venv_<config>
#   source $workdir/.venv_<config>/bin/activate
#   pip install --upgrade pip
#   deactivate
#   module purge
#
# --- Step 2: Install packages (as a SLURM job on the correct node) ---
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
#   #SBATCH --gres=gpu:a40:1               # We only need a GPU for the "distributed", "synth", and "trl" configs to resolve hardware-specific packages. Omit for "data".
#   #SBATCH --oversubscribe
#
#   source $workdir/.modules_amd.sh        # or .modules_intel.sh
#   source $workdir/.venv_<config>/bin/activate
#   pip install -e /path/to/llm-foundry/.[<config>]
#   EOF
#
# Example — installing the "distributed" config:
#
#   source $workdir/.modules_amd.sh
#   python3 -m venv $workdir/.venv_distributed
#   source $workdir/.venv_distributed/bin/activate
#   pip install --upgrade pip
#   deactivate
#   module purge
#
#   sbatch --export=ALL <<'EOF'
#   #!/bin/bash -l
#   #SBATCH --account=ag_cst_gabriel
#   #SBATCH --partition=mlgpu_short
#   #SBATCH --job-name=install-distributed
#   #SBATCH --output=/lustre/mlnvme/data/nklugeco_hpc-polyglot/run_outputs/install-distributed-%j.out
#   #SBATCH --time=01:00:00
#   #SBATCH --nodes=1
#   #SBATCH --ntasks-per-node=1
#   #SBATCH --threads-per-core=1
#   #SBATCH --mem=500G
#   #SBATCH --gres=gpu:a40:1
#   #SBATCH --oversubscribe
#
#   source /lustre/mlnvme/data/nklugeco_hpc-polyglot/.modules_amd.sh
#   source /lustre/mlnvme/data/nklugeco_hpc-polyglot/.venv_distributed/bin/activate
#   pip install -e /lustre/mlnvme/data/nklugeco_hpc-polyglot/llm-foundry/.[distributed]
#   EOF
#############################################