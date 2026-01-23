#!/bin/bash -l

#############################################
# This script sets up a workspace to work with the 
# dual stack system of the Marvin HPC cluster.
# More information about the dual stack system can be found here:
# https://wiki.hpc.uni-bonn.de/en/dualstacks
#############################################

#############################################
# Workspace Setup Script for HPC Usage
#############################################

# ----------- User Customization Section -----------
username="nklugeco_hpc"        # <-- Change to your username
file_system="mlnvme"           # <-- Change to your filesystem
work_group="ag_cst_gabriel"    # <-- Change to your work group
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
echo "Configuration Summary for the new workspace:"
echo "Username: $username"
echo "File System: $file_system"
echo "Work Group: $work_group"
echo "Email: $email"
echo "Remainder: $remainder"
echo "Number of Days: $num_days"
echo "Workspace Name: $workspace_name"
echo "Workspace Directory: $workdir"

#############################################
# Clone Polyglot and copy installation files
# and modules
#############################################
git clone --branch main https://github.com/Nkluge-correa/polyglot.git
cd polyglot
cp ./.modules* $workdir
cp ./install_* $workdir
echo "Installation files and module files copied to workspace."

#############################################
# Create Cache and Output Directories
#############################################
mkdir -p $workdir/.cache
mkdir -p $workdir/run_outputs
echo "Cache and output directories created."

#############################################
# Create Python Virtual Environments (AMD & Intel)
#############################################

# Marvin operates with a dual software stack: AMD and Intel.
# Learn more about it here: https://wiki.hpc.uni-bonn.de/en/dualstacks
# AMD environment
source $workdir/.modules_amd.sh
python3 -m venv $workdir/.venv_amd
source $workdir/.venv_amd/bin/activate
which python3
deactivate
module purge
echo "Python virtual environment created for AMD Stack."

# Intel environment
source $workdir/.modules_intel.sh
python3 -m venv $workdir/.venv_intel
source $workdir/.venv_intel/bin/activate
which python3
deactivate
module purge
echo "Python virtual environment created for Intel Stack."

#############################################
# Submit Installation Jobs
#############################################
echo "Submitting installation jobs to the scheduler."
sbatch $workdir/install_amd.sh
sbatch $workdir/install_intel.sh
#############################################
# End of Script
#############################################
echo "Installation jobs submitted."
echo "Please monitor the output logs in $workdir/run_outputs for installation progress."