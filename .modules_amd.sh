# This file sets up the module environment for AMD-based installations/operations.
export MODULEPATH=/opt/software/easybuild-AMD/modules/all:/etc/modulefiles:/usr/share/modulefiles:/opt/software/modulefiles:/usr/share/modulefiles/Linux:/usr/share/modulefiles/Core:/usr/share/lmod/lmod/modulefiles/Core
module purge
module load CUDA/12.6.0 Python/3.12.3-GCCcore-13.3.0
