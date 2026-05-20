#########################################################
# 2025 Modules for JSC JUPITER
# This is the module file for the 2025 software stack.
# Python 3.12 keeps us compatible with the upstream aarch64
# flash-attn wheel for torch 2.9. CUDA/cuDNN/NCCL are loaded
# explicitly so the JUPITER installer can build a local CUDA
# PyTorch wheel and source-built extension packages when needed.
#########################################################

module --force purge

module load Stages/2025
module load GCCcore/.13.3.0
module load CUDA/12
module load cuDNN/9.5.0.50-CUDA-12
module load NCCL/default-CUDA-12
module load Python/3.12.3
module load CMake Ninja
