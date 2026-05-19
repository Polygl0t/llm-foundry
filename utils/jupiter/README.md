# JSC Jupiter Support

This folder contains module and installation scripts for the JSC Jupiter booster environment.

## What is included

- [`jupiter_modules_2025.sh`](jupiter_modules_2025.sh) — a module setup script for the JSC Stages/2025 software stack.
- [`jupiter_installation_2025.sh`](jupiter_installation_2025.sh) — an installation script that creates a Python virtual environment, installs the project dependencies, and builds or installs CUDA-aware PyTorch and attention extensions for the 2025 stack.
- [`jupiter_modules_2026.sh`](jupiter_modules_2026.sh) — a module setup script for the JSC Stages/2026 software stack.
- [`jupiter_installation_2026.sh`](jupiter_installation_2026.sh) — an installation script that creates a Python virtual environment, installs the project dependencies, and builds or installs CUDA-aware PyTorch and attention extensions for the 2026 stack.

## Usage

These scripts are intended to be run on a JSC login node, not inside an SLURM batch job.

Example:

```bash
bash llm-foundry/utils/jupiter/jupiter_installation_2026.sh
```

The 2026 installer is designed to work with the JSC Stages/2026 toolchain and builds PyTorch locally when needed so CUDA and PyTorch versions match the local CUDA 13 stack.

## Notes

- You can learn more about Jupiter in https://apps.fz-juelich.de/jsc/hps/jupiter/index.html.
