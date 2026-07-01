# JSC Jupiter Support

This folder contains module and installation scripts for the JSC Jupiter booster environment.

## What is included

- [`jupiter_modules_2026.sh`](jupiter_modules_2026.sh) — a module setup script for the JSC Stages/2026 software stack. This is equivalent to the [`.modules.sh`](../.modules.sh) script that we use for the other clusters, but it is tailored to the JSC environment.
- [`jupiter_installation_2026.sh`](jupiter_installation_2026.sh) — an installation script that creates a Python virtual environment, installs the project dependencies, and builds or installs CUDA-aware PyTorch and attention extensions for the 2026 stack.

> - **Note:** On Jupiter/JSC, we do not have internet access from the compute nodes, so the installation script must be run on a login node.

## Usage

This will install all necessary dependencies and build a distributed training environment for the 2026 stack.

Example:

```bash
bash llm-foundry/utils/jupiter/jupiter_installation_2026.sh
```

## Notes

- You can learn more about Jupiter in https://apps.fz-juelich.de/jsc/hps/jupiter/index.html.
