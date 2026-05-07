# Test Suites

Unit and integration tests for the foundry code base.

## Overview

This folder contains unit and integration test scripts for the foundry code base. All scripts are designed to run as standalone Python programs with module imports resolved from their respective source folders via `sys.path` setup at module level.

## Available Test Scripts

- **tests_data.py** — Unit and integration tests for the data processing scripts.
- **tests_distributed.py** — CPU-only pre-flight suite for the DDP and FSDP training stack (imports, configs, MFU, dataloaders, model setup, optimizers, utilities, integration, trainer).
- **tests_gym.py** — End-to-end verifier and sample-generation checks for the gym pipeline (instruction, long-context, haystack, math, email, and tool-call tasks).
- **tests_synthetic.py** — End-to-end verifier for the synthetic data pipeline.

## Running Tests

From the repository root, run all scripts in sequence:

```bash
python tests/
```

Or run a single script directly:

```bash
python tests/tests_data.py
python tests/tests_distributed.py
python tests/tests_gym.py
python tests/tests_synthetic.py
```

### Module Loading on Marvin

To run tests regarding the module loading logic on Marvin's dual stack (Intel|AMD CPU and NVIDIA GPU), use the following scripts:

```bash
bash tests/test_modules_cpu.sh
bash tests/test_modules_gpu.sh
```

## Notes

- All test scripts set `sys.pycache_prefix` to `tmp/pycache` in the Linux temp directory (via `tempfile.gettempdir()`).
