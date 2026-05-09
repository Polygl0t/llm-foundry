# Test Suites

Unit and integration tests for the foundry code base.

## Overview

This folder contains unit and integration test scripts for the foundry code base. All scripts are designed to run as standalone Python programs with module imports resolved from their respective source folders via `sys.path` setup at module level.

## Running Tests

From the repository root, run all scripts in sequence:

```bash
python tests/
```

Or run a single script directly:

```bash
python tests/tests_distributed.py
```

### Module Loading on Marvin

To run tests regarding the module loading logic on Marvin's dual stack (Intel|AMD CPU and NVIDIA GPU), use the following scripts:

```bash
bash tests/test_modules_cpu.sh
bash tests/test_modules_gpu.sh
```

## Notes

- All test scripts set `sys.pycache_prefix` to `tmp/pycache` in the Linux temp directory (via `tempfile.gettempdir()`).
