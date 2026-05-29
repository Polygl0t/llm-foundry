# Test Suites

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

### Module Loading

To run tests regarding the module loading logic on Marvin|Bender dual stack, use the following script:

```bash
bash tests/test_modules.sh
```
