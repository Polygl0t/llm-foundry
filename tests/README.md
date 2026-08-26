# Test Suites

This folder contains the unit and integration test suites for the foundry code base, collected and run with [pytest](https://docs.pytest.org/).

## Running Tests

From the repository root, run the whole suite:

```bash
pytest
```

or, equivalently:

```bash
pytest tests/
```

To run a single suite in-process with the full pytest feature set (per-test
output, `-k` selection, `-x`, `-v`, ...):

```bash
pytest tests/tests_distributed.py -v
```

### Module Loading

To run tests regarding the module loading logic on Marvin|Bender dual stack, use the following script:

```bash
bash tests/test_modules.sh
```
