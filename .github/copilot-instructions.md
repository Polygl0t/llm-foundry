# Copilot Instructions for LLM Foundry

This file provides guidance for code agents (e.g. GitHub Copilot) working with this codebase.

## Core Project Structure

- `alignment/` — Post-training and alignment: SFT, DPO, GRPO, and reward model training.
  - `alignment/gym/` — Training/evaluation on custom environments with verifiable rewards.
- `data/` — Data processing: Common Crawl filtering, quality filters, formatting, and tokenization.
- `distributed/` — Pretraining: DDP and FSDP2 entry points, model setup, and optimizers.
- `evals/` — Evaluation scripts built on `lm-evaluation-harness`.
- `merge/` — Model merging via `mergekit`.
- `shared/` — Shared utility modules reused across the major pipelines.
- `synthetic/` — Synthetic data generation with vLLM + DataTrove, plus agent-based traces with `smolagents`.
- `tests/` — Standalone unit and integration test scripts.
- `tokenizer/` — Tokenizer training and evaluation (SentencePiece and HF Tokenizers).
- `tools/` — Miscellaneous tools and utilities (downloads, uploads, model inspection, etc.).
- `docs/` — Cluster-specific documentation (BAF, JSC Jupiter).

## Coding Conventions

- Keep PRs brief and focused. Bug fixes can often be a few lines and do not need large comments, docstrings, or new functions in that case. Aim to minimize the diff.
- Follow PEP 8. Style is enforced by pre-commit + Ruff. Run `pre-commit run --all-files` (or `ruff check .` and `ruff format .`) before committing.
- Comment non-obvious logic — this codebase favors clear explanatory comments.
- Keep changes scoped to the affected module; do not touch unrelated files.
- Dependencies are grouped as extras in `pyproject.toml`: `.[data]`, `.[tokenizer]`, `.[distributed]`, `.[synth]`, `.[agents]`, `.[tests]`.

## Cluster / Environment Notes

- This codebase primarily runs on the University of Bonn HPC clusters (Marvin, Bender), which have dual AMD/Intel software stacks.
- Source `.modules.sh` from the repo root to load the correct stack; it auto-detects from SLURM. Force it with `LLM_FOUNDRY_STACK=amd` or `LLM_FOUNDRY_STACK=intel` when there is no SLURM context.
- Some dependency stacks are hard to build (e.g., `torch`, `liger-kernel`, `flash-attn`, `causal-conv1d` sharing one `nvcc`). Where a dedicated `create_venv` script exists (e.g., `distributed/slurm/create_venv_marvin.sh`, `docs/jupiter/jupiter_installation_2026.sh`, `docs/baf/create_venv.sh`), prefer it over ad-hoc `pip`/`uv` installs.

## Testing

- Tests are collected and run with pytest. From the repository root run `pytest` (or `pytest tests/`) to run all suites, or `pytest tests/tests_distributed.py` for a single suite.
- Each suite (`tests/tests_*.py`) is executed in its own subprocess by `tests/conftest.py`, because the suites patch `sys.modules` and use colliding flat-layout `utils` modules that would leak across suites in a single interpreter.
- For module-loading logic on the dual stack, run `bash tests/test_modules.sh`.
- Install test dependencies with `pip install -e ".[tests]"`.
