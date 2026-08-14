# Copilot Instructions for LLM Foundry

This file provides guidance for code agents (e.g. GitHub Copilot) working with this codebase.

## Core Project Structure

- `alignment/` — Post-training and alignment: SFT, DPO, GRPO, and reward model training.
  - `alignment/gym/` — Training/evaluation on custom environments with verifiable rewards.
- `data/` — Data processing: Common Crawl filtering, quality filters, formatting, and tokenization.
- `distributed/` — Pretraining: DDP and FSDP2 entry points, model setup, and optimizers.
- `evals/` — Evaluation scripts built on `lm-evaluation-harness`.
- `merge/` — Model merging via `mergekit`.
- `synthetic/` — Synthetic data generation with vLLM + DataTrove, plus agent-based traces with `smolagents`.
- `tests/` — Standalone unit and integration test scripts.
- `tokenizer/` — Tokenizer training and evaluation (SentencePiece and HF Tokenizers).
- `utils/` — Miscellaneous utilities (downloads, uploads, model inspection, etc.).

## Coding Conventions

- Keep PRs brief and focused. Bug fixes can often be a few lines and do not need large comments, docstrings, or new functions in that case. Aim to minimize the diff.
- Follow PEP 8. Style is enforced by pre-commit + Ruff. Run `pre-commit run --all-files` (or `ruff check .` and `ruff format .`) before committing.
- Comment non-obvious logic — this codebase favors clear explanatory comments.
- Keep changes scoped to the affected module; do not touch unrelated files.
- Dependencies are grouped as extras in `pyproject.toml`: `.[data]`, `.[tokenizer]`, `.[distributed]`, `.[trl]`, `.[synth]`, `.[agents]`, `.[tests]`.

## Cluster / Environment Notes

- This codebase primarily runs on the University of Bonn HPC clusters (Marvin, Bender), which have dual AMD/Intel software stacks.
- Source `.modules.sh` from the repo root to load the correct stack; it auto-detects from SLURM. Force it with `LLM_FOUNDRY_STACK=amd` or `LLM_FOUNDRY_STACK=intel` when there is no SLURM context.

## Testing

- Tests are standalone Python scripts, not pytest. From the repository root run `python tests/` to run all suites in sequence, or `python tests/tests_distributed.py` for a single suite.
- For module-loading logic on the dual stack, run `bash tests/test_modules.sh`.
- Install test dependencies with `pip install -e ".[tests]"`.
