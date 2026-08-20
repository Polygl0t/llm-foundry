# Contributing to LLM Foundry

Thank you for contributing to LLM Foundry! This document explains how to get started, how to set up your development environment, and how to submit changes.

---

## Table of Contents

- [Getting started](#getting-started)
- [Setting up your environment](#setting-up-your-environment)
- [Catching mistakes before committing](#catching-mistakes-before-committing)
- [The contribution workflow](#the-contribution-workflow)
  - [1. Fork the repository](#1-fork-the-repository)
  - [2. Clone your fork](#2-clone-your-fork)
  - [3. Keep your fork in sync](#3-keep-your-fork-in-sync)
  - [4. Create a branch](#4-create-a-branch)
  - [5. Make your changes](#5-make-your-changes)
  - [6. Commit your work](#6-commit-your-work)
  - [7. Squash your commits](#7-squash-your-commits)
  - [8. Open a pull request](#8-open-a-pull-request)
- [Pull request checklist](#pull-request-checklist)
- [Running the tests](#running-the-tests)
- [Style guide](#style-guide)
- [Getting help](#getting-help)

---

## Getting started

Before you start working on something, please check the open issues and pull requests to make sure nobody else is already working on the same thing. If you have a question or want to discuss an idea, feel free to open an issue first.

Good first contributions include:

- Fixing typos or improving the documentation
- Improving existing scripts for clarity or robustness
- Adding tests for untested code paths
- Filing well-described bug reports

---

## Setting up your environment

LLM Foundry runs on the [Marvin cluster](https://www.hpc.uni-bonn.de/) (University of Bonn). Refer to the [README](README.md) for a full overview of the repository structure and installation steps.

In short:

- **Create a workspace on the cluster.** Use `tools/slurm/marvin_create_workspace.sh` to allocate a workspace, clone the repository, and prepare the directory layout. Open the script first and edit the user customization section at the top (`username`, `file_system`, `work_group`, `email`, `workspace_name`) to match your account, then run it from a Marvin login node:

    ```bash
    bash tools/slurm/marvin_create_workspace.sh
    ```

    The script also contains commented step-by-step instructions for creating the per-config virtual environments (`.venv_data`, `.venv_distributed`, `.venv_synth`, `.venv_trl`) and submitting the `pip install` jobs to the right partition. You **don't** need any of this on your local machine - only on the cluster.

- **Stack sourcing.** Marvin has a dual software stack (AMD and Intel). The single `.modules.sh` file at the repository root loads the right one for you. It auto-detects the stack from the SLURM environment so most of the time you just source it and forget about it:

    ```bash
    # Inside a SLURM job: auto-detected from #SBATCH directives
    #   - GPU job (--gres=gpu:...)             -> AMD
    #   - partition name contains "gpu"        -> AMD
    #   - any other partition                  -> Intel
    source "$workdir/.modules.sh"
    ```

    On a login node there is no SLURM context, so you must force the stack explicitly when creating venvs or running ad-hoc commands:

    ```bash
    LLM_FOUNDRY_STACK=amd   source "$workdir/.modules.sh"   # GPU/training stack
    LLM_FOUNDRY_STACK=intel source "$workdir/.modules.sh"   # CPU/data stack
    ```

    Sourcing prints which stack was selected, why, and the resulting `module list`, so your job logs always show the resolved environment.

To install a specific set of dependencies (e.g., for training or data processing), use the extras defined in `pyproject.toml`:

```bash
pip install -e ".[distributed]"  # for DDP/FSDP training
pip install -e ".[tests]"        # for running the test suite
```

> **NOTE: Some dependency stacks are hard to build.** For example, when several libraries (e.g., `torch`, `liger-kernel`, `flash-attn`, `causal-conv1d`, etc.) have to live together and share the same `nvcc` compiler, building this environment can be a real nightmare. For some of the environments we use, we have dedicated `create_venv` scripts that take care of all of the complexity of this build (e.g., `distributed/slurm/create_venv_marvin.sh`, `docs/jupiter/jupiter_installation_2026.sh`, `docs/baf/create_venv.sh`). In these cases, we recommend using those scripts instead of praying to the gods that `pip` or `uv` will find their way through the forest of incompatibilities.

---

## Catching mistakes before committing

LLM Foundry uses [**pre-commit**](https://pre-commit.com) with [**Ruff**](https://docs.astral.sh/ruff/) to catch and fix style issues, formatting errors, and other common mistakes automatically before they ever reach a commit. This saves you from CI failures and keeps the codebase consistent.

### What gets checked

The pre-commit config (`.pre-commit-config.yaml`) runs these hooks automatically on every `git commit`:

| Hook                                       | What it does                                                                                           |
|--------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **Ruff** (`ruff --fix`)                    | Lints Python code and auto-fixes issues where possible (replaces flake8, isort, and dozens of plugins) |
| **Ruff format** (`ruff format`)            | Enforces consistent code formatting (replaces Black)                                                   |
| `check-yaml` / `check-json` / `check-toml` | Validates syntax of config files                                                                       |
| `check-merge-conflict`                     | Blocks files with unresolved `<<<<<<<` merge markers                                                   |
| `check-added-large-files`                  | Rejects files larger than 1 MB                                                                         |
| `detect-private-key`                       | Prevents accidentally committing SSH keys or tokens                                                    |
| `trailing-whitespace`                      | Strips trailing spaces (respects Markdown line breaks)                                                 |
| `end-of-file-fixer`                        | Ensures files end with a single newline                                                                |
| `nbstripout`                               | Strips notebook outputs before committing                                                              |

The Ruff rules are configured in `pyproject.toml` under `[tool.ruff]`, `[tool.ruff.lint]`, and `[tool.ruff.format]`.

### Installing pre-commit

Install pre-commit and set up the hooks once per clone:

```bash
pip install pre-commit
pre-commit install
```

After this, the hooks will run automatically on every `git commit`. If a hook finds an issue, it will either fix it in-place (Ruff auto-fixes) or block the commit with an error message explaining what went wrong.

### Running hooks manually

You can run all hooks against staged files at any time:

```bash
pre-commit run
```

To run a specific hook (e.g., just Ruff):

```bash
pre-commit run ruff --all-files
```

To run against all files in the repo (useful when setting up for the first time):

```bash
pre-commit run --all-files
```

If a hook auto-fixes files, the changes will be unstaged — you'll need to `git add` them and commit again. This is intentional: it lets you review the fixes before committing.

### When your commit is blocked

If pre-commit blocks your commit, **don't panic**. The error output tells you exactly what went wrong and which file. Here's the recommended workflow to diagnose and fix the issues:

**Step 1 — Dump all Ruff errors to a log file:**

```bash
ruff check . --output-file ruff-errors.log
```

The `--output-file` flag writes a clean list of every rule violation to `ruff-errors.log`, one per line. Open the file and work through the problems systematically:

```bash
cat ruff-errors.log
```

Each line follows the format `path/to/file.py:line:col: RULE_CODE explanation`. For example:

```
distributed/trainer.py:42:1: F401 `os` imported but unused
alignment/utils.py:88:80: E501 Line too long (92 > 88 characters)
```

**Step 2 — Understand what each rule means.** Ruff's rule codes are searchable. Run `ruff rule RULE_CODE` to see a full explanation with examples:

```bash
ruff rule F401    # explains the unused-import rule
ruff rule E501    # explains the line-too-long rule
```

**Step 3 — Fix the issues.** Most problems are straightforward:

- **Unused imports / variables** — delete them.
- **Line too long** — break the line or shorten a variable name.
- **Bare except** — use `except Exception:` instead.
- **Missing trailing comma** — Ruff auto-fixes these; just run `ruff check --fix .` again.

After fixing, re-run the check to confirm everything is clean:

```bash
ruff check .     # should produce no output
```

Then stage your fixes and commit again:

```bash
git add .
git commit -m "Your commit message"
```

**Step 4 — If auto-fixes are enough.** Many Ruff violations can be fixed automatically. Try this first before manually editing:

```bash
ruff check --fix .
ruff format .
```

This fixes what it can and leaves the rest for you to handle manually.

> **Tip:** If you're overwhelmed by a large number of errors on an existing file, focus only on the lines you changed. Run `ruff check path/to/your_file.py` instead of `ruff check .` to scope the check to your changes.

---

## The contribution workflow

### 1. Fork the repository

Go to [https://github.com/Polygl0t/llm-foundry](https://github.com/Polygl0t/llm-foundry) and click **Fork** in the top-right corner. This creates your own copy of the repository under your GitHub account.

### 2. Clone your fork

```bash
git clone git@github.com:<your-github-username>/llm-foundry.git
cd llm-foundry
```

Then add the original repository as a remote called `upstream` so you can pull in future changes:

```bash
git remote add upstream https://github.com/Polygl0t/llm-foundry.git
```

You can verify your remotes with:

```bash
git remote -v
# origin    git@github.com:<your-username>/llm-foundry.git (fetch)
# origin    git@github.com:<your-username>/llm-foundry.git (push)
# upstream  https://github.com/Polygl0t/llm-foundry.git (fetch)
# upstream  https://github.com/Polygl0t/llm-foundry.git (push)
```

### 3. Keep your fork in sync

Before starting any new work, make sure your local `main` branch is up to date with the upstream:

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

Doing this regularly avoids painful merge conflicts later.

### 4. Create a branch

**Never work directly on `main`.** Create a new branch for every change you make. Give it a short, descriptive name:

```bash
git checkout -b improve-readme
```

Keep branches focused - one logical change per branch. This makes them easier to review and merge.

### 5. Make your changes

Edit files, write code, and run tests as needed. Keep your changes as focused and self-contained as possible.

### 6. Commit your work

Stage and commit your changes as you go. Write clear, informative commit messages:

```bash
git add path/to/changed_file.py
git commit -m "Fix tokenizer padding logic for variable-length inputs"
```

A good commit message:

- Uses the **imperative mood** ("Fix bug" not "Fixed bug")
- Is **concise but descriptive** (aim for under 72 characters in the subject line)
- Explains **what** changed and **why**, not just how

You can make as many commits as you like while working. You'll clean them up before opening a PR (see next step).

> Remember to run `pre-commit` hooks before committing to catch style issues early!

### 7. Squash your commits

Before opening a pull request, squash your work-in-progress commits into a **single, clean commit**. This keeps the project history tidy and makes code review easier.

Use an interactive rebase to squash. For example, if you made 4 commits:

```bash
git rebase -i HEAD~4
```

In the editor that opens, change `pick` to `squash` (or `s`) for all commits except the first:

```
pick  a1b2c3d  Initial work on tokenizer fix
squash  e4f5a6b  Address edge case
squash  7c8d9e0  Add test
squash  1f2e3d4  Fix typo
```

Save and close the editor. You'll then be prompted to write a final commit message for the combined commit.

> **Tip:** If you run into trouble with `git`, https://ohshitgit.com is a great resource for common fixes.

### 8. Open a pull request

Push your branch to your fork:

```bash
git push -u origin improve-readme
```

Then go to your fork on GitHub. You should see a banner prompting you to **Compare & pull request**. Click it, and:

- Write a **short, clear title** summarizing what you've done.
- Add a **description** explaining the motivation and what changed.
- If your PR addresses an open issue, reference it with `Fixes #<issue-number>`.
- If the work is still in progress, prefix the title with `[WIP]`.

A maintainer will review your PR and either merge it or suggest changes. If changes are requested, push new commits to your branch - they will appear automatically in the PR.

---

## Pull request checklist

Before marking your PR as ready for review, go through this checklist:

- [ ] The title clearly summarizes the change.
- [ ] The description explains the motivation and what was changed.
- [ ] The branch is up to date with `upstream/main` (rebase if needed).
- [ ] Commits have been squashed into a single clean commit.
- [ ] Existing tests still pass (see [Running the tests](#running-the-tests)).
- [ ] New functionality is covered by tests (if applicable).
- [ ] No unrelated files or debugging scripts have been committed.

---

## Running the tests

Install the test dependencies first:

```bash
pip install -e ".[tests]"
```

Run all three test scripts in sequence:

```bash
python tests/
```

Or run a specific script:

```bash
python tests/tests_distributed.py
```

---

## Style guide

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
- Use descriptive variable names and keep functions focused.
- We like comments here. Comment your code to explain non-obvious logic, decisions, and what certain things are doing.
- Stick to plain Markdown for documentation (no raw HTML).

---

## Getting help

If you have any questions, feel free to reach out by email (kluge@uni-bonn.de) or open an issue on GitHub. No need to get everything perfect on the first try - iteration is part of the process. 🤗
