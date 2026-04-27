# Contributing to LLM Foundry

Thank you for contributing to LLM Foundry! This document explains how to get started, how to set up your development environment, and how to submit changes.

---

## Table of Contents

- [Getting started](#getting-started)
- [Setting up your environment](#setting-up-your-environment)
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

- The modules for your stack (AMD or Intel) are defined in `.modules_amd.sh` and `.modules_intel.sh`. Source the appropriate one to load the correct environment. You **don't** need to do this on your local machine - only on the cluster.

    ```bash
    # Load the correct modules for your stack (AMD or Intel)
    source .modules_amd.sh   # or .modules_intel.sh

    ```

- The installation script `installation.sh` is only for setting up a new workspace on the cluster. If you are developing locally, you can install the dependencies directly with pip (see below). If you're working on the cluster, run the installation script to set up your workspace and install the dependencies:

    ```bash
    # Run the installation helper
    bash installation.sh
    ```

To install a specific set of dependencies (e.g., for training or data processing), use the extras defined in `pyproject.toml`:

```bash
pip install -e ".[distributed]"  # for DDP/FSDP training
pip install -e ".[trl]"          # for SFT/DPO
pip install -e ".[tests]"        # for running the test suite
```

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
python tests/tests_gym.py
python tests/tests_synthetic.py
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
