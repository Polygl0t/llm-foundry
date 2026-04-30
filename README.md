
# LLM Foundry

LLM Foundry is the source repository for the development of models, datasets, and accompanying artifacts of the **Polyglot** project at the University of Bonn. It bundles training, evaluation, post-training, data processing, and tokenization pipelines into a single, cluster-ready code base.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
  - [Workspace Setup on Marvin](#workspace-setup-on-marvin)
  - [Module Stack Selection](#module-stack-selection)
  - [Installing Dependencies](#installing-dependencies)
- [Running the Tests](#running-the-tests)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This repository contains all source code used for the development of the models, datasets, and all other accompanying artifacts tied to the Polyglot project at the University of Bonn. It is designed to run on the [Marvin cluster](https://www.hpc.uni-bonn.de/) (University of Bonn), which has a dual software stack (AMD and Intel) that the code base is aware of.

## Repository Structure

The code base is organized into the following main folders:

- [`data/`](data/) — Scripts for downloading and preprocessing datasets (e.g., HF Hub, Common Crawl).
- [`distributed/`](distributed/) — Scripts for training and evaluating language models with DDP and FSDP.
- [`dpo/`](dpo/) — Implementation for Direct Preference Optimization via TRL.
- [`evals/`](evals/) — Scripts for evaluating language models via the `lm-evaluation-harness`.
- [`gym/`](gym/) — Scripts for training and evaluating language models on custom environments (WIP).
- [`hf_hub/`](hf_hub/) — Scripts for interacting with the Hugging Face Hub.
- [`merge/`](merge/) — Scripts for running different merging techniques via `mergekit`.
- [`sft/`](sft/) — Implementation of Supervised Fine-Tuning via TRL.
- [`synthetic/`](synthetic/) — Scripts for generating synthetic datasets with vLLM.
- [`tests/`](tests/) — Unit and integration tests for our code base.
- [`tokenization/`](tokenization/) — Scripts for training, evaluating, and using tokenizers.
- [`utils/`](utils/) — Miscellaneous utilities for our code base.

## Installation

All of our codebase is designed to run on the Marvin cluster (University of Bonn). You will only need to set things up on the cluster itself - not on your local machine. For your local machine, you can just clone the repository and work with the files (e.g., editing code, writing new scripts, etc.) without worrying too much about dual stack setups or module loading.

### Workspace Setup on Marvin

Use [`utils/marvin_create_workspace.sh`](utils/marvin_create_workspace.sh) to allocate a workspace, clone the repository, and prepare the directory layout. Open the script first and edit the user customization section at the top (`username`, `file_system`, `work_group`, `email`, `workspace_name`) to match your account, then run it from a Marvin login node:

```bash
bash utils/marvin_create_workspace.sh
```

The script also contains commented step-by-step instructions for creating the per-config virtual environments (`.venv_data,` `.venv_distributed`, `.venv_synth`, `.venv_trl`) and submitting the pip install jobs to the right partition.

### Module Stack Selection
Marvin has a dual software stack (AMD and Intel). The single `.modules.sh` file at the repository root loads the right one for you. It auto-detects the stack from the SLURM environment, so most of the time you just source it and forget about it:

```bash
# Inside a SLURM job: auto-detected from #SBATCH directives
#   - GPU job (--gres=gpu:...)             -> AMD
#   - partition name contains "gpu"        -> AMD
#   - any other partition                  -> Intel
source "$workdir/.modules.sh"
```

On a login node, there is no SLURM context, so you must force the stack explicitly when creating venvs or running ad-hoc commands:

```bash
LLM_FOUNDRY_STACK=amd   source "$workdir/.modules.sh"   # GPU/training stack
LLM_FOUNDRY_STACK=intel source "$workdir/.modules.sh"   # CPU/data stack
```

Sourcing prints whose stack was selected, why, and the resulting module list, so your job logs always show the resolved environment.

### Installing Dependencies
Use the [`pyproject.toml`](https://github.com/Polygl0t/llm-foundry/blob/main/pyproject.toml) to install a specific set of dependencies. The available extras are:

* `data` — For downloading and preprocessing datasets.  
* `distributed` — For training language models with our DDP and FSDP implementations.  
* `synth` — For generating synthetic samples with vLLM.  
* `trl` — For post-training and alignment with TRL.  
* `tests` — For running our test suite.  

For example:

```bash
pip install -e "./llm-foundry/.[distributed]"  # for DDP/FSDP training
pip install -e "./llm-foundry/.[trl]"          # for SFT/DPO
pip install -e "./llm-foundry/.[tests]"        # for running the test suite
```

## Running the Tests
Install the test dependencies first:

```bash
pip install -e "./llm-foundry/.[tests]"
```
Run all test scripts in sequence:
```bash
python tests/
```
Or run a specific script ([`tests/tests_distributed.py`](https://github.com/Polygl0t/llm-foundry/compare/tests/tests_distributed.py?expand=1), [`tests/tests_gym.py`](https://github.com/Polygl0t/llm-foundry/compare/tests/tests_gym.py?expand=1), [`tests/tests_synthetic.py`](https://github.com/Polygl0t/llm-foundry/compare/tests/tests_synthetic.py?expand=1)):
```bash
python tests/tests_distributed.py
python tests/tests_gym.py
python tests/tests_synthetic.py
```


## Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for details on how to set up your development environment, the contribution workflow (forking, branching, squashing commits, opening a pull request), and the project's style guide.

## License

This project is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) for the full license text.

## Acknowledgments

Polyglot is a project funded by the Federal Ministry of Education and Research (BMBF) and the Ministry of Culture and Science of the State of North Rhine-Westphalia (MWK) as part of TRA Sustainable Futures (University of Bonn) and the Excellence Strategy of the federal and state governments.

We also gratefully acknowledge access to the Marvin cluster, hosted by the University of Bonn, along with support from its High Performance Computing & Analytics Lab.

