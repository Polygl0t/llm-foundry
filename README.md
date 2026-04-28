```
 .----------------.  .----------------.  .----------------.                      .----------------.  .----------------.  .----------------.  .-----------------. .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. |                    | .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
| |   _____      | || |   _____      | || | ____    ____ | |                    | |  _________   | || |     ____     | || | _____  _____ | || | ____  _____  | || |  ________    | || |  _______     | || |  ____  ____  | |
| |  |_   _|     | || |  |_   _|     | || ||_   \  /   _|| |                    | | |_   ___  |  | || |   .'    `.   | || ||_   _||_   _|| || ||_   \|_   _| | || | |_   ___ `.  | || | |_   __ \    | || | |_  _||_  _| | |
| |    | |       | || |    | |       | || |  |   \/   |  | |                    | |   | |_  \_|  | || |  /  .--.  \  | || |  | |    | |  | || |  |   \ | |   | || |   | |   `. \ | || |   | |__) |   | || |   \ \  / /   | |
| |    | |   _   | || |    | |   _   | || |  | |\  /| |  | |                    | |   |  _|      | || |  | |    | |  | || |  | '    ' |  | || |  | |\ \| |   | || |   | |    | | | || |   |  __ /    | || |    \ \/ /    | |
| |   _| |__/ |  | || |   _| |__/ |  | || | _| |_\/_| |_ | |                    | |  _| |_       | || |  \  `--'  /  | || |   \ `--' /   | || | _| |_\   |_  | || |  _| |___.' / | || |  _| |  \ \_ | || |    _|  |_    | |
| |  |________|  | || |  |________|  | || ||_____||_____|| |                    | | |_____|      | || |   `.____.'   | || |    `.__.'    | || ||_____|\____| | || | |________.'  | || | |____| |___| | || |   |______|   | |
| |              | || |              | || |              | |                    | |              | || |              | || |              | || |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' |                    | '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'                      '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------' 
```

LLM Foundry

Overview
--------

This repository contains all source code used for the development of the models, datasets, and all other accompanying artifacts tied to the Polyglot project at the University of Bonn. The code base is organized into the following main folders:

* Data: Scripts for downloading and preprocessing datasets (e.g., HF Hub, Common Craul).
* Distributed: Scripts for training and evaluating language models with DDP and FSDP.
* DPO: Implementation for Direct Preference Optimization via TRL.
* Evaluations: Scripts for evaluating language models via the lm-evaluation-harness.
* Gym: Scripts for training and evaluating language models on custom environments (WIP).
* hf_hub: Scripts for interacting with the Hugging Face Hub.
* Merge: Scripts for running different merging techniques via mergekit.
* SFT: Implementation of Supervised Fine-Tuning via TRL.
* Synthetic: Scripts for generating synthetic datasets with vLLM.
* Tests: Unit and integration tests for our code base.
* Tokenization: Scripts for training, evaluating, and using tokenizers.
* Utils: Some miscellaneous utilities for our code base.

All of our code base is made to run on the Marvin cluster (University of Bonn).

Installation
------------

You can use the `installation.sh` to help you create workspaces in Marvin. Marvin has a dual stack setup (Intel / AMD), so make sure to create your local environments with this in mind (see `installation.sh` for details).

The `.modules_{amd|intel}.sh` file contains all the modules you need to load in order to run our code base in Marvins AMD or Intel stack (see `installation.sh` for details).

You can also use the `pyproject.toml` to install certain specific/working builds of our code base. The `pyproject.toml` currently has some basic builds to work with:

* `data`: For downloading and preprocessing datasets.
* `distributed`: For training and evaluating language models with DDP and FSDP.
* `synth`: For generating synthetic datasets with vLLM.
* `trl`: For training and evaluating language models with TRL.
* `tests`: For running our test suite.

Acknowledgments
-------------

Polyglot is a project funded by the Federal Ministry of Education and Research (BMBF) and the Ministry of Culture and Science of the State of North Rhine-Westphalia (MWK) as part of TRA Sustainable Futures (University of Bonn) and the Excellence Strategy of the federal and state governments.

We also gratefully acknowledge access to the Marvin cluster, hosted by the University of Bonn, along with support from its High Performance Computing & Analytics Lab.
