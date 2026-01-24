<pre>
██████╗  ██████╗ ██╗  ██╗   ██╗ ██████╗  ██████╗ ████████╗
██╔══██╗██╔═══██╗██║  ╚██╗ ██╔╝██╔════╝ ██╔═████╗╚══██╔══╝
██████╔╝██║   ██║██║   ╚████╔╝ ██║  ███╗██║██╔██║   ██║   
██╔═══╝ ██║   ██║██║    ╚██╔╝  ██║   ██║████╔╝██║   ██║   
██║     ╚██████╔╝███████╗██║   ╚██████╔╝╚██████╔╝   ██║   
╚═╝      ╚═════╝ ╚══════╝╚═╝    ╚═════╝  ╚═════╝    ╚═╝   

Developing foundation models for low-resource languages.

## Overview

- `data`: Code for downloading and preprocessing datasets (e.g., download WARC files from CommonCrawl, deduplicate text, apply quality filters, remove PII), it also contains code for creating/using several types of filters and classifiers.
- `ddp`: Our current implementation of Distributed Data Parallel training (used in the Tucano, Tucano2, and the Lil series).
- `dpo`: Implementation for Direct Preference Optimization (DPO) via TRL.
- `evals`: Contains code for evaluating language models (e.g., scripts for running different evaluation harnesses).
- `fsdp`: Our current implementation of Fully Sharded Data Parallelism (used in Tucano2 series).
- `hf_hub`: Code for interacting with the Hugging Face Hub (mainly downloading/uploading datasets and models to the hub).
- `merge`: Code for running different merging techniques via mergekit.
- `sft`: Implementation of Supervised Fine-Tuning (SFT) via TRL.
- `synthetic`: Code for generating synthetic datasets with vLLM.
- `tokenization`: Code for training and using tokenizers (e.g., training a tokenizer on a dataset, using a tokenizer to encode text, chat templates).

You can use the `installation.sh` script to set up a new working environment on the cluster. All of our code base is made to run on the Marvin cluster (University of Bonn).

## Aknowlegments

Polyglot is a project funded by the Federal Ministry of Education and Research (BMBF) and the Ministry of Culture and Science of the State of North Rhine-Westphalia (MWK) as part of TRA Sustainable Futures (University of Bonn) and the Excellence Strategy of the federal and state governments.

We also gratefully acknowledge the granted access to the [Marvin cluster](https://www.hpc.uni-bonn.de/en/systems/marvin) hosted by [University of Bonn](https://www.uni-bonn.de/en) along with the support provided by its High Performance Computing & Analytics Lab.
</pre>
