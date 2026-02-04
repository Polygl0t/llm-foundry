<pre>
██████╗  ██████╗ ██╗  ██╗   ██╗ ██████╗  ██████╗ ████████╗
██╔══██╗██╔═══██╗██║  ╚██╗ ██╔╝██╔════╝ ██╔═████╗╚══██╔══╝
██████╔╝██║   ██║██║   ╚████╔╝ ██║  ███╗██║██╔██║   ██║   
██╔═══╝ ██║   ██║██║    ╚██╔╝  ██║   ██║████╔╝██║   ██║   
██║     ╚██████╔╝███████╗██║   ╚██████╔╝╚██████╔╝   ██║   
╚═╝      ╚═════╝ ╚══════╝╚═╝    ╚═════╝  ╚═════╝    ╚═╝   

Developing foundation models for low-resource languages.

## Overview

- data: Scripts for downloading and preprocessing datasets.
- ddp: Our current implementation of Distributed Data Parallel training.
- dpo: Implementation for Direct Preference Optimization via TRL.
- evals: Scripts for evaluating language models via the lm-evaluation-harness.
- fsdp: Our current implementation of Fully Sharded Data Parallel 2.
- hf_hub: Scripts for interacting with the Hugging Face Hub.
- merge: Scripts for running different merging techniques via mergekit.
- sft: Implementation of Supervised Fine-Tuning via TRL.
- synthetic: Scripts for generating synthetic datasets with vLLM.
- tokenization: Scripts for training, evaluating, and using tokenizers.

All of our code base is made to run on the Marvin cluster (University of Bonn).

## Aknowlegments

Polyglot is a project funded by the Federal Ministry of Education and Research (BMBF) and the Ministry of Culture and Science of the State of North Rhine-Westphalia (MWK) as part of TRA Sustainable Futures (University of Bonn) and the Excellence Strategy of the federal and state governments.

We also gratefully acknowledge access to the Marvin cluster, hosted by the University of Bonn, along with support from its High Performance Computing & Analytics Lab.
</pre>
