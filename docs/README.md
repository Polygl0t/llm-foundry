# Documentation

This folder contains cluster-specific documentation for running LLM Foundry on systems other than the default Marvin/Bender clusters. The core code base is designed to run on the University of Bonn HPC clusters (Marvin and Bender), but we also document how to set up and run jobs on the additional clusters we support.

The documentation is organized by cluster:

- [`baf/`](baf/) — Documentation and scripts for running jobs on the **BAF (Bonn Analysis Facility)** cluster. BAF uses containers and HTCondor instead of SLURM. See [`baf/README.md`](baf/README.md) for the full guide.
- [`jupiter/`](jupiter/) — Documentation and scripts for running jobs on the **JSC Jupiter** booster. See [`jupiter/README.md`](jupiter/README.md) for the full guide.

Each subfolder contains its own `README.md` with step-by-step instructions, along with the module, installation, download, and job-submission scripts referenced by those guides.
