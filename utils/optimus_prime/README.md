# Optimus Prime — LLM Pre-Training Guide

> **⚠️ Experimental / Work-in-Progress**
>
> This guide is based on experiments with a small model (~50M parameters, dummy
> dataset). The goal was to verify that all scripts work end-to-end on BAF.
> Longer training runs with larger models may expose new issues. Experiments are
> underway to cover more scenarios. If you encounter an issue not covered here,
> please update this document or contact me.

> **📝 Note:** BAF (The Bonn Analysis Facility) is referred to as **Optimus Prime**

> **📝 Note:** Before running any scripts or commands, please update the file and folder paths to match your local environment and directory structure.
---

## Before You Start

Once you get access to the BAF cluster, **read the official documentation
first**:

🔗 [BAF (Confluence)](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/10814258/BAF)

Some information there may be outdated, but it gives you the essential background
on how the cluster works (HTCondor, containers, CephFS, etc).

Every user by default is equipped with a personal data storage directory at

_/cephfs/user/_

Within the Job-containers, this directory can be easily accessed via the environment variable $BUDDY. 
There is a quota on the **number of files (100 000) and the available space (500 GB)** for your BUDDY directory

---

## Table of Contents

| Section | Description |
|---|---|
| [1. Create the Python venv](#1-create-the-python-venv) | One-time setup: build a venv tarball with all packages needed for DDP pre-training |
| [2. Run the Training Pipeline](#2-run-the-training-pipeline) | Submit a training job via HTCondor and access the logs and checkpoints |

---

## 1. Create the Python venv
All jobs submitted to the cluster are executed inside containers providing the desired runtime environment for each job.
Inside the container, you will be inside a Job Working Directory (jwd), which is Scratch space cleaned up after the job. Therefore, you need to move your venv to a persistent location (CephFS) so that it can be reused by subsequent jobs.

This is a one-time step. You build a venv with all required packages on the container, compress it into a tarball and put it on CephFS, then every subsequent training job extracts it to `/jwd` at startup. (All of these steps are already handled by the files in this repo.)

### 1.1 Start an interactive GPU job
You need a GPU node because some packages (like `flash-attn`) need CUDA kernels during installation.
Adjust the resources as needed, since we are only going to install packages, we don't need a lot of resources.

E.g.
```bash
condor_submit -interactive \
    -append '+ContainerOS = "Rocky9"' \
    -append '+CephFS_IO = "low"' \
    -append '+MaxRuntimeHours=1' \
    -append 'Request_gpus = 1' \
    -append 'requirements = (CUDADeviceName == "NVIDIA H200")' \
    -append 'Request_cpus = 16' \
    -append 'Request_memory = 32000 MB'
```

**Note:**
- To know more about the available `ContainerOS`, `CephFS_IO`, `MaxRuntimeHours` options, go to the [BAF documentation](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/10814633/HTCondor+on+BAF).
- Before deciding on the values for `Request_cpus`, `Request_gpus`, `requirements` and `Request_memory`,  make sure they are available. You can run this command to check the available resources on the cluster:

```bash
condor_status -compact -constraint 'GPUs > 0' -af:h Machine State GPUs Cpus Memory Disk CUDADeviceName
```
or this command to check max resources and how many of them are available:
```bash
condor_status -compact -constraint 'GPUs > 0' -af:h Machine State TotalGPUs GPUs TotalCpus Cpus TotalMemory Memory TotalDisk Disk CUDADeviceName
```

### 1.2 Run the venv creation script
Once you are inside the container, run this script:

```bash
bash path/to/create_venv_training.sh
```

**What the script does:**

| Step | Description |
|---|---|
| Load modules | Sources `.modules.sh` → detects BAF, exports `CUDA_HOME`, loads Python via miniforge |
| Create venv | creates a new venv 'venv_ddp' (change it to your preferred name) |
| Install packages | Installs packages required for pre-training. (add your own packages if needed) |
| Verify | Prints all installed package versions |
| Package tarball | Creates `venv_ddp.tar.gz` on CephFS|


The tarball is saved to `${BUDDY}/venv_ddp.tar.gz` (i.e., `/cephfs/user/<user-name>/venv_ddp.tar.gz`).

---

## 2. Run the Training Pipeline

### 2.1 Files you need

The following files are required to run the pre-training pipeline using DDP(Distributed Data Parallel). You can find them in the [utils/optimus_prime](./) and [distributed](../../distributed) directories of this repository.

I will give a short description of each file and its purpose but for detailed information please refer to the documentation [distributed/README.md](../../distributed/README.md) and [polyglot/README.md](../../README.md).


| File | Purpose |
|---|---|
| [llm-foundry/utils/optimus_prime/train_ddp.jdl](./train_ddp.jdl) | HTCondor job description — resources, requirements, container settings |
| [llm-foundry/utils/optimus_prime/train_ddp.sh](./train_ddp.sh) | The bash script that extracts the venv, sets up env vars, and launches the training script |
| [llm-foundry/distributed/train_ddp.py](../../distributed/train_ddp.py) | The python script that contains the training code |
| [llm-foundry/distributed/specifications.yaml](../../distributed/specifications.yaml) | All training hyperparameters (batch size, LR, model config, etc.) and paths for your checkpoints, training data, cache_dir etc |
| [llm-foundry/utils/optimus_prime/config.json](./config.json) | Model architecture (layers, hidden size, attention heads, etc.). The current config.json creates Dense Transformer of ~50M parameters. You can tweak the values to create larger or smaller models. If you want to try another architecture(e.g., MoE, Hybrid), check the section [Example Architecture Configs](../README.md#example-architecture-configs) |
| venv_ddp.tar.gz | The pre-built venv tarball (created in [Section 1](#1-create-the-python-venv)) |

### 2.2 Configure your training

You can configure some values according to your preferences.

- In `specifications.yaml`, you can set the following paths:

_checkpoint_dir, train_dataset_dir, val_dataset_dir, cache_dir, path_to_model_config_
- Be careful about your cache_dir path. If your training run is long and requires multiple jobs submissions and last checkkpoints to continue training, you should put your cache_dir in your CephFS directory. Otherwise, if you put it in your jwd, it will be cleaned up after the job is done and you will lose your cache. (Note: I have not tested with cache_dir in CephFS yet, because I was just doing test runs, so I put the cache_dir in jwd.)

- In `create_venv_training.sh`, you can add the packages you need to install
- In `train_ddp.jdl`, you can set the resources(e.g., GPUs, CPUs, memory) you need for your training job (check the script for details).


### 2.3 Submit the job

Once you are done with the configuration, you can start the training job by submitting the `train_ddp.jdl` file to HTCondor.

```bash
cd ${BUDDY}                          # or wherever your llm-foundry is
condor_submit llm-foundry/utils/optimus_prime/train_ddp.jdl
```

### 2.4 Monitor the job

After submitting the job, to ensure your job is running, you can check the status of your job using the following commands:

```bash
condor_q                            # see all your jobs
# Ideally if your job is running you should see a line like this:
#OWNER    BATCH_NAME    SUBMITTED   DONE   RUN    IDLE  TOTAL JOB_IDS
#<user-id> ID: 100994   6/30 10:14      _      1      _      1 100994.0

condor_q -better-analyze <JOBID>    # If your job is not running or if you want to know detailed information about your job, you can use this command. <JOBID> is the number you get under BATCH_NAME
condor_rm <JOBID>                   # cancel a job
condor_release <JOBID>              # Release a held job
condor_history <your-user-name>     # See your completed jobs
```
For other useful commands, check the [BAF documentation](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/10814637/Helpful+HTCondor+commands).

### 2.5 Check training output

All logs and training outputs are written to your CephFS directory.

- Logs and errors:
    - Path: 
        - `/cephfs/user/<user-name>/run_outputs/ddp-out.<CLUSTER_ID>`,
        - `/cephfs/user/<user-name>/run_outputs/ddp-err.<CLUSTER_ID>`
        - `/cephfs/user/<user-name>/ddp.log.<CLUSTER_ID>/`

- checkpoints:
    - Path: `/cephfs/user/<user-name>/checkpoints/<CLUSTER_ID>/`



### 2.6 Common issues

| Issue | Possible Fix |
|---|---|
| Job stays idle (`condor_q` shows `IDLE`) | Check `condor_q -better-analyze <JOBID>`. Usually the resources you requested in train_ddp.jdl are not available. |
|  OOM Error | Reduce both `total_batch_size` and `micro_batch_size` equally, or reduce model architecture values in `config.json`. |

> **📝 Note:** So far, these are the errors I think you can encounter. Most of the issues were related to establishing a working
> training environment. Several packages (e.g., `flash-attn`) are tightly coupled
> to the cluster's CUDA version, which in turn constrains PyTorch and other
> dependencies. A seemingly simple version pin can cascade into conflicts across
> multiple packages. But with the current create_venv_training.sh and .modules.sh it should work now, unless you are trying to install another package which creates its own dependency.





