
# BAF Support

## Table of Contents

| Section                                                                               | Description                                        |
|---------------------------------------------------------------------------------------|----------------------------------------------------|
| [Accessing the BAF Cluster](#accessing-the-baf-cluster-aka-the-optimus-prime-cluster) | SSH setup, key pair, and login                     |
| [General Documentation](#general-documentation)                                       | Official docs, storage layout, quotas              |
| [Running Jobs on BAF](#running-jobs-on-baf)                                           | Create venv, submit training jobs, monitor         |
| [Working with Datasets](#working-with-datasets)                                       | Downloading and preparing datasets on BAF          |
| [Common Issues](#common-issues)                                                       | Troubleshooting idle jobs, OOM, and other problems |

## Accessing the BAF Cluster (aka, the Optimus Prime Cluster)

To work on BAF, you first need to be granted access. Contact your local administrator to get access. Once you have been added (through your UniID), you can log in using the following command:

```bash
ssh <-USERID->@desktop.physik.uni-bonn.de
```

You will be prompted for your UniID password.

By default, you must enter your password every time you connect. To avoid this, set up an SSH key pair by following the steps below.

### 1. Create the SSH directory

If `~/.ssh` does not already exist, create it:

```bash
mkdir -p ~/.ssh
```

### 2. Generate an SSH key pair

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_baf
```

> **Note:** You will be prompted for a passphrase. Choose something secure and memorable. To skip the passphrase, press "Enter" when prompted.

### 3. Configure the SSH alias

Open (or create) `~/.ssh/config` and add the following entry, replacing `<-USERID->` with your UniID:

```text
Host baf
  HostName desktop.physik.uni-bonn.de
  User <-USERID->
  IdentityFile ~/.ssh/id_ed25519_baf
  IdentitiesOnly yes
```

### 4. Copy your public key to the cluster

```bash
ssh-copy-id -i ~/.ssh/id_ed25519_baf.pub <-USERID->@desktop.physik.uni-bonn.de
```

Enter your UniID password when prompted — this is the last time you will need it.

### 5. Connect

You can now log in without having to enter your UniID password every time.


```bash
ssh baf
```

## General Documentation

Once you get access to the BAF cluster, **read the official documentation first**:

- 🔗 [BAF (Confluence)](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/10814258/BAF)

Some information there may be outdated, but it gives you the essential background on how the cluster works (HTCondor, containers, CephFS, etc).

Every user by default is equipped with a personal data storage directory at

```
/cephfs/user/<-USERID->/
```

Within the Job-containers, this directory can be easily accessed via the environment variable `$BUDDY`. There is a quota on the **number of files (100,000) and the available space (500 GB)** for your BUDDY directory.

> - **Note:** Do NOT submit from /cephfs. Run `condor_submit` from your home directory. You should always submit jobs from your home directory, and use the BUDDY directory for storing data, checkpoints, logs, etc. See [the docs](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/10814633/HTCondor+on+BAF#HTCondoronBAF-SubmittingaClusterJob%2FJobArray) for more information.

## Running Jobs on BAF

Following, in the next sections, we provide a step-by-step guide to run the distributed training pipeline from the foundry on BAF. The steps are summarized in the table below:

> - **Note:** The training pipeline is used here as an example to demonstrate how to run jobs on BAF. You can adapt the same approach to run other jobs (e.g., inference, evaluation, etc.) on the cluster.

### 1. Create the Python venv

All jobs submitted to the cluster are executed inside containers providing the desired runtime environment for each job. Inside the container, you will be inside a **Job Working Directory (jwd)**, which is cleaned up after the job. Therefore, you need to move your venv to a persistent location (CephFS) so that it can be reused by subsequent jobs.

This is a one-time step. You build a venv with all required packages on the container, compress it into a tarball and put it on CephFS, then every subsequent training job extracts it to `/jwd` at startup. (All of these steps are already handled by the scripts in this repo.)

### 1.1 Start an interactive GPU job

For some packages (e.g., `flash-attn`), they require information about the GPU and CUDA version during installation. Therefore, we recommend starting an interactive GPU job to create the venv. Since we are only going to install packages, we don't need a lot of resources. You can adjust the resources as needed.

> - **Note:** Remember to start jobs from your home directory, not from `/cephfs`. You can use the `$BUDDY` environment variable to access your CephFS directory. However, you cannot access your home directory from inside the container. Hence, you should also copy (`cp -r ~/llm-foundry $BUDDY/llm-foundry`) the `llm-foundry`, or any other thing you need to run in the container, to your CephFS directory (`$BUDDY`). **Symlinks will not work**.

```bash
condor_submit -interactive \
    -append '+ContainerOS = "Rocky9"' \
    -append '+CephFS_IO = "medium"' \
    -append '+MaxRuntimeHours=2' \
    -append 'Request_gpus = 1' \
    -append 'requirements = (CUDADeviceName == "NVIDIA H200")' \
    -append 'Request_cpus = 24' \
    -append 'Request_memory = 32000 MB'
```

To know more about the available `ContainerOS`, `CephFS_IO`, `MaxRuntimeHours` options, go to the [BAF documentation](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/10814633/HTCondor+on+BAF). Or see our [example job description file](./job.jdl) for more details.

Also, before deciding on the values for `Request_cpus`, `Request_gpus`, `requirements` and `Request_memory`,  make sure they are available. You can run this command to check the available resources on the cluster:

```bash
condor_status -compact -constraint 'GPUs > 0' -af:h Machine State GPUs Cpus Memory Disk CUDADeviceName
```

or this command to check max resources and how many of them are available:

```bash
condor_status -compact -constraint 'GPUs > 0' -af:h Machine State TotalGPUs GPUs TotalCpus Cpus TotalMemory Memory TotalDisk Disk CUDADeviceName
```

### 1.2 Run the venv creation script

The [`create_venv.sh`](./create_venv.sh) script serve as an example of how to create a python virtual environment (venv) for running the distributed training pipeline implemented in the foundry. You can modify it to create a venv for your own use case.

Once you are inside the container, run this script:

```bash
bash $BUDDY/create_venv.sh
```

**What the script does:**

| Step             | Description                                                                            |
|------------------|----------------------------------------------------------------------------------------|
| Load modules     | Sources `.modules.sh`. It detects BAF, exports `CUDA_HOME`, loads Python via miniforge |
| Create venv      | creates a new virtual environment (`.venv`)                                            |
| Install packages | Installs packages required for the job.                                                |
| Verify           | Prints all installed package versions                                                  |
| Package tarball  | Creates `.venv.tar.gz` on CephFS                                                       |

The tarball is saved to `${BUDDY}/.venv.tar.gz` (i.e., `/cephfs/user/<-USERID->/.venv.tar.gz`). You can change the name of the tarball in the script if you want. This tarball will be used by all subsequent training jobs to extract the venv into their jwd.

## 2. Run the Training Pipeline

### 2.1 Files you need

Here we show how to run the distributed training pipeline on BAF. The training pipeline is used here as an example to demonstrate how to run jobs. You can adapt the same approach to run other jobs (e.g., inference, evaluation, etc.) on the cluster. To learn more about the training pipeline, please refer to the [distributed/README.md](../../distributed/README.md).

The only thing that is different from running the training pipeline on other SLURM managed clusters is that on BAF, you need to submit the job via HTCondor instead of SLURM. To submit the job, you need to a HTCondor job description file (`job.jdl`) and the executable bash script (e.g., `train_ddp.sh`) that extracts the venv, sets up env vars, and launches the training script. These files are already provided in this repo.

We provide example files in:

- [`utils/baf/job.jdl`](./job.jdl) — HTCondor job description file. You can modify it to change the resources you need for your training job (e.g., GPUs, CPUs, memory). Check the script for details.
- [`utils/baf/train_ddp.sh`](./train_ddp.sh) — bash script that extracts the venv, sets up env vars, and launches the training script. You can modify it to change the packages you need to install in the venv.

> - **Note:** Remember to submit jobs from your home directory, not from `/cephfs`. For optimal performance, you should lunch the jdl from your home directory, and use the BUDDY directory for storing the executables, data, checkpoints, logs, etc. See [the docs](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/10814633/HTCondor+on+BAF#HTCondoronBAF-SubmittingaClusterJob%2FJobArray) for more information.
>
> - **Note:** Due to some default configurations in the BAF cluster, you cannot directly clone repositories from GitHub inside `$BUDDY`. Therefore, you should clone the repository in your home directory (i.e., `/home/<-USERID->/physik/llm-foundry`) and copy it to `$BUDDY`. When you already have the repository in your `$BUDDY`, you go into the said repository, and set this configuration: `git config fetch.unpackLimit 10000`. This will allow you to do regular `git fetch` and `git pull` commands straight from your `$BUDDY` directory.


### 2.2 Configure your training

To configure the training, you need to follow the instructions in the [distributed/README.md](../../distributed/README.md) to set up your training configuration. In short, you will need to set up the following files:

- `config.json` — defines the model architecture (see "[Example Architecture Configs](../../distributed/README.md#example-architecture-configs)").
- `specifications.yaml` — defines the training hyperparameters (see [`distributed/specifications.yaml`](../../distributed/specifications.yaml) for an example).

>  **Note:** Be careful about your `cache_dir` path. If your training run is long and requires multiple jobs submissions, it is wise to put your `cache_dir` in your CephFS directory. Otherwise, if you put it in your jwd, it will be cleaned up after the job is done and you will lose your cache.

### 2.3 Submit the job

Once you are done with the configuration, you can start the training job by submitting the `job.jdl` file to HTCondor.

```bash
condor_submit ~/scripts/job.jdl # or whatever path you put your job.jdl
```

### 2.4 Monitor the job

After submitting the job, to ensure your job is running, you can check the status of your job using the following commands:

```bash
condor_q # see all your jobs
# Ideally if your job is running you should see a line like this:
#>> OWNER    BATCH_NAME    SUBMITTED   DONE   RUN    IDLE  TOTAL JOB_IDS
#>> <user-id> ID: 100994   6/30 10:14      _      1      _      1 100994.0

condor_q -better-analyze <JOBID>    # If your job is not running or if you want to know detailed information about your job, you can use this command. <JOBID> is the number you get under BATCH_NAME
condor_rm <JOBID>                   # cancel a job
condor_release <JOBID>              # Release a held job
condor_history <-USERID->            # See your completed jobs
```

For other useful commands, check the [BAF documentation](https://confluence.team.uni-bonn.de/spaces/PHYIT/pages/10814637/Helpful+HTCondor+commands).

## Working with Datasets

### 1. Downloading

According to the BAF IT-support team, large dataset downloads should not be submitted as cluster jobs. Instead, download directly from a login node (desktop12.physik.uni-bonn.de) and write the output to your $BUDDY directory (`/cephfs/user/<-USERID->/`).
No condor_submit, just run the download right from the login node.

Why:
- Downloading from inside a cluster job goes through the shared cluster gateway, which limits the number of concurrent outbound connections. This can makes large downloads slow or stuck.
- Downloading to your home directory (`/home/<-USERID->/`) puts heavy load on the shared home file system, which impacts other users and can easily be avoided by writing directly to CephFS instead.

```bash
# On the login node
# If you need a small venv for downloading for example if downloading from HuggingFace, we would need libraries like huggingface_hub
module load miniforge/24.7.1-0-py312
source venv  # or your venv
```

Run in `screen` to survive disconnects:

```bash
screen -S download

# Example: downloading a dataset from HuggingFace. You can change the repo_name, output_dir (point to $BUDDY), cache_dir (point to $BUDDY), hf_token, repo_type and allow_patterns as needed.
python3 llm-foundry/utils/download.py \
    --repo_name "Polygl0t/gigalekh-v1" \
    --output_dir "$BUDDY/gigalekh" \
    --cache_dir "$BUDDY/.cache" \
    --token "$HF_TOKEN" \
    --repo_type "dataset" \
    --allow_patterns "default/*.parquet"
```

## Common Issues


### Job stays idle (`condor_q` shows `IDLE`)

Check `condor_q -better-analyze <JOBID>`. Usually the resources you requested in job.jdl are not available.

### OOM Error

Reduce both `total_batch_size` and `micro_batch_size` equally, or reduce model architecture values in `config.json`. Also, increasing the `request_memory` in your jdl script may help.
