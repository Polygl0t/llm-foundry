# JSC Jupiter Support

This folder contains module and installation scripts for the JSC Jupiter booster environment,
plus a quick reference on how to run jobs on JUPITER.

## What is included

- [`download.sh`](download.sh) — Simple download script for HuggingFace repos on the JSC JUPITER login node.
- [`jupiter_modules_2026.sh`](jupiter_modules_2026.sh) — a module setup script for the JSC Stages/2026 software stack. This is equivalent to the [`.modules.sh`](../.modules.sh) script that we use for the other clusters, but it is tailored to the JSC environment.
- [`jupiter_installation_2026.sh`](jupiter_installation_2026.sh) — an installation script that creates a Python virtual environment, installs the project dependencies, and builds or installs CUDA-aware PyTorch and attention extensions for the 2026 stack.

> - **Note:** On Jupiter/JSC, we do not have internet access from the compute nodes, so the installation script must be run on a login node.

## Usage

This will install all necessary dependencies and build a distributed training environment for the 2026 stack.

Example:

```bash
bash llm-foundry/utils/jupiter/jupiter_installation_2026.sh
```

---

# Running jobs on JUPITER (quick reference)

JUPITER is Europe's first exascale supercomputer, operated by JSC at Forschungszentrum Jülich.

> Official documentation: <https://apps.fz-juelich.de/jsc/hps/jupiter/index.html>

## Login vs. compute nodes

|            | Login nodes                  | Booster (compute) nodes                          |
|------------|------------------------------|--------------------------------------------------|
| Hostnames  | `jpbl-*` (12 nodes)          | `jpbo-*` (5,884 nodes)                           |
| CPUs       | 72 × Arm Grace (Neoverse-V2) | 288 × Arm Grace (Neoverse-V2) = 4 × 72 cores     |
| GPUs       | 1 × NVIDIA H100 (96 GB HBM3) | 4 × NVIDIA H100 (96 GB HBM3 each = 384 GB total) |
| CPU memory | 480 GB LPDDR5X               | 480 GiB total (4 × 120 GB LPDDR5X)               |
| Internet   | ✅ available                 | ❌ no internet access                            |


- Each booster node contains **4 NVIDIA GH200 Grace-Hopper superchips**. Each superchip is one 72-core Grace CPU + one H100 GPU, linked by NVLink-C2C (900 GB/s).
- **Install software, download models/datasets, and stage data on the login nodes.** Compute nodes are offline, so jobs that need internet will fail.

## Partitions and time limits

Currently (the Early Access phase) there is **only one partition**:

| Partition | Node type                  | Max wall time | Default wall time | Min nodes | Max nodes    |
|-----------|----------------------------|---------------|-------------------|-----------|--------------|
| `booster` | `mem480` (480 GiB CPU mem) | **12 h**      | 1 h               | 1         | whole system |

- `booster` is the **default** partition, so `--partition=booster` is technically optional, but it is good practice to write it explicitly.
- For now, *all* jobs — including CPU-only jobs — run on the `booster` partition.

## Important rules

- **Nodes are exclusive.** There is no node sharing: the smallest allocation unit is one full node, and you are billed per **node × wall time**. `--exclusive` is implicit.
- **Use `srun`, not `mpiexec`.** `mpiexec` is not supported on JUPITER.
- **`--gres=gpu:4` is applied automatically** if you do not set `gres`. You may request fewer GPUs (`--gres=gpu:1` … `gpu:4`) for testing, but production jobs are expected to use all 4 GPUs per node. You are charged for the full node regardless of the number of GPUs.
- **GPU visibility:** Slurm assigns one GPU per task by setting `CUDA_VISIBLE_DEVICES`. For a **single-task** job, only one GPU is visible by default — export all four manually: `export CUDA_VISIBLE_DEVICES=0,1,2,3`.
- **CPU power budget:** each superchip's CPU is capped at 100 W by default to favour the GPU.

---

## SLURM templates

> Replace `--account=polyglot` with your own SLURM account / budget.

### 1. CPU-only job (single node)

There is no separate CPU partition yet, so CPU-only work also runs on `booster` and occupies a full node. Use all 288 cores with a single task, or fewer cores if your workload is I/O-bound.

```bash
#!/bin/bash -l
#SBATCH --account=polyglot
#SBATCH --partition=booster
#SBATCH --job-name=my-cpu-job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288        # full node; use fewer (e.g. 96) for I/O-bound work
#SBATCH --time=08:00:00            # max 12:00:00 on booster

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --cpu-bind=none python3 my_script.py
```

### 2. Single-node GPU job (4 GPUs)

```bash
#!/bin/bash -l
#SBATCH --account=polyglot
#SBATCH --partition=booster
#SBATCH --job-name=my-gpu-job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4        # one task per GPU
#SBATCH --cpus-per-task=72         # 288 cores / 4 GPUs
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

# 4 tasks -> Slurm already gives each task one GPU.
srun --cpu-bind=none python3 my_script.py
```

If your program is a single process that needs all 4 GPUs (typical for PyTorch data-parallel):

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4

export CUDA_VISIBLE_DEVICES=0,1,2,3   # single task -> export all GPUs manually
srun --cpu-bind=none python3 my_script.py
```

### 3. Multi-node GPU job (distributed training)

Scale with `--nodes=N` and one task per GPU (4 per node). `--gres=gpu:4` applies per node.

```bash
#!/bin/bash -l
#SBATCH --account=polyglot
#SBATCH --partition=booster
#SBATCH --job-name=my-ddp-job
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

# Torch distributed: derive MASTER_ADDR from the first allocated node.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
export MASTER_ADDR
export MASTER_PORT=29500

srun --cpu-bind=none python3 train.py ...
```

Tips for multi-node jobs:

- `--ntasks-per-node=4` + `--cpus-per-task=72` gives good CPU-GPU affinity (72 cores per superchip).
- For network-sensitive jobs you can constrain placement with `--switches=<count>@<max-wait>` (16 nodes per leaf switch) or with node features via `--constraint`, e.g. `--constraint=rack001` or `--constraint=dfpg01`.

### 4. Interactive session

```bash
salloc --account=polyglot --partition=booster --nodes=2 --time=00:30:00

# then, inside the allocation:
srun --cpu-bind=none --nodes=2 --ntasks-per-node=4 --cpus-per-task=72 --gres=gpu:4 \
     --pty /bin/bash -i
```

Remember: the allocation is billed whether or not you use it, so prefer batch jobs.

---

## Useful SLURM commands

```bash
sinfo                 # show partitions and node states
squeue                # show pending/running jobs
sbatch job.sh         # submit a batch job
salloc ...            # request an interactive allocation
srun ...              # launch a job step inside an allocation
scancel <jobid>       # cancel a job
sacct -j <jobid>      # accounting info for a job
scontrol show job <jobid>   # detailed job info
```

## Notes

- You can learn more about Jupiter in <https://apps.fz-juelich.de/jsc/hps/jupiter/index.html>.
- Because JUPITER is in Early Access, always double-check the official pages for [partitions/batch system](https://apps.fz-juelich.de/jsc/hps/jupiter/batchsystem.html) and [GPU computing](https://apps.fz-juelich.de/jsc/hps/jupiter/gpu-computing.html), as limits may change.
