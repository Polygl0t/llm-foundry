# JSC Jupiter Support

This folder contains module and installation scripts for the JSC Jupiter booster environment, plus a quick reference on how to run jobs on JUPITER.

## What is included

- [`download.sh`](download.sh) — Simple download script for HuggingFace repos on the JSC JUPITER login node.
- [`jupiter_modules_2026.sh`](jupiter_modules_2026.sh) — a module setup script for the JSC Stages/2026 software stack. This is equivalent to the [`.modules.sh`](../../.modules.sh) script that we use for the other clusters, but it is tailored to the JSC environment.
- [`jupiter_installation_2026.sh`](jupiter_installation_2026.sh) — an installation script that creates a Python virtual environment, installs the project dependencies, and builds or installs CUDA-aware PyTorch and attention extensions for the 2026 stack.
- [`jupiter_installation_trl_2026.sh`](jupiter_installation_trl_2026.sh) — builds an environment for the post-training pipeline (`alignment/sft_trainer.py`, `dpo_trainer.py`, `reward_trainer.py`, `grpo_trainer.py`).
- [`jupiter_installation_eval_harness_2026.sh`](jupiter_installation_eval_harness_2026.sh) — installs the dependencies needed for `lm-evaluation-harness` and pre-caches task datasets/NLTK data/RULER haystack.
- [`ruler_patch.sh`](ruler_patch.sh) — helper sourced by the eval-harness installer to download the RULER haystack file into the shared cache so the `ruler_pt` task can run offline.
- [`run_eval_harness.sh`](run_eval_harness.sh) — SLURM batch script that runs `lm-evaluation-harness` on a JUPITER booster node, evaluating multiple checkpoints in parallel (one per GH200) and post-processing the JSON results into YAML.

> - **Note:** On Jupiter/JSC, we do not have internet access from the compute nodes, so the installation script must be run on a login node.

## Usage

This will install all necessary dependencies and build a distributed training environment for the 2026 stack.

Example:

```bash
bash llm-foundry/docs/jupiter/jupiter_installation_2026.sh
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

## Checking maintenance / system status

To find out whether Jupiter is under maintenance or experiencing an incident, use the **JSC Service Status page**:

**https://status.jsc.fz-juelich.de/**

- Click on **JUPITER** to see its current state, recent events, and **planned events** (e.g. scheduled maintenance).
- **MOTD (Message of the Day):** maintenance announcements and the current system status are shown when you log in over SSH. This is synced from the status page every ~5 minutes.

Maintenance announcements are also listed on the documentation page: <https://apps.fz-juelich.de/jsc/hps/jupiter/maintenance.html>

> - **Note:** Most status-page content is not backed by automated monitoring, so there may be a short delay between an issue appearing and it being listed. Jupiter is also still in an Early-Access phase, so also check the [Build-Up Operation](https://apps.fz-juelich.de/jsc/hps/jupiter/buildup.html) page for known issues.

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

# Torch distributed: rank 0's address. Prefer a NUMERIC IPv4 -- on JUPITER a node answers
# to several names (jpbo-028-33 == r28-nod30 == jpbo-028-33-interconnect-1) and every
# node's /etc/hosts maps its OWN short name to loopback, so a hostname is not guaranteed
# to mean the same address on every node. See "Multi-node accelerate / TRL jobs" below.
MASTER_NODE="$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)"
MASTER_ADDR="$(scontrol show node "$MASTER_NODE" | tr ' ' '\n' | sed -n 's/^NodeAddr=//p' | head -n 1)"
export MASTER_ADDR
# Derive the port from the job id so concurrent jobs cannot collide on 29500.
export MASTER_PORT=$(( 49152 + (SLURM_JOB_ID % 16384) ))

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

## Multi-node accelerate / TRL jobs (DDP, SFT, DPO, GRPO, Reward)

`accelerate launch` across several nodes is the one place on JUPITER where a job needs more than an `#SBATCH` header. accelerate builds a **rendezvous**: one node runs a `TCPStore` server and the other N-1 launchers must reach it and agree on who is who. Four things about this cluster break the obvious way of writing that down, and all four are silent — the job just stalls until a timeout, with no error naming the culprit.

The shape of an accelerate job on booster:

| Setting | Value | Why |
|---|---|---|
| `#SBATCH --nodes=N` | N | one launcher **per node** is what makes the ranks meaningful |
| `#SBATCH --ntasks-per-node=1` | 1 | accelerate forks the 4 per-GPU workers itself |
| `#SBATCH --cpus-per-task=288` | 288 | the single launcher owns the whole node |
| `--gres=gpu:4` | 4 | accelerate's `gpu_ids: all` then sees all four |
| `--machine_rank` | `$SLURM_NODEID` | resolved **per task** (see trap 1) |

Everything the `.ddp_config.yaml` carries for the topology (`num_machines`, `machine_rank`, `main_process_ip`, `main_process_port`) is a single-node default here, so all of it has to be overridden on the `accelerate launch` command line.

### `--machine_rank` must be expanded by each task

A batch script runs **once**, on the first allocated node, where `SLURM_NODEID` is *always* 0. Capture it there and `export` it and every node inherits rank 0, because `srun` propagates the submitting environment verbatim:

```bash
# WRONG -- this script only ever runs on node 0, so MACHINE_RANK is 0 for everybody
MACHINE_RANK="${SLURM_NODEID:-0}"
export GPUS_PER_NODE NUM_MACHINES NUM_PROCESSES MACHINE_RANK
...
--machine_rank $MACHINE_RANK \
```

```bash
# RIGHT -- export only node-INDEPENDENT values, and let each task expand its own rank.
# The backslash keeps `$SLURM_NODEID` literal in $CMD, so the `bash -c "$CMD"` that srun
# runs on every node expands it to that node's rank.
export GPUS_PER_NODE NUM_MACHINES NUM_PROCESSES

export LAUNCHER="accelerate launch \
--config_file $ACCELERATE_CONFIG \
--num_machines $NUM_MACHINES \
--num_processes $NUM_PROCESSES \
--machine_rank \$SLURM_NODEID \
--main_process_ip $MASTER_ADDR \
--main_process_port $MASTER_PORT \
--rdzv_backend static \
--rdzv_conf timeout=300"
```

With `--rdzv_backend static`, `StaticTCPRendezvous` decides who serves the store with `is_master = (self.rank == 0)`. If all N agents claim rank 0, **all N of them build the rendezvous store as the master** (the extra binds do not fail because it passes `multi_tenant=True`, i.e. `SO_REUSEPORT`). Nobody writes `role_info/1..N-1`, every agent's self-dial lands on a random sibling store with a different keyspace, and only rank 0 performs the collective read, so every node waits out the whole timeout and dies together.

The signature to recognise:

```
torch.distributed.DistStoreError: wait timeout after 900000ms, keys:
/none/torchelastic/role_info/0, ..., /none/torchelastic/role_info/N-1
```

You can prove the per-task expansion on a login node without submitting anything:

```bash
bash -c '
SLURM_NODEID=0                                   # what the batch script sees

export LAUNCHER="echo accelerate launch --machine_rank \$SLURM_NODEID"
export CMD="$LAUNCHER trainer.py"
for rank in 0 1 5 15; do SLURM_NODEID=$rank bash -c "$CMD"; done   # what each task prints
'
# -> --machine_rank 0 / 1 / 5 / 15        (with $MACHINE_RANK exported you get 0 four times)
```

### `MASTER_ADDR` should be a numeric IPv4

Names are ambiguous on JUPITER:

- **One node, several names.** `jpbo-028-33` == `r28-nod30` == `jpbo-028-33-interconnect-1`; every agent may pick a different one, so they disagree about which name is "the" master.
- **Bare names resolve to loopback.** Every node's `/etc/hosts` maps its *own* short name to `127.0.0.1` / `::1`, so the usual guard (i.e., "append the domain if `getent hosts $NODE` fails") never fires: the lookup *succeeds*, returning loopback, which is meaningless to the other N-1 nodes.
- **IPv4-mapped IPv6 answers exist,** and c10d tries every `getaddrinfo` candidate. On a compute node (which has no IPv6) each unusable candidate logs `errno: 97 - Address family not supported by protocol`. **These warnings are noise**, i.e., they appear even when the endpoint is already a numeric IP.

So resolve rank 0 to a single numeric IPv4, preferring Slurm's own record of the node (`NodeAddr`), which slurmctld reports identically to every node:

```bash
MASTER_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

numeric_ipv4() {                     # one numeric IPv4, never loopback, never IPv6
    local name="$1"
    [[ -n "$name" ]] || return 1
    if [[ "$name" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        [[ "$name" != 127.* ]] && printf '%s' "$name"
        return 0
    fi
    getent ahostsv4 "$name" 2>/dev/null \
        | awk '$2 == "STREAM" && $1 !~ /^127\./ {print $1; exit}'
}

MASTER_ADDR=""
_slurm_addr="$(scontrol show node "$MASTER_NODE" 2>/dev/null \
    | tr ' ' '\n' | sed -n 's/^NodeAddr=//p' | head -n 1)"
# NodeAddr first; then DNS, trying the FQDN before the bare name (/etc/hosts only
# short-circuits the bare name to loopback).
for cand in "$_slurm_addr" "$MASTER_NODE" \
            ${MASTER_NODE:+$(sed -n 's/^search[[:space:]]*//p' /etc/resolv.conf 2>/dev/null \
                              | tr ' ' '\n' | sed "s|^|${MASTER_NODE}.|")}; do
    MASTER_ADDR="$(numeric_ipv4 "$cand")"
    [[ -n "$MASTER_ADDR" ]] && break
done
[[ -n "$MASTER_ADDR" ]] || MASTER_ADDR="$MASTER_NODE"   # last resort: log a warning

export MASTER_ADDR MASTER_NODE
MASTER_PORT=$(( 49152 + (SLURM_JOB_ID % 16384) ))   # unique per job, in the private range
export MASTER_PORT
```

Rank 0 must be the node whose address you publish: `scontrol show hostnames ... | head -n 1` is the first node of the allocation, i.e. `SLURM_NODEID == 0`. Keep those two in sync.

### The rendezvous timeout defaults to 15 minutes

`--rdzv_conf timeout=300` caps the wait at 5 minutes. Without it a mis-addressed rendezvous eats the entire wall clock silently. This avoids wasting time when the rendezvous is misconfigured.

### Slurm can report a dead job as `COMPLETED`

If the training `srun` fails and the script then runs a couple of `echo`s and a cache cleanup, the script's exit status is whatever the last command returned (i.e., `0`), so `sacct` shows `COMPLETED 0:0` even though every rank died. It is better to capture the step's status and return it:

```bash
srun --nodes="$NUM_MACHINES" --ntasks="$NUM_MACHINES" --cpu-bind=none \
    bash -c "$CMD" 1>>"$out" 2>>"$err"
TRAIN_RC=$?
...
# last line of the script, after the cleanup:
exit "$TRAIN_RC"
```

Then always double-check with `sacct -j <jobid>` and look for the **step** lines. A `FAILED` step under a `COMPLETED` job is the tell:

```
1995708     sft-0.5B  COMPLETED  0:0   00:18:30  16    <- batch script (echoes to success)
1995708.0   bash      FAILED     1:0   00:17:21  16    <- the step that did the work
```

### Pre-flight checks (fail in 15 s instead of 15 min)

Both failure modes above are cheap to test before any GPU time is spent. Put this between building `$CMD` and launching it:

```bash
if [ "$NUM_MACHINES" -gt 1 ]; then
    TRAIN_RC=0

    # (1) Ranks: exactly 0..N-1, each once.
    RANK_MAP="$(srun --nodes="$NUM_MACHINES" --ntasks="$NUM_MACHINES" --cpu-bind=none \
        bash -c 'printf "%s:%s " "${SLURM_NODEID:-unset}" "$(hostname -s)"' 2>>"$err")"
    echo "#   $RANK_MAP" >> "$out"
    DISTINCT_RANKS="$(tr ' ' '\n' <<< "$RANK_MAP" \
        | sed -n 's/^\([0-9][0-9]*\):.*/\1/p' | sort -n -u | wc -l)"
    [ "$DISTINCT_RANKS" -eq "$NUM_MACHINES" ] || TRAIN_RC=1

    # (2) Reachability: rank 0 listens on the rendezvous port, every node dials it.
    python3 "$workdir/rendezvous_preflight.py" --listen "$MASTER_PORT" \
        >> "$out" 2>&1 &
    PREFLIGHT_PID=$!
    sleep 3
    srun --nodes="$NUM_MACHINES" --ntasks="$NUM_MACHINES" --cpu-bind=none \
        python3 "$workdir/rendezvous_preflight.py" \
        --connect "$MASTER_ADDR" "$MASTER_PORT" 1>>"$out" 2>>"$err" || TRAIN_RC=1
    kill "$PREFLIGHT_PID" 2>/dev/null; wait "$PREFLIGHT_PID" 2>/dev/null

    [ "$TRAIN_RC" -eq 0 ] || echo "# Pre-flight FAILED -- not launching." >> "$out"
fi

if [ "$TRAIN_RC" -eq 0 ]; then
    srun --nodes="$NUM_MACHINES" --ntasks="$NUM_MACHINES" --cpu-bind=none \
        bash -c "$CMD" 1>>"$out" 2>>"$err"
    TRAIN_RC=$?
fi
```

The helper is [`rendezvous_preflight.py`](../../tools/rendezvous_preflight.py) (stdlib only, ~15 lines of logic).

---

## Useful SLURM commands

```bash
sinfo                 # show partitions and node states
squeue --me           # show pending/running jobs
squeue --start        # show estimated start time of pending jobs
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
