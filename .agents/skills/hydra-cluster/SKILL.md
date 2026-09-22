---
name: hydra-cluster
description: Use when working with the TU Berlin Hydra cluster from this thesis repo, including SLURM sbatch or srun jobs, Hydra job scripts, Apptainer containers, SquashFS datasets, cluster dataset paths, shared storage, GPU or CPU partitions, and Jupyter-on-Hydra workflows.
---

# Hydra Cluster

Use this skill for TU Berlin Hydra cluster work. Hydra here means the ML-group HPC cluster, not Python Hydra or hydra-zen.

## First Steps

1. Read `CLUSTER.md` (repo root) — authoritative source for SSH, partitions, GPU constraints, storage paths, and dataset locations.
2. Know where you are: local/WSL → edit and inspect; login node → prep, queue, `sbatch` only; compute node → heavy work, container builds, SquashFS.
3. If behavior conflicts with `CLUSTER.md`, treat upstream Hydra docs as truth; update `CLUSTER.md` if asked.

## Remote Commands From Local

**Always use a login shell** — `squeue`, `sacct`, `sbatch`, and any Hydra-env command are not on `PATH` in a plain SSH session:

```bash
ssh hydra 'bash -lc "squeue -u $USER"'
ssh hydra 'bash -lc "sacct -j <id> --format=JobID,State,ExitCode,Elapsed"'
```

From **PowerShell**: single-quote the remote command; avoid unescaped `$(...)` or `2>/dev/null` (PowerShell evaluates them locally).

Paths with spaces must be double-quoted in every shell layer.

## Writing And Submitting Jobs

**Never write job scripts via heredoc through SSH** — quoting breaks across local shell → SSH → remote shell. Instead:

1. Write a Python submission script locally.
2. `scp` it to `/tmp/` on the cluster.
3. Run with `ssh hydra 'bash -lc "python3 /tmp/script.py"'`.

**`/tmp` is node-local.** The login-node driver is fine in `/tmp`, but anything the **job**
reads (sbatch scripts, worker `.py`, modules) must live on shared `/home` — the compute node
has its own empty `/tmp`.

```python
# submit_job.py — write locally, scp, run remotely
import subprocess, shlex
from pathlib import Path

SIF = "/home/yannik.qu/master-thesis/experiments/environment.sif"
PYPATH = "/home/yannik.qu/master-thesis/experiments/01_benchmark/02_benchmark_patch/code"
RESULTS = "/home/yannik.qu/master-thesis/experiments/01_benchmark/02_benchmark_patch/outputs"

# Verify paths before submitting — a missing path gives an immediate failure
for p in [SIF, RESULTS]:
    assert Path(p).exists(), f"missing: {p}"

cmd = ["/usr/bin/apptainer", "run", "-B", "/home/space:/home/space:rw",
       SIF, "python3", "-m", "mymodule", f"--results-dir={RESULTS}"]

script = "\n".join([
    "#!/bin/bash",
    "#SBATCH --job-name=myjob",
    "#SBATCH --partition=cpu-2h",
    "#SBATCH --cpus-per-task=4",
    "#SBATCH --output=logs/myjob-%j.out",  # %j prevents overwriting between tasks
    f"export APPTAINERENV_PYTHONPATH={shlex.quote(PYPATH)}",
    " ".join(shlex.quote(p) for p in cmd),
])

jid = subprocess.run(["sbatch", "--parsable"], input=script,
                     text=True, capture_output=True, check=True).stdout.strip()
print("submitted:", jid)
```

**Float formatting trap**: `f"{1.0:g}"` → `"1"`, but directory names are often `parameter=1.0`. Use plain `f"{value}"` and verify the constructed path exists before submitting.

### Dependency Chaining

```python
jid1 = submit(job1)
jid2 = submit(job2)
# afterok: all upstreams must succeed; afterany: run regardless
sbatch --dependency=afterok:{jid1}:{jid2} downstream.sh
```

In Python: `f"afterok:{':'.join(ids)}"` passed as `--dependency=...`.

### Apptainer

`apptainer` is **only on compute nodes** — the login node gives `command not found`. Use `srun --partition=cpu-test --pty bash` for an interactive shell, or submit via `sbatch`.

Containers have no SLURM client — never call `sbatch`/`srun` from inside one, and don't bind `/opt/slurm`+`/etc/slurm`+`/run/munge` in to work around that (dead end: container `/etc/passwd` lacks the `slurm` user, auth fails). A project's `submit` CLI is login-node/orchestrator work; only the jobs it schedules run inside the container. If it needs numpy/pandas/etc. itself, use the project venv (`~/master-thesis/.venv`) on the login node, not system `python3`:

```bash
ssh hydra 'bash -lc "cd ~/master-thesis/experiments/<group>/<name> && ~/master-thesis/.venv/bin/python3 code --config configs/<x>.yaml submit"'
```

Pass Python paths into the container via `APPTAINERENV_PYTHONPATH` (Apptainer strips the prefix and sets `PYTHONPATH` inside):

```bash
export APPTAINERENV_PYTHONPATH="/path/to/shared:/path/to/code"
apptainer run -B /home/space:/home/space:rw environment.sif python3 -m mymodule
```

## Inspection

```bash
squeue -u "$USER"                                    # running / pending
sacct -j <id> --format=JobID,State,ExitCode,Elapsed  # completed job outcome
seff <id>                                            # efficiency summary
scontrol show jobid -dd <id>                         # full job details
```

### Waiting For Jobs

Use `scripts/hydra_wait.sh` (in this skill) instead of `sleep` loops or short `ScheduleWakeup` polls. Run it with `run_in_background`; the task notification fires when the jobs are done:

```bash
ssh hydra 'bash -l -s -- <fit_array_id> <analyze_id>' < .claude/skills/hydra-cluster/scripts/hydra_wait.sh
```

It polls every 30 s. It cancels jobs stuck on `DependencyNeverSatisfied`, prints `sacct` per task, and prints the stderr tail of up to 3 failed tasks. Exit 0 means every task `COMPLETED`. For runs longer than a few hours, add one long `ScheduleWakeup` (1200 s or more) as a fallback.

### Known Failure Modes

- **`DependencyNeverSatisfied` / "dep never satisfied"**: an `afterok` upstream failed. The downstream pends forever. Read the upstream `.err` logs, `scancel` the downstream, fix, then resubmit both.
- **Analyze crashed after fits succeeded**: resubmit only the analyze stage. Fit shards skip records that already exist, so a full resubmit only wastes queue time.
- **Manual `srun` smoke test**: `mkdir -p` the output and log dirs first (the sbatch template does this, a manual `srun` does not). `cpu-test` caps runtime at 15 min. A smoke test cut off by the cap without an error has passed the path and import stage. It is not a bug.
- **Stray modules in login-node `/tmp`**: other users' files such as `/tmp/bisect.py` shadow the stdlib (`cannot import name 'bisect' from partially initialized module`). Run driver scripts with `python3 -P` from `~`, not with `cd /tmp`.
- **Imports resolve to the wrong package**: stale `code/` dirs that hold only `__pycache__` can shadow shared packages through `_bootstrap.py`. Delete dirs with no `.py` files.
- **Results files**: experiment outputs land in `<experiment>/outputs/<dataset>/patch/` on Hydra. `ls` that path before `scp`. Do not guess `~/<name>.json`.

### Git Sync

`git pull --ff-only` fails if tracked files are modified on the cluster (e.g. a script overwrote a committed output). Inspect the diff, then discard:

```bash
git diff HEAD -- path/to/file   # inspect first
git checkout -- path/to/file
git pull --ff-only
```

## Storage And Data

Full details in `CLUSTER.md`. Key rules:

- `/home` is BeeGFS — avoid many small files.
- Keep `~` clean: use `/tmp` for transient submission scripts and remove stale files when the job no longer needs them.
- Use Apptainer `.sif` containers, not conda/venv trees on `/home`.
- Use SquashFS for datasets with many files; check shared squashed datasets first.
- `/home/space/` datasets are read-only unless you created the specific files.
- Stage `.sqfs` images to job-local `/tmp` for repeated training reads.
- Never cancel others' jobs; never modify shared datasets you didn't create.
