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

```python
# submit_job.py — write locally, scp, run remotely
import subprocess, shlex
from pathlib import Path

SIF    = "/home/yannik.qu/master-thesis/experiments/class_imbalance/environment.sif"
PYPATH = "/home/yannik.qu/master-thesis/experiments/shared:/home/yannik.qu/master-thesis/experiments/design_dataset/code"
RESULTS = "/home/yannik.qu/master-thesis/experiments/design_dataset/results/full_scale"

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
- Use Apptainer `.sif` containers, not conda/venv trees on `/home`.
- Use SquashFS for datasets with many files; check shared squashed datasets first.
- `/home/space/` datasets are read-only unless you created the specific files.
- Stage `.sqfs` images to job-local `/tmp` for repeated training reads.
- Never cancel others' jobs; never modify shared datasets you didn't create.
