---
name: hydra-cluster
description: Use when working with the TU Berlin Hydra cluster from this thesis repo, including SLURM sbatch or srun jobs, Hydra job scripts, Apptainer containers, SquashFS datasets, cluster dataset paths, shared storage, GPU or CPU partitions, and Jupyter-on-Hydra workflows.
---

# Hydra Cluster

Use this skill for TU Berlin Hydra cluster work. Hydra here means the ML-group HPC cluster, not Python Hydra or hydra-zen.

## First Steps

1. Read repo-root `CLUSTER.md` before using Hydra. It is the detailed source for SSH access, partition tables, GPU constraints, storage paths, dataset locations, examples, and upstream documentation provenance.
2. Confirm where the shell is before running commands:
   - local machine or WSL: safe for editing job scripts and local inspection
   - Hydra login/main node: use only for file inspection, light preparation, queue inspection, and `sbatch`
   - SLURM allocation or compute node: required for compute-heavy work, container builds, and SquashFS creation
3. If current cluster behavior conflicts with `CLUSTER.md`, treat upstream Hydra documentation as the source of truth and update `CLUSTER.md` separately if asked.

## Job Workflow

- Prefer `sbatch` jobs. Use interactive `srun` only when necessary, keep sessions short, and avoid long interactive allocations.
- Validate scripts, imports, containers, data paths, and CUDA availability on `cpu-test` or `gpu-test` before using longer partitions.
- Choose the shortest partition that can realistically finish the job. Shorter runtime partitions have higher priority; long partitions have fewer running slots.
- Use job-id-specific log paths such as `logs/job-%j.out` so parallel jobs do not overwrite each other.
- Use array jobs for independent sweeps or repeated independent runs.
- For CUDA jobs, request a GPU partition and GPU resources, and run containers with `apptainer run --nv`.
- Add `--mail-type=BEGIN,FAIL,END --mail-user=<address>` only when the user asks for email notifications.
- Never cancel, hold, release, or requeue jobs that are not clearly the user's jobs unless explicitly asked.

## Inspection And Troubleshooting

### Remote SSH from local or WSL

When checking the queue from a **local machine** (PowerShell, WSL, or macOS), run SLURM through a **login shell**. Plain `ssh hydra "squeue ..."` often uses a non-login shell where `squeue`, `sacct`, and `scontrol` are not on `PATH`, which produces a misleading `command not found`. SLURM is installed on Hydra; the session environment is wrong, not the cluster.

```bash
ssh hydra 'bash -lc "squeue -u $USER"'
ssh hydra 'bash -lc "sacct -j <job-id> --format=JobID,JobName,Elapsed -n | tail -30"'
```

From **PowerShell**, wrap the remote command in single quotes and avoid unescaped `$(...)` or `2>/dev/null` in the `ssh` argument (PowerShell parses those locally).

### SLURM commands on Hydra

Useful Hydra-side SLURM commands:

```bash
sinfo
sinfo -NEL
squeue -u "$USER"
squeue -u "$USER" -t RUNNING
squeue -u "$USER" -t PENDING
scontrol show jobid -dd <job-id>
sstat --format=AveCPU,AvePages,AveRSS,AveVMSize,JobID -j <job-id> --allsteps
sacct -j <job-id> --format=JobID,JobName,MaxRSS,Elapsed
seff <job-id>
sprio -j <job-id>
```

Use `cwho <username>` only as an optional Hydra-side lookup when identifying a username is relevant.

## Storage, Data, And Environments

- Do not run heavy compute, container builds, or SquashFS creation on the login node.
- Avoid many small files on `/home`; shared filesystem lookups are expensive.
- Use Apptainer `.sif` containers instead of large conda or venv trees on `/home`.
- Containers are immutable. Rebuild for stable dependency changes; use writable overlays only as an exceptional temporary layer.
- For architecture-sensitive containers, build on the intended CPU architecture and constrain jobs to the same architecture.
- Use SquashFS images for datasets with many files. Check shared squashed datasets before creating new ones.
- Treat shared dataset directories as read-only unless the user created the exact files being modified.
- For repeated training reads, copy the `.sqfs` image to job-local `/tmp` and bind it into the container from there.
- Use `/temp` only for short-lived persistent local storage and refresh it deliberately when needed.
- Use `/archive/YEAR/PROJECT` only for essential project data with a sibling metadata JSON file.

## Jupyter On Hydra

- Run Jupyter inside a SLURM job, not on the login node.
- Ensure the container includes Jupyter or Notebook dependencies.
- Start Jupyter with `--ip 0.0.0.0 --no-browser` inside the job.
- Identify the assigned compute head from the job or log output, then tunnel from the local machine through Hydra to that head.
- Use the token URL from the job log after the tunnel is active.
