# Hydra Cluster

This document summarizes the TU Berlin Hydra cluster documentation for agents working from this repository.
Source documentation: https://git.tu-berlin.de/ml-group/hydra/documentation at commit `bbef4c93688269287b9a1007aab0b50ec637eb87`.

Hydra is the HPC cluster of the ML, MLSEC, UNIML, and Cognition research groups at TU Berlin. It uses SLURM and has separate login/main and compute nodes. The Hydra cluster is unrelated to Python `hydra` or `hydra-zen`.

## Access

Use the SSH shortcut from WSL or PowerShell:

```bash
ssh hydra
```

Raw fallback:

```bash
ssh yannik.qu@hydra.ml.tu-berlin.de
```

The shortcut should resolve to:

```sshconfig
Host hydra
  HostName hydra.ml.tu-berlin.de
  User yannik.qu
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  ForwardAgent yes
  ServerAliveInterval 60
```

PowerShell uses the same key through `C:/Users/Yannik/.ssh/id_ed25519`.

Do not run compute-heavy work on the main/login node. Use it to prepare files, inspect state, and submit SLURM jobs.

## SLURM Basics

Prefer batch jobs with `sbatch`. Use interactive jobs only when necessary, and keep them short.

Common commands:

```bash
sinfo
squeue -u "$USER"
sbatch job.sh
scancel <job-id>
seff <job-id>
scontrol show jobid -dd <job-id>
sacct -j <job-id> --format=JobID,JobName,MaxRSS,Elapsed
```

Short interactive sessions:

```bash
srun --partition=cpu-2h --pty bash
srun --partition=gpu-2h --gpus=1 --pty bash
```

Minimal CPU job:

```bash
#!/bin/bash
#SBATCH --job-name=my_cpu_job
#SBATCH --partition=cpu-test
#SBATCH --gpus-per-node=0
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

apptainer run /opt/apps/pytorch-2.0.1-gpu.sif python script.py
```

Minimal GPU job:

```bash
#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

apptainer run --nv /opt/apps/pytorch-2.0.1-gpu.sif python script.py
```

Use `logs/job-%j.out` or another job-id-based path so parallel jobs do not overwrite each other.

## Partitions And GPUs

Choose the shortest runtime that can finish the job. Shorter partitions have higher scheduling priority, and long partitions have fewer running slots per account. Validate scripts and containers on the test partitions before submitting long jobs.

Partitions:

| Name | Kind | Runtime | Running jobs per account |
| --- | --- | ---: | ---: |
| `cpu-test` | CPU | 15m | 1 |
| `gpu-test` | GPU | 15m | 1 |
| `cpu-9m` | CPU | 9m | 200 |
| `cpu-2h` | CPU | 2h | 100 |
| `cpu-5h` | CPU | 5h | 80 |
| `cpu-2d` | CPU | 2d | 50 |
| `cpu-7d` | CPU | 7d | 30 |
| `gpu-9m` | GPU | 9m | 100 |
| `gpu-2h` | GPU | 2h | 50 |
| `gpu-5h` | GPU | 5h | 35 |
| `gpu-2d` | GPU | 2d | 28 |
| `gpu-7d` | GPU | 7d | 3 |

GPU constraints from the cluster docs:

| Constraint | Hardware |
| --- | --- |
| `mig40` | A100 80GB partitioned with MIG, 16 virtual GPUs |
| `80gb` | A100 80GB |
| `40gb` | A100 40GB |
| `h100` | H100 80GB |
| `h200` | H200 |
| `blackwell` | RTX PRO 6000 Blackwell 97GB |
| `3090` | RTX 3090 24GB |
| `6000` | Quadro RTX 6000 24GB |

Example GPU constraint:

```bash
sbatch --partition=gpu-2h --gpus=1 --constraint="80gb|40gb" job.sh
```

For CPU partitions over 2h, Hydra uses gang scheduling, so jobs can be suspended and resumed in slices.

## Files And Storage

`/home` is shared BeeGFS across heads. Avoid many small files because each lookup creates network filesystem traffic. Prefer single large image files for environments and datasets.

Use these locations deliberately:

| Path | Use |
| --- | --- |
| `/home/<user>` | Code, job scripts, logs, containers, small config files |
| `/tmp` | Per-job fast local storage; automatically removed when the job ends |
| `/temp` | Fast local storage that persists briefly; deleted after 7 days unless refreshed with `touch /temp/<path>` on the head |
| `/archive/YEAR/PROJECT` | Long-term project storage; optimized for capacity, not speed |

Archive data should be essential, project-scoped, and accompanied by a sibling metadata JSON file under `/archive/YEAR/`.

## Environments

Use Apptainer `.sif` containers for software environments. Do not create large conda or venv directory trees on `/home`.

The login node does not have Apptainer for builds. Build containers on a compute node:

```bash
srun --partition=cpu-2h --pty bash
apptainer build python_container.sif python_container.def
```

Run containers:

```bash
apptainer run python_container.sif python script.py
apptainer run --nv python_container.sif python -c "import torch; print(torch.cuda.is_available())"
```

Pre-built GPU containers documented on Hydra:

```text
/opt/apps/pytorch-2.0.1-gpu.sif
/opt/apps/jax-0.4.13-gpu.sif
/opt/apps/tf-2.13.0-gpu.sif
```

Containers are immutable. Rebuild them when dependencies change, or use an overlay only when a job truly needs a writable layer.

## Datasets

Use SquashFS images for datasets with many files. Check shared squashed datasets first:

```text
/home/space/datasets-sqfs/
```

Shared datasets may also exist under `/home/space/datasets/`. For this thesis, the project-relevant TCGA-UT dataset location is:

```text
/home/space/datasets/patho_ds/tcga-ut
```

BRACS lives at:

```text
/home/space/datasets/patho_ds/BRACS
```

PANDA (prostate cancer grade assessment) lives at:

```text
/home/space/datasets/panda/raw
```

See `/home/space/datasets/panda/README.md` for source, license, and layout details.

Treat shared dataset directories as read-only unless you created the specific files yourself. It is fine to inspect `/home/space/datasets/` to find existing data, but do not modify, rename, or remove datasets created by other users.

Create SquashFS images from a compute node, not the login node:

```bash
srun --partition=cpu-2h --pty bash
squash-dataset /path/to/dataset /home/space/datasets-sqfs/name.sqfs
```

For training jobs, copy the `.sqfs` image to local `/tmp` and bind it into the Apptainer container:

```bash
cp /home/space/datasets-sqfs/name.sqfs /tmp/
apptainer run -B /tmp/name.sqfs:/input-data:image-src=/ container.sif python train.py
```

BeeOND can create a shared fast filesystem across multiple heads during a job. Request it with the `beeond` constraint when a multi-node job needs shared local-speed data access.

## Jupyter

Run Jupyter inside a SLURM job, not directly on the login node. Include `notebook` in the container, submit a job that starts Jupyter with `--ip 0.0.0.0 --no-browser`, then tunnel through Hydra.

Example tunnel after identifying the assigned compute head from the job/log output:

```bash
ssh -L 8888:headxyz:8888 -o ServerAliveInterval=60 hydra
```

Open the `127.0.0.1:8888` URL with the token from the job log.

## Git On Hydra

Keep the repo at `~/master-thesis` (clone: `git clone --depth 1 https://github.com/devWhyqueue/master-thesis.git ~/master-thesis`). Sync code with `git pull --ff-only`; gitignored artifacts under `experiments/<name>/` stay on disk. Run Hydra jobs from the relevant `experiments/<name>/` directory (see that experiment's README).

## Agent Safety Checklist

- Before running anything expensive, confirm whether the shell is on the login node or inside a SLURM allocation.
- Use `cpu-test` or `gpu-test` before long partitions.
- Keep heavy file reads off `/home`; stage datasets to `/tmp` inside jobs.
- Do not modify or remove shared datasets unless you created the specific files yourself.
- Prefer Apptainer and SquashFS over many small files.
- Never cancel jobs you did not start unless explicitly asked.
- Keep queued+running jobs at or under 100 per account (`squeue -u $USER | wc -l`) — colleagues share the cluster; shrink array/bundle size or stage submissions instead of flooding the queue.
- Use job-id-specific log names.
- Keep upstream docs as the source of truth when cluster behavior changes.
