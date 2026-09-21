# Hydra Cluster

Summary of TU Berlin Hydra cluster docs for agents in this repo.
Source documentation: https://git.tu-berlin.de/ml-group/hydra/documentation at commit `bbef4c93688269287b9a1007aab0b50ec637eb87`.

Hydra = HPC cluster of ML, MLSEC, UNIML, Cognition groups at TU Berlin. SLURM, separate login/main and compute nodes. Unrelated to Python `hydra` or `hydra-zen`.

## Access

SSH shortcut from WSL or PowerShell:

```bash
ssh hydra
```

Raw fallback:

```bash
ssh yannik.qu@hydra.ml.tu-berlin.de
```

Shortcut resolves to:

```sshconfig
Host hydra
  HostName hydra.ml.tu-berlin.de
  User yannik.qu
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  ForwardAgent yes
  ServerAliveInterval 60
```

PowerShell uses same key via `C:/Users/Yannik/.ssh/id_ed25519`.

No compute-heavy work on main/login node. Use it to prepare files, inspect state, submit SLURM jobs.

## SLURM Basics

Prefer batch jobs (`sbatch`). Interactive jobs only when necessary, keep short.

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

Use `logs/job-%j.out` or other job-id-based path so parallel jobs not overwrite each other.

## Partitions And GPUs

Pick shortest runtime that finishes job. Shorter partitions = higher scheduling priority; long partitions = fewer running slots per account. Validate scripts + containers on test partitions before long jobs.

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

GPU constraints:

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

CPU partitions over 2h use gang scheduling: jobs can be suspended and resumed in slices.

## Files And Storage

`/home` = shared BeeGFS across heads. Avoid many small files; each lookup creates network filesystem traffic. Prefer single large image files for environments and datasets.

| Path | Use |
| --- | --- |
| `/home/<user>` | Code, job scripts, logs, containers, small config files |
| `/tmp` | Per-job fast local storage; removed when job ends |
| `/temp` | Fast local storage, persists briefly; deleted after 7 days unless refreshed with `touch /temp/<path>` on head |
| `/archive/YEAR/PROJECT` | Long-term project storage; optimized for capacity, not speed |

Archive data: essential, project-scoped, with sibling metadata JSON file under `/archive/YEAR/`.

## Environments

Use Apptainer `.sif` containers for software environments. No large conda or venv trees on `/home`.

Login node has no Apptainer for builds. Build on compute node:

```bash
srun --partition=cpu-2h --pty bash
apptainer build python_container.sif python_container.def
```

Run containers:

```bash
apptainer run python_container.sif python script.py
apptainer run --nv python_container.sif python -c "import torch; print(torch.cuda.is_available())"
```

Pre-built GPU containers on Hydra:

```text
/opt/apps/pytorch-2.0.1-gpu.sif
/opt/apps/jax-0.4.13-gpu.sif
/opt/apps/tf-2.13.0-gpu.sif
```

Containers immutable. Rebuild when dependencies change; overlay only when job truly needs writable layer.

## Datasets

Use SquashFS images for datasets with many files. Check shared squashed datasets first:

```text
/home/space/datasets-sqfs/
```

Shared datasets also under `/home/space/datasets/`. TCGA-UT for this thesis:

```text
/home/space/datasets/patho_ds/tcga-ut
```

BRACS:

```text
/home/space/datasets/patho_ds/BRACS
```

PANDA (prostate cancer grade assessment):

```text
/home/space/datasets/panda/raw
```

Source, license, layout: `/home/space/datasets/panda/README.md`.

CAMELYON16:

```text
/home/space/datasets/camelyon16
```

Pre-tiled 20x patches under `patches/20x/<slide>/<id>.jpg`; SquashFS copy staged from `/home/space/datasets-sqfs/camelyon16-patches-20x.sqfs` for `prepare` stage.

Shared dataset directories read-only unless you created the specific files. Inspecting `/home/space/datasets/` fine; never modify, rename, or remove datasets of other users.

Create SquashFS images on compute node, not login node:

```bash
srun --partition=cpu-2h --pty bash
squash-dataset /path/to/dataset /home/space/datasets-sqfs/name.sqfs
```

Training jobs: copy `.sqfs` image to local `/tmp`, bind into Apptainer container:

```bash
cp /home/space/datasets-sqfs/name.sqfs /tmp/
apptainer run -B /tmp/name.sqfs:/input-data:image-src=/ container.sif python train.py
```

BeeOND creates shared fast filesystem across multiple heads during job. Request with `beeond` constraint when multi-node job needs shared local-speed data access.

## Jupyter

Run Jupyter inside SLURM job, never directly on login node. Include `notebook` in container, submit job starting Jupyter with `--ip 0.0.0.0 --no-browser`, tunnel through Hydra.

Tunnel after finding assigned compute head in job/log output:

```bash
ssh -L 8888:headxyz:8888 -o ServerAliveInterval=60 hydra
```

Open `127.0.0.1:8888` URL with token from job log.

## Git On Hydra

Repo at `~/master-thesis` (clone: `git clone --depth 1 https://github.com/devWhyqueue/master-thesis.git ~/master-thesis`). Sync with `git pull --ff-only`; gitignored artifacts under `experiments/<name>/` stay on disk. Run Hydra jobs from relevant `experiments/<name>/` directory (see that experiment's README).

## Agent Safety Checklist

- Before anything expensive, confirm shell is on login node or inside SLURM allocation.
- Use `cpu-test` or `gpu-test` before long partitions.
- Keep heavy file reads off `/home`; stage datasets to `/tmp` inside jobs.
- Never modify or remove shared datasets unless you created the specific files.
- Prefer Apptainer + SquashFS over many small files.
- Never cancel jobs you did not start unless explicitly asked.
- Keep queued+running jobs ≤ 100 per account: count with `squeue -u $USER -r | tail -n +2 | wc -l`. Shrink array/bundle size, pass only incomplete indices on resubmit, or stage submissions instead of flooding queue.
- Use job-id-specific log names.
- Upstream docs = source of truth when cluster behavior changes.
