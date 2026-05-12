# Native TCGA-UT Class-Imbalance Experiments

This experiment package studies mitigation methods for the native TCGA-UT
class distribution using a shared frozen feature representation.

Default feature path on Hydra:

```text
/home/space/datasets/patho_ds/tcga-ut/patch_features/cls_patchmean/virchow_virchow2/raw
```

## Workflow

From this directory:

```bash
python -m scripts.prep.manifest --config configs/default.yaml
python -m scripts.prep.splits --config configs/default.yaml
python -m scripts.prep.explore --config configs/default.yaml
python -m scripts.training.train --config configs/default.yaml --method ce --seed 0
python -m scripts.report.aggregate --config configs/default.yaml
python -m scripts.report.figures --config configs/default.yaml
```

On Hydra, submit from this experiment directory:

```bash
sbatch scripts/hydra/build_container.sbatch
# Wait until the build job has finished successfully.
export EXPERIMENT_CONTAINER="$PWD/environment.sif"
sbatch scripts/hydra/run_smoke.sbatch
sbatch scripts/hydra/run_prepare.sbatch
sbatch scripts/hydra/run_train_array.sbatch
sbatch scripts/hydra/run_aggregate.sbatch
```

The default Hydra `python3` may not include `torch` and `scikit-learn`. To use an
already-built image instead of `build_container.sbatch`, set:

```bash
export EXPERIMENT_CONTAINER=/path/to/environment.sif
sbatch scripts/hydra/run_smoke.sbatch
```

The smoke job uses only the first seed and the `ce`/`knn` methods with a very
small per-class row cap. Use it to verify manifest construction, feature loading,
and result writing before launching the full sweep.

`run_all.sbatch` still exists as a sequential fallback, but the prepare +
training-array + aggregate sequence is the intended Hydra workflow.
The training array is capped at four concurrent tasks to avoid overwhelming
shared storage while each job preloads feature tensors.

The Apptainer definition and Python requirements live in `configs/`.
The paper lives in `paper/main.tex`. Generated figures and tables are written
under `outputs/figures/` and `outputs/tables/`.
Per-run JSON files, progress files, model checkpoints, logs, and manifests are
runtime artifacts and are ignored by Git. After aggregation, detailed per-run
metrics are also written to one compressed artifact:

```text
outputs/tables/result_details.jsonl.gz
```

Keep `outputs/tables/` and `outputs/figures/` for paper artifacts; keep
`outputs/results/`, `data/`, and `logs/` as local or Hydra scratch outputs.

## Current status

- The experiment code has been validated with a local synthetic smoke fixture.
- Native TCGA-UT experiments have not yet been conducted.
- Before treating the paper as a results manuscript, run the Hydra workflow and
  inspect `outputs/tables/missing_results.json`; its `missing` list should be
  empty for the configured methods and seeds.
- Keep numerical claims out of `paper/main.tex` until they are generated from
  saved result JSON files.

## Monitoring Hydra runs

Use SLURM plus per-method progress files to catch slow or stuck jobs:

```bash
squeue -u "$USER"
tail -f logs/train-<array-job-id>-<task-id>.out
cat outputs/results/<method>/seed=<seed>/progress.json
cat outputs/tables/missing_results.json
```

MLP-style methods update `progress.json` after every epoch. Probe methods such
as KNN and nearest centroid write coarse `started` and `completed` statuses.
Training logs also report one-time feature preload counts before the first
epoch, for example `Loaded 36542 feature tensors into memory`.
Cancel individual slow array tasks with `scancel <array-job-id>_<task-id>`.
