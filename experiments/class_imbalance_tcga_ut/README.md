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

The Apptainer definition and Python requirements live in `configs/`.
The paper lives in `paper/main.tex`. Generated figures and tables are written
under `outputs/figures/` and `outputs/tables/`.
