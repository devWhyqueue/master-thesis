# TCGA-UT Class-Imbalance Experiment

Taxonomy-grounded class-imbalance experiments on TCGA-UT using frozen Virchow2
WSI feature bags.

Default Hydra feature path:

```text
/home/space/datasets/patho_ds/tcga-ut/patch_features/cls_patchmean/virchow_virchow2/raw
```

## Main Hydra Run

Submit from this experiment directory:

```bash
sbatch scripts/hydra/build_container.sbatch
export EXPERIMENT_CONTAINER="$PWD/environment.sif"
sbatch scripts/hydra/run_smoke.sbatch
sbatch scripts/hydra/run_prepare.sbatch
sbatch scripts/hydra/run_train_array.sbatch
sbatch scripts/hydra/run_aggregate.sbatch
```

Outputs used by the paper are kept in `outputs/tables/` and `outputs/figures/`.
Runtime outputs such as `outputs/results/`, `outputs/synthetic_*`, `data/`, and
`logs/` are scratch artifacts.

## Methods

Baselines: `ce`, `weighted_ce`, `focal`, `balanced_sampler_ce`, `knn`, `ncc`.

Representative methods: `rankmix_mil`, `feature_gan_mil`, `cfal_mil`,
`mde_mil`, `sc_mil`.

## Image-GAN Bridge

The full synthetic-image path is:

```bash
gan=$(sbatch --parsable scripts/hydra/run_synthetic_gan.sbatch)
collect=$(sbatch --parsable --dependency=afterok:$gan scripts/hydra/run_synthetic_collect.sbatch)
encode=$(sbatch --parsable --dependency=afterok:$collect scripts/hydra/run_synthetic_encode.sbatch)
feature_gan=$(sbatch --parsable --dependency=afterok:$encode scripts/hydra/run_feature_gan_mil_array.sbatch)
sbatch --dependency=afterok:$feature_gan scripts/hydra/run_aggregate.sbatch
```

`run_synthetic_encode.sbatch` needs `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` for the
gated `paige-ai/Virchow2` model.

Latest completed run: image-GAN job `4341844` generated 4096 patches per seed,
encoder job `4341846` encoded 4096 features per seed, and `feature_gan_mil` job
`4341847` consumed all encoded synthetic features.

## Monitoring

```bash
squeue -u "$USER"
cat outputs/results/<method>/seed=<seed>/progress.json
cat outputs/tables/missing_results.json
```

The paper source is `paper/main.tex`; rebuild it after refreshing tables or
figures.
