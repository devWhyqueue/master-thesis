# TCGA-UT Class-Imbalance Experiment

Controlled class-imbalance benchmarks on TCGA-UT with two native input regimes:

- a **patch-level benchmark** on labeled histopathology patches,
- a **WSI-bag benchmark** on frozen Virchow2 feature bags.

The paper source is `paper/main.tex`; benchmark outputs are stored separately under
`outputs/tables/`, `outputs/figures/`, `outputs/results_patch/`, and
`outputs/results_wsi_bag/`.

## Shared Preparation

```bash
sbatch scripts/hydra/build_container.sbatch
export EXPERIMENT_CONTAINER="$PWD/environment.sif"
sbatch scripts/hydra/run_prepare.sbatch
```

Preparation builds one slide-level split manifest per seed and derives the
controlled patch manifests from those slide assignments, so patches from one
slide never cross splits.

## Benchmarks

Patch methods: `patch_ce`, `patch_weighted_ce`, `patch_focal`,
`patch_balanced_sampler_ce`, `patch_cfal`, `patch_progan_aug`.

WSI-bag methods: `mil_ce`, `mil_weighted_ce`, `mil_focal`,
`mil_balanced_sampler_ce`, `rankmix_mil`, `sc_mil`.

Submit the benchmarks independently:

```bash
sbatch scripts/hydra/run_patch_train_array.sbatch
sbatch scripts/hydra/run_wsi_train_array.sbatch
sbatch scripts/hydra/run_aggregate.sbatch
```

For an end-to-end smoke run:

```bash
sbatch scripts/hydra/run_smoke.sbatch
```

Before a full WSI sweep, profile the untruncated bags once:

```bash
sbatch scripts/hydra/run_wsi_profile.sbatch
```

The default configuration leaves `max_instances_per_bag` unset. If full bags are
not feasible on the cluster allocation, set one profiled fixed cap in
`configs/default.yaml` and report that cap in the paper.

`patch_progan_aug` is a bounded Ruiz-Casado-style adaptation: one class-specific
ProGAN is trained for every training class below the head-class patch count,
generators grow progressively to 256 px using the paper's depth-dependent batch
schedule, and synthetic patches raise those classes to the training-set head
count. The synthetic summary records per-class generated counts and Inception
FID whenever `torchvision` is available in the runtime environment.

## Main Artifacts

- `outputs/tables/result_summary_patch.csv`
- `outputs/tables/result_summary_wsi_bag.csv`
- `outputs/figures/method_macro_f1_patch_test.png`
- `outputs/figures/method_macro_f1_wsi_bag_test.png`
- `outputs/results_patch/<method>/seed=<seed>/`
- `outputs/results_wsi_bag/<method>/seed=<seed>/`

Runtime outputs such as generated synthetic patches, detailed result folders,
`data/`, and `logs/` remain scratch artifacts.
