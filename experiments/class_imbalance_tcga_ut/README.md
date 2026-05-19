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
`patch_balanced_sampler_ce`, `patch_ce_soft_f1_balanced`, `patch_ce_soft_mcc_balanced`,
`patch_progan_aug`.

WSI-bag methods: `mil_ce`, `mil_weighted_ce`, `mil_focal`,
`mil_balanced_sampler_ce`, `rankmix_mil`, `sc_mil`.

Submit the benchmarks independently:

```bash
sbatch scripts/hydra/run_patch_train_array.sbatch
bash scripts/hydra/submit_patch_progan.sh
sbatch scripts/hydra/run_wsi_train_array.sbatch
sbatch scripts/hydra/run_aggregate.sbatch
```

### Patch I/O on Hydra

Patch GPU jobs stage images to node-local `$SLURM_TMPDIR` before training (see
`CLUSTER.md`). Each job runs `scripts.staging.patch`, then trains from the staged
manifest. If `paths.patch_sqfs` exists on the cluster, staging mounts that SquashFS via
`squashfuse`; otherwise it hardlinks/copies the manifest images into `$SLURM_TMPDIR`.

One-time SquashFS build (recommended for repeated patch runs):

```bash
sbatch scripts/hydra/build_patch_sqfs.sbatch
```

`run_patch_train_array.sbatch` covers the six non-ProGAN patch methods (`array=0-17`).
ProGAN is submitted separately: one SLURM array task per `(seed, tail class)` on
`gpu-5h` with `--constraint=80gb|40gb|h100`, capped at 35 concurrent GPUs (Hydra account
limit), then a dependent three-task array trains `patch_progan_aug` on `gpu-2d` after all
GAN jobs finish. Classifier training writes `checkpoint_latest.pt` each epoch and
supports `--resume` (used by the train array sbatch). Reuse completed class folders when
counts still match the manifest.

To rerun classifier training only (GAN artifacts already on disk):

```bash
sbatch scripts/hydra/run_patch_progan_train_array.sbatch
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
count. The generator uses ProGAN-style pixel normalization and minibatch standard
deviation, validates cached generated patches against the current manifest, and
writes per-class generation counts, per-depth training diagnostics, and Inception
FID status whenever `torchvision` is available in the runtime environment.
Patch checkpoints include the class list and model weights for standard logit evaluation.

## Main Artifacts

- `outputs/tables/result_summary_patch.csv`
- `outputs/tables/result_summary_wsi_bag.csv`
- `outputs/figures/method_macro_f1_patch_test.png`
- `outputs/figures/method_macro_f1_wsi_bag_test.png`
- `outputs/results_patch/<method>/seed=<seed>/`
- `outputs/results_wsi_bag/<method>/seed=<seed>/`

Runtime outputs such as generated synthetic patches, detailed result folders,
`data/`, and `logs/` remain scratch artifacts.
