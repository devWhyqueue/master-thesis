# TCGA-UT Class-Imbalance Experiment

Controlled class-imbalance benchmarks on TCGA-UT with two native input regimes:

- a **patch-level benchmark** on labeled histopathology patches,
- a **WSI-bag benchmark** on frozen Virchow2 feature bags.

The paper source is `report/main.tex`. After a Hydra run, benchmark data lives in
`outputs/results.sqlite`; paper-facing tables and figures are under
`outputs/tables/*.tex` and `outputs/figures/*.png`.

## Hydra Workflow

```bash
bash scripts/hydra/submit.sh build-container
export EXPERIMENT_CONTAINER="$PWD/environment.sif"
bash scripts/hydra/submit.sh prepare
bash scripts/hydra/submit.sh patch-train
bash scripts/hydra/submit.sh progan
bash scripts/hydra/submit.sh wsi-train
bash scripts/hydra/submit.sh aggregate
```

Use `submit.sh` as the stable entry point for cluster jobs; it owns SLURM
partitions, arrays, dependencies, and optional maintenance jobs such as SquashFS
builds, smoke runs, WSI profiling, and ProGAN reruns.

```bash
bash scripts/hydra/submit.sh --help
```

Preparation builds one participant-level split manifest per seed and derives the
controlled patch manifests from those case assignments, so patches and features
from slides belonging to the same TCGA participant never cross splits. See
`CLUSTER.md` before changing Hydra storage or submission behavior.

## Benchmarks

Patch methods: `patch_ce`, `patch_weighted_ce`, `patch_focal`,
`patch_balanced_sampler_ce`, `patch_ce_soft_f1_balanced`,
`patch_ce_soft_mcc_balanced`, `patch_progan_aug`.

WSI-bag methods: `mil_ce`, `mil_weighted_ce`, `mil_focal`,
`mil_balanced_sampler_ce`, `rankmix_mil`, `sc_mil`.

Patch GPU jobs stage images to node-local storage before training. If the real
or synthetic patch SquashFS images are available, staging uses those images;
otherwise it falls back to the manifest paths.

`patch_progan_aug` is a bounded Ruiz-Casado-style adaptation: one class-specific
ProGAN is trained for every training class below the head-class patch count,
generators grow progressively to 256 px using the paper's depth-dependent batch
schedule, and synthetic patches raise those classes to the training-set head count.

The default configuration leaves `max_instances_per_bag` unset. If full bags are
not feasible on the cluster allocation, set one profiled fixed cap in
`configs/default.yaml` and report that cap in the paper.
