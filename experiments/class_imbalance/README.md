# TCGA-UT Class-Imbalance Experiment

Controlled class-imbalance benchmarks on TCGA-UT: patch-level classifier and WSI attention-MIL, with a ProGAN synthetic augmentation bridge. Paper source: `report/main.tex`.

## Key paths

| Path | Purpose |
|---|---|
| `configs/default.yaml` | Shared experiment config (seeds, splits, caps) |
| `configs/environment.def` | Apptainer container definition |
| `configs/requirements-experiment.txt` | Container Python dependencies |
| `code/` | All Python pipeline code — see `code/README.md` |
| `code/hydra/submit.sh` | Stable entry point for all cluster jobs |
| `code/run_pipeline.py` | Local sequential pipeline runner |
| `outputs/results.sqlite` | Generated: benchmark results DB (after `aggregate`) |
| `outputs/tables/` | Generated: paper-facing LaTeX tables |
| `outputs/figures/` | Generated: paper-facing figures |
| `report/main.tex` | Paper source |
| `report/main.pdf` | Current paper draft |
| `tests/` | `test_benchmarks.py`, `conftest.py` |

## Hydra workflow

```bash
bash code/hydra/submit.sh build-container
export EXPERIMENT_CONTAINER="$PWD/environment.sif"
bash code/hydra/submit.sh prepare
bash code/hydra/submit.sh patch-train
bash code/hydra/submit.sh progan
bash code/hydra/submit.sh wsi-train
bash code/hydra/submit.sh aggregate
```

`prepare` builds participant-level split manifests and derives patch manifests case-disjointly (no patient leakage across splits). See `CLUSTER.md` before changing storage or submission behaviour.

```bash
bash code/hydra/submit.sh --help
```

## Benchmarks

**Patch:** `patch_ce`, `patch_weighted_ce`, `patch_focal`, `patch_balanced_sampler_ce`, `patch_ce_soft_f1_balanced`, `patch_ce_soft_mcc_balanced`, `patch_progan_aug`

**WSI-bag:** `mil_ce`, `mil_weighted_ce`, `mil_focal`, `mil_balanced_sampler_ce`, `rankmix_mil`, `sc_mil`

`patch_progan_aug` trains one class-specific ProGAN per minority class, growing progressively to 256 px, and raises those classes to the head-class patch count.

## Development

```bash
uv run python -m code.run_pipeline --smoke   # fast local validation
uv run pytest experiments/class_imbalance/tests
uv run ruff check experiments/class_imbalance/code
uv run vulture experiments/class_imbalance/code
uv run pyright
```
