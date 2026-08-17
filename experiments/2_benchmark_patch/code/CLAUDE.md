# Class-Imbalance Benchmark Code

This directory is the unified `imbalance_benchmark` implementation. Use this map before searching broadly.

## Where to look

| Need | Primary location |
| --- | --- |
| CLI and command dispatch | [`__main__.py`](__main__.py), then `imbalance_benchmark/commands/` |
| Configuration, paths, output namespaces, run records, hashes | [`imbalance_benchmark/common.py`](imbalance_benchmark/common.py) and [`../configs/default.yaml`](../configs/default.yaml) |
| Dataset adapters, feature loading, and provenance | [`imbalance_benchmark/datasets/`](imbalance_benchmark/datasets/) |
| Controlled class allocations and evidence selection | [`imbalance_benchmark/construction.py`](imbalance_benchmark/construction.py), then `manifest/construction_*.py` |
| Pilot support study, floors, seeds, signed freeze | [`imbalance_benchmark/manifest/`](imbalance_benchmark/manifest/) |
| Models, losses, training, and evaluation | [`imbalance_benchmark/modeling/`](imbalance_benchmark/modeling/) |
| Tuning or confirmation orchestration | `commands/tuning.py`, `commands/confirm.py`, and `modeling/workflows/` |
| Metrics, calibration, and result aggregation | [`imbalance_benchmark/analysis/`](imbalance_benchmark/analysis/) |
| Bootstrap, permutation tests, gates, recovery, Holm | `analysis/inference/` |
| Tables, plots, ingestion, completeness, clustered endpoints | `analysis/reporting/` |
| RQ3 predictors, signal profile, and hierarchical models | `analysis/predictors/` |
| Hydra/SLURM submission generation | [`imbalance_benchmark/hydra/workflow.py`](imbalance_benchmark/hydra/workflow.py) |
| Regression coverage | [`tests/`](tests/); start with the closest `test_<area>*.py` |

`commands/` coordinates phases; reusable scientific logic belongs in the layer named above, not in a command handler.

## Workflow and protocol

- The command order is `prepare` -> `pilot` -> `freeze` -> `signals` -> `match` -> `tune` -> `confirm` -> `analyze`; `match` only needs `signals` and is cross-dataset like `combine-rq3`, which runs only after split-level analysis.
- Use `__main__.py` from this directory: `uv run python __main__.py --config ../configs/default.yaml prepare`. `smoke` exercises the local end-to-end path; `submit --dry-run` previews Hydra submission.
- [`../report/2_benchmark_protocol_patch.tex`](../report/2_benchmark_protocol_patch.tex) is the protocol authority. Do not edit it unless explicitly asked.
- Preserve patient splits, seed families, tuning locks, signed manifests, and frozen evidence. Never select a configuration or replace a failed run using test results.
- Use the default configuration for paths. Outputs are generated evidence: do not hand-edit manifests, selections, or run records.

## Change gate

From the repository root:

```powershell
uv run pytest experiments/2_benchmark_patch/code/tests
uv run python "$env:USERPROFILE\.codex\skills\clean-code\run.py" --scope experiments/2_benchmark_patch/code --vulture-scope experiments/2_benchmark_patch/code
```

Run the clean-code gate after each Python change and fix findings without weakening rules. For Hydra or SLURM work, first follow the repository's `hydra-cluster` instructions and `CLUSTER.md`.
