# Agent Navigation & Script Guide: `experiments/class_imbalance/scripts`

This document provides a concise navigation map of the scripts directory for AI agents.

## Core Navigation Map

```
scripts/
├── run_pipeline.py          # CLI pipeline orchestrator (runs prep -> patch -> WSI sequentially)
├── common.py                # Shared config loading, path resolution, and run-record json I/O
├── metadata.py              # Central metadata definitions for methods & taxonomy mapping
├── data/
│   ├── prep/                # Manifest generation (splits.py, feature.py, patch.py) and bag cache
│   ├── progan/              # Ruiz-Casado style progressive GAN synthetic augmentation
│   └── staging/             # Node-local SSD image staging (io.py via ThreadPoolExecutor/SquashFS)
├── modeling/
│   ├── patch/               # ResNet-based patch-level classifier training (train.py, data.py)
│   ├── mil/                 # Attention MIL dataset, training loss definitions, and metrics
│   └── training/            # Orchestrator for WSI-bag classification training loops
├── analysis/
│   ├── results/             # SQLite result DB schemas, queries, and ingestion
│   ├── report/              # Plotting, paired delta tables, and post-hoc calibration
│   └── tuning/              # Hparam sweep aggregation and reporting
└── hydra/                   # Cluster job submission wrappers (submit.sh, job.sbatch)
```

## Common Developer Workflows

### Running the pipeline locally (Smoke Mode)
To run a fast validation pipeline locally on a subset of data:
```bash
uv run python -m scripts.run_pipeline --smoke
```

### Running Cluster Jobs
Submit jobs via the wrapper script to handle SLURM partitions and arrays:
```bash
bash scripts/hydra/submit.sh [prepare|patch-train|progan|wsi-train|aggregate]
```

### Running Checks (Ruff, Vulture, Pyright, Pytest)
Always verify your edits before final submission:
* **Linter:** `uv run ruff check experiments/class_imbalance/scripts`
* **Dead Code:** `uv run vulture experiments/class_imbalance/scripts`
* **Type Safety:** `uv run pyright`
* **Tests:** `uv run pytest experiments/class_imbalance/tests`

## Style & Architecture Guidelines for Agents
* **No abstractions:** Standard library or native Pandas/PyTorch operations should be preferred over helper classes.
* **Strict Type Safety:**
  * Return values of dataframe transformations (e.g., `.drop()`, `pd.concat()`, `groupby().agg()`) should be cast back to `pd.DataFrame` or `pd.Series` via `typing.cast()` to pass Pyright.
  * In `scipy.optimize.minimize`, pass solver arguments (like `maxiter`) through the `options` dict: `minimize(..., options={"maxiter": 250})`.
* **Unused code:** Keep the codebase clean. If imports or helper functions are not actively called, remove them completely rather than leaving commented-out sections.
