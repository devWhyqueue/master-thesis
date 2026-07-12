# TCGA-UT Controlled Imbalance Benchmark

This experiment stack builds strict full-scale TCGA-UT training splits for the controlled follow-up to the native class-imbalance benchmark.

## Repository Layout
- [data/](file:///d:/Git/master-thesis/experiments/2_cls_imb_benchmark/native/code/data): Dataset classes (`dataset.py`), Virchow2 feature stores (`feature_store.py`, `feature_cache.py`), filtering (`filtering.py`), and full-scale constructed sampling (`full_scale/`).
- [modeling/](file:///d:/Git/master-thesis/experiments/2_cls_imb_benchmark/native/code/modeling): Loss functions (`losses/`), models (`models/`), and constructed WSI trainers (`training/`).
- [hydra/](file:///d:/Git/master-thesis/experiments/2_cls_imb_benchmark/native/code/hydra): Orchestrates SLURM and local execution for sampling, caching, training, and report aggregation.
- [analysis/](file:///d:/Git/master-thesis/experiments/2_cls_imb_benchmark/native/code/analysis): Plotting/visualizations (`plotting/`) and metrics evaluation (`evaluation/`).
- [tests/](file:///d:/Git/master-thesis/experiments/2_cls_imb_benchmark/native/code/tests): Smoke and unit tests.

## Setup
Copy the configuration template and fill in local dataset/feature directory paths:
```bash
cp hydra/config.json.template hydra/config.json
```

## Core Tasks

### 1. Derive One-Seed Feasibility
```bash
python hydra/run.py --local --no-container max-feasible-pool-size --parameter 1.3 --seed 0
```
Update `full_scale_pool_size` in `hydra/config.json` with the minimum feasible pool size across seeds.

### 2. Run Strict Sampling
```bash
python hydra/run.py sample-full-scale --parameter 0.8 --seed 0
```

### 3. Training & Tuning
```bash
python hydra/run.py tune      # Patch-feature tuning
python hydra/run.py tune-wsi  # WSI-bag tuning
```
