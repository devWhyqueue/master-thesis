# TCGA-UT Class-Imbalance Experiment Scripts

This directory contains scripts for running controlled patch-level and WSI-bag class-imbalance experiments on TCGA-UT.

## Workflow Overview

1. **Pipeline Orchestrator:** Use `run_pipeline.py` to run the sequential steps locally. Supporting a `--smoke` mode for testing logic.
   ```bash
   uv run python -m code.run_pipeline --smoke
   ```
2. **Data Preparation (`data/`):** Initializes manifests, partitions splits case-disjointly (preventing patient leakage), and stages patch images locally on computing nodes.
3. **Model Training (`modeling/`):**
   * **Patch classifier:** ResNet models trained on staged patch images.
   * **WSI attention MIL:** Weakly supervised bag feature classifiers.
4. **Synthetic Augmentation (`data/progan/`):** Trains Progressive GANs on minority class patches up to 256px resolution to augment tail classes to the head-class count.
5. **Ingestion & Analysis (`analysis/`):** Loads run results into a central SQLite database, evaluates post-hoc calibration methods (Temperature, Vector, Dirichlet), and renders publication-ready LaTeX tables and matplotlib figures.
6. **Cluster Submission (`hydra/`):** Wrapper scripts to run array jobs, SquashFS mounts, and training phases on the Hydra SLURM cluster.

## Code Quality Check commands

Before committing code changes, make sure all tests and quality checks pass:

```bash
uv run ruff check experiments/class_imbalance/code    # Linter check
uv run vulture experiments/class_imbalance/code       # Dead code check
uv run pyright                                           # Type safety check
uv run pytest experiments/class_imbalance/tests          # Run tests
```
