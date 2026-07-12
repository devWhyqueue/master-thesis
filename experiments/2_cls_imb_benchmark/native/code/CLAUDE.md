# AGENTS.md - 2_cls_imb_benchmark/native/code

AI assistant guidelines for working on the TCGA-UT Controlled Imbalance Benchmark codebase.

## Development Workflow
- **Root Context**: Run all commands and python modules from the repository root directory (`d:\Git\master-thesis`).
- **Configuration**: Copy `hydra/config.json.template` to `hydra/config.json` and customize environment paths. Do not commit `hydra/config.json`.
- **Dependencies**: Relies on `master-thesis-tools` environment dependencies (standard Pytorch, Scikit-learn, Pandas, Numpy, and custom local `common_code`).

## Package Guidelines
- **data/full_scale/**: Sampling logic. Any new sampling strategy must be integrated into `sampling.py` or placed here. Keep dependencies contained.
- **modeling/training/**: `constructed_wsi_data.py` contains `ConstructedBagDataset` which inherits from `common_code.wsi.bag_dataset.BagFeatureDataset`. Do not break this typing hierarchy.
- **common_code**: Shared benchmark logic is imported from `common_code` package (located in `experiments/shared/common_code`). Avoid duplicating logic in local modules.

## Clean-Code Enforcement
- Run the `/clean-code` audit command before committing any python edits:
  `uv run python "C:\Users\Yannik\.claude\skills\clean-code\run.py" --scope 2_cls_imb_benchmark/native/code`
  Do **not** add `--vulture-scope 2_cls_imb_benchmark/native/code`; that flag passes invalid paths and causes vulture to exit with an error.
- Unused code/modules must be deleted immediately (per the lazy senior dev ponytail rule) rather than commented out or left dead.
