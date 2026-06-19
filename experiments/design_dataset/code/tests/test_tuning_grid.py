from argparse import Namespace

from analysis.evaluation.tuning_grid import patch_grid, task_count, task_for_index, wsi_grid
from analysis.evaluation.tuning_run import _patch_command


def test_tuning_task_counts() -> None:
    assert task_count("patch") == len(patch_grid()) * 3 * 3
    assert task_count("wsi") == len(wsi_grid()) * 3 * 3


def test_patch_grid_restores_progan_and_oko() -> None:
    methods = {variant.method for variant in patch_grid()}

    assert "patch_feature_progan_aug" in methods
    assert "patch_feature_oko" in methods


def test_patch_command_uses_progan_manifest_and_split_filters(tmp_path) -> None:
    stem = (
        tmp_path
        / "constructed_order=native_prevalence_parameter=0.8_seed=0"
    )
    stem.mkdir()
    (stem / "patch_feature_cache_progan.pt").write_bytes(b"x")
    task = next(task for task in [task_for_index("patch", index) for index in range(task_count("patch"))] if task.variant.method == "patch_feature_progan_aug")
    args = Namespace(
        benchmark="patch",
        array_task_id=0,
        config="config.json",
        constructed_dataset_dir=str(tmp_path),
        results_dir=str(tmp_path / "results"),
        feature_path=str(tmp_path / "features"),
        dry_run=False,
    )

    cmd = _patch_command(args, task, str(tmp_path / "out"))

    dataset_arg = next(arg for arg in cmd if arg.startswith("--dataset-structure-path="))
    assert dataset_arg.endswith("manifest_splits_progan.csv")
    assert "--dataset-split=train" in cmd
    assert "--validation-dataset-split=validation" in cmd
    assert "--test-dataset-split=test" in cmd
    assert f"--feature-cache-path={stem / 'patch_feature_cache_progan.pt'}" in cmd
