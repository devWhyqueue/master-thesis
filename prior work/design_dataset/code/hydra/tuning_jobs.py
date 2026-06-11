import argparse

from job_defs import Job, prefix
from tcga_ut_imbalanced.evaluation.tuning_grid import task_count


def tune(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build patch validation-tuning jobs."""
    return [
        Job(
            _tune_cmd(args, config, "patch", index),
            "tune_patch",
            "logs/tuning/tune_patch%j.out",
            partition=config.get("tune_partition", "cpu-2h"),
        )
        for index in range(task_count("patch"))
    ]


def tune_wsi(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build WSI validation-tuning jobs."""
    partition = config.get("wsi_partition", "gpu-2h")
    return [
        Job(
            _tune_cmd(args, config, "wsi", index),
            "tune_wsi",
            "logs/tuning/tune_wsi%j.out",
            partition=partition,
            gpus_per_node=1,
        )
        for index in range(task_count("wsi"))
    ]


def tune_aggregate(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build tuning aggregation job."""
    cmd = prefix(config, args) + [
        "-m",
        "tcga_ut_imbalanced.evaluation.tuning_aggregate",
        f"--results-dir={config.get('results_dir', '')}",
        f"--output-dir={config.get('report_output_dir', '')}",
    ]
    if args.allow_incomplete:
        cmd.append("--allow-incomplete")
    return [
        Job(cmd, "tune_aggregate", "logs/tuning/tune_aggregate%j.out"),
    ]


def _tune_cmd(
    args: argparse.Namespace,
    config: dict[str, str],
    benchmark: str,
    array_index: int,
) -> list[str]:
    cmd = prefix(config, args) + [
        "-m",
        "tcga_ut_imbalanced.evaluation.tuning_run",
        f"--benchmark={benchmark}",
        f"--array-task-id={array_index}",
        f"--config={args.config}",
        f"--constructed-dataset-dir={config.get('constructed_dataset_dir', '')}",
        f"--results-dir={config.get('results_dir', '')}",
        f"--feature-path={config.get('feature_path', '')}",
    ]
    root = config.get("class_imbalance_root", "")
    if root:
        cmd.append(f"--class-imbalance-root={root}")
    return cmd
