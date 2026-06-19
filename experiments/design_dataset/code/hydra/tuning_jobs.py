import argparse

from job_defs import Job, prefix
from analysis.evaluation.tuning_grid import task_count


def tune(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build patch validation-tuning jobs as one SLURM array."""
    count = task_count("patch")
    throttle = config.get("tune_array_throttle", "50")
    return [
        Job(
            _tune_cmd(args, config, "patch"),
            "tune_patch",
            "logs/tuning/tune_patch_%A_%a.out",
            partition=config.get("tune_partition", "cpu-2h"),
            array_spec=f"0-{count - 1}%{throttle}",
            cpus_per_task=int(config.get("cpus_per_task", "4")),
            time_limit=config.get("tune_time_limit"),
            mem=config.get("tune_mem", "32G"),
        )
    ]


def tune_wsi(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build WSI validation-tuning jobs as one SLURM array."""
    count = task_count("wsi")
    throttle = config.get("wsi_tune_array_throttle", "20")
    partition = config.get("wsi_partition", "gpu-2h")
    return [
        Job(
            _tune_cmd(args, config, "wsi", gpu=True),
            "tune_wsi",
            "logs/tuning/tune_wsi_%A_%a.out",
            partition=partition,
            gpus_per_node=1,
            array_spec=f"0-{count - 1}%{throttle}",
            cpus_per_task=int(config.get("cpus_per_task", "4")),
            time_limit=config.get("wsi_time_limit"),
        )
    ]


def tune_aggregate(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build tuning aggregation job."""
    cmd = prefix(config, args) + [
        "-m",
        "analysis.evaluation.tuning_aggregate",
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
    *,
    gpu: bool = False,
) -> list[str]:
    return prefix(config, args, gpu=gpu) + [
        "-m",
        "analysis.evaluation.tuning_run",
        f"--benchmark={benchmark}",
        f"--config={args.config}",
        f"--constructed-dataset-dir={config.get('constructed_dataset_dir', '')}",
        f"--results-dir={config.get('results_dir', '')}",
        f"--feature-path={config.get('feature_path', '')}",
    ]
