import argparse

from job_defs import Job, parameters, prefix


def wsi_cache(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build constructed WSI bag-cache jobs."""
    seeds = [0, 1, 2] if args.sweep else [args.seed]
    return [
        Job(
            _wsi_cache_cmd(args, config, parameter, seed),
            "wsi_cache",
            "logs/wsi/wsi_cache%j.out",
        )
        for parameter in parameters(args)
        for seed in seeds
    ]


def _wsi_cache_cmd(
    args: argparse.Namespace,
    config: dict[str, str],
    parameter: float,
    seed: int,
) -> list[str]:
    stem = _constructed_stem(args, config, parameter, seed)
    return prefix(config, args) + [
        "-m",
        "tcga_ut_imbalanced.training.constructed_wsi_cache",
        f"--manifest-path={stem}/manifest_splits.csv",
        f"--cache-dir={stem}/wsi_bag_cache",
    ]


def _constructed_stem(
    args: argparse.Namespace,
    config: dict[str, str],
    parameter: float,
    seed: int,
) -> str:
    name = (
        f"constructed_order={args.class_order_name}_parameter={parameter}_seed={seed}"
    )
    return f"{config.get('constructed_dataset_dir', '')}/{name}"
