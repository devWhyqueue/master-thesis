import argparse

from job_defs import Job, parameters, prefix


def patch_cache(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build per-regime patch feature cache jobs."""
    seeds = [0, 1, 2] if args.sweep else [args.seed]
    return [
        Job(
            _patch_cache_cmd(args, config, parameter, seed),
            "patch_cache",
            "logs/patch/patch_cache%j.out",
            # ponytail: cpu-9m: patch cache completes in ~3 min (sacct evidence)
            partition="cpu-9m",
        )
        for parameter in parameters(args)
        for seed in seeds
    ]


def _patch_cache_cmd(
    args: argparse.Namespace,
    config: dict[str, str],
    parameter: float,
    seed: int,
) -> list[str]:
    stem = _constructed_stem(args, config, parameter, seed)
    return prefix(config, args) + [
        "hydra/build_feature_cache.py",
        f"--manifest-path={stem}/manifest_splits.csv",
        f"--file-save-path={stem}/patch_feature_cache.pt",
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
