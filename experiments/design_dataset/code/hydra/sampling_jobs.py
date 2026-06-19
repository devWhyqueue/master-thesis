import argparse

from job_defs import Job, parameters, prefix


def max_feasible_pool_size(
    args: argparse.Namespace, config: dict[str, str]
) -> list[Job]:
    """Build one job that reports the strict one-seed feasible pool size."""
    cmd = prefix(config, args) + [
        "-m",
        "data.full_scale_cli",
        f"--slide-manifest-path={_seed_path(config.get('slide_manifest_csv', ''), args.seed)}",
        f"--parameter={args.parameter}",
        f"--seed={args.seed}",
        f"--class-order-name={args.class_order_name}",
        "--max-feasible-pool-size",
        f"--train-name={config.get('train_split_name', 'train')}",
        f"--validation-name={config.get('validation_split_name', 'validation')}",
        f"--test-name={config.get('test_split_name', 'test')}",
    ]
    split_path = _seed_path(config.get("split_assignment_csv", ""), args.seed)
    if split_path:
        cmd.append(f"--split-assignment-path={split_path}")
    if args.class_order_file is not None:
        cmd.append(f"--class-order-file={args.class_order_file}")
    return [Job(cmd, "max_pool", "logs/sampling/max_feasible_pool_size%j.out")]


def sample_full_scale(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build strict full-scale constructed sampling jobs."""
    return [
        Job(
            _sample_full_scale_cmd(args, config, parameter, seed),
            "sample_full",
            "logs/sampling/sample_full_scale%j.out",
        )
        for parameter in parameters(args)
        for seed in ([0, 1, 2] if args.sweep else [args.seed])
    ]


def _sample_full_scale_cmd(
    args: argparse.Namespace,
    config: dict[str, str],
    parameter: float,
    seed: int,
) -> list[str]:
    pool_size = config.get("full_scale_pool_size")
    if pool_size is None:
        raise ValueError(
            "Missing config key full_scale_pool_size for strict constructed sampling."
        )
    cmd = prefix(config, args) + [
        "-m",
        "data.full_scale_cli",
        f"--slide-manifest-path={_seed_path(config.get('slide_manifest_csv', ''), seed)}",
        f"--file-save-path={config.get('constructed_dataset_dir', '')}",
        f"--parameter={parameter}",
        f"--seed={seed}",
        f"--pool-size={pool_size}",
        f"--class-order-name={args.class_order_name}",
        "--n-patches-per-slide=30",
        f"--train-name={config.get('train_split_name', 'train')}",
        f"--validation-name={config.get('validation_split_name', 'validation')}",
        f"--test-name={config.get('test_split_name', 'test')}",
    ]
    split_path = _seed_path(config.get("split_assignment_csv", ""), seed)
    if split_path:
        cmd.append(f"--split-assignment-path={split_path}")
    if args.class_order_file is not None:
        cmd.append(f"--class-order-file={args.class_order_file}")
    feature_dir = config.get("feature_path", "")
    if feature_dir:
        cmd.append(f"--feature-dir={feature_dir}")
    return cmd


def _seed_path(path: str, seed: int) -> str:
    return path.format(seed=seed) if path else path
