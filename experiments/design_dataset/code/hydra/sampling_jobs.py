import argparse

from job_defs import Job, parameters, prefix


def sample_balanced(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build balanced-sampling jobs."""
    cmd = prefix(config, args) + [
        "-m",
        "tcga_ut_imbalanced.cli.sample_balanced",
        f"--dataset-path={config.get('dataset_path', '')}",
        f"--file-save-path={config.get('balanced_dataset_csv', '')}",
        f"--n-slides-per-class={args.n_slides_per_class}",
        f"--n-patches-per-slide={args.n_patches_per_slide}",
    ]
    return [Job(cmd, "sample_balanced", "logs/sampling/sample_balanced%j.out")]


def sample_imbalanced(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build imbalanced-sampling jobs."""
    return [
        Job(
            _sample_imbalanced_cmd(args, config, parameter),
            "sample",
            "logs/sampling/sample_imbalanced%j.out",
        )
        for parameter in parameters(args)
    ]


def sample_full_scale(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build full-scale constructed sampling jobs."""
    return [
        Job(
            _sample_full_scale_cmd(args, config, parameter, seed),
            "sample_full",
            "logs/sampling/sample_full_scale%j.out",
        )
        for parameter in parameters(args)
        for seed in ([0, 1, 2] if args.sweep else [args.seed])
    ]


def _sample_imbalanced_cmd(
    args: argparse.Namespace,
    config: dict[str, str],
    parameter: float,
) -> list[str]:
    bal_csv = config.get("balanced_dataset_csv", "")
    out = _imbalanced_csv(config.get("imbalanced_dataset_dir", ""), parameter)
    return prefix(config, args) + [
        "-m",
        "tcga_ut_imbalanced.cli.sample_imbalanced",
        f"--balanced-dataset-path={bal_csv}",
        f"--file-save-path={out}",
        f"--parameter={parameter}",
        "--dataset-size=500",
        "--sample-balanced-validation",
        "--n-slides-per-class=10",
        "--visualize",
        '--overflow-strategy="redistribute"',
        "--n-regions-per-slide=3",
        "--n-patches-per-region=10",
        "--store-class-names",
    ]


def _sample_full_scale_cmd(
    args: argparse.Namespace,
    config: dict[str, str],
    parameter: float,
    seed: int,
) -> list[str]:
    cmd = prefix(config, args) + [
        "-m",
        "tcga_ut_imbalanced.data.full_scale_cli",
        f"--slide-manifest-path={_seed_path(config.get('slide_manifest_csv', ''), seed)}",
        f"--file-save-path={config.get('constructed_dataset_dir', '')}",
        f"--parameter={parameter}",
        f"--seed={seed}",
        f"--class-order-name={args.class_order_name}",
        "--n-patches-per-slide=30",
        f"--train-name={config.get('train_split_name', 'train')}",
        f"--validation-name={config.get('validation_split_name', 'validation')}",
        f"--test-name={config.get('test_split_name', 'test')}",
        f"--overflow-strategy={config.get('overflow_strategy', 'redistribute')}",
    ]
    pool_size = config.get("full_scale_pool_size")
    if pool_size is not None:
        cmd.append(f"--pool-size={pool_size}")
    split_path = _seed_path(config.get("split_assignment_csv", ""), seed)
    if split_path:
        cmd.append(f"--split-assignment-path={split_path}")
    if args.class_order_file is not None:
        cmd.append(f"--class-order-file={args.class_order_file}")
    feature_dir = config.get("feature_path", "")
    if feature_dir:
        cmd.append(f"--feature-dir={feature_dir}")
    return cmd


def _imbalanced_csv(imbalanced_dir: str, parameter: float) -> str:
    stem = f"TCGA-UT_imbalanced_parameter={parameter}_dataset_size=500_seed=0"
    return f"{imbalanced_dir}/{stem}/imbalanced_dataset.csv"


def _seed_path(path: str, seed: int) -> str:
    return path.format(seed=seed) if path else path
