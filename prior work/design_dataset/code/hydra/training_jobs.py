import argparse

from job_defs import Job, parameters, prefix, train_base, train_csvs


def train(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build training jobs."""
    if args.model == "mlp":
        return _mlp_jobs(args, config)
    if args.model == "knn":
        return _knn_jobs(args, config)
    return _ncc_jobs(args, config)


def _mlp_jobs(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    return [
        Job(
            _mlp_cmd(args, config, parameter, seed),
            "train",
            "logs/training/train%j.out",
        )
        for parameter in parameters(args)
        for seed in ([0, 1, 2] if args.sweep else [args.seed])
    ]


def _mlp_cmd(
    args: argparse.Namespace,
    config: dict[str, str],
    parameter: float,
    seed: int,
) -> list[str]:
    ds, val = train_csvs(config.get("imbalanced_dataset_dir", ""), parameter)
    out = f"{config.get('results_dir', '')}/param={parameter}/seed={seed}"
    return (
        prefix(config, args)
        + train_base(config, ds, val, out)
        + [
            '--device="cpu"',
            "--learning-rate=0.001",
            "--n-epochs=50",
            '--loss="cross_entropy"',
            '--alpha="uniform"',
            "--batch-balancing",
            f"--seed={seed}",
            "--visualize",
            f"--class-names-path={config.get('class_names_path', '')}",
        ]
    )


def _knn_jobs(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    return [
        Job(
            _knn_cmd(args, config, neighbors),
            "train_knn",
            "logs/training/train_knn%j.out",
        )
        for neighbors in ([3, 9, 27] if args.sweep else [args.k])
    ]


def _knn_cmd(
    args: argparse.Namespace, config: dict[str, str], neighbors: int
) -> list[str]:
    bal_csv = config.get("balanced_dataset_csv", "")
    val_csv = config.get("validation_dataset_csv", bal_csv)
    out = f"{config.get('results_dir', '')}/results_knn/k={neighbors}/"
    return (
        prefix(config, args)
        + train_base(config, bal_csv, val_csv, out)
        + [
            '--device="cpu"',
            '--model="knn"',
            f"--k={neighbors}",
            "--visualize",
            f"--class-names-path={config.get('class_names_path', '')}",
        ]
    )


def _ncc_jobs(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    return [
        Job(
            _ncc_cmd(args, config, parameter),
            "train_ncc",
            "logs/training/train_ncc%j.out",
        )
        for parameter in parameters(args)
    ]


def _ncc_cmd(
    args: argparse.Namespace, config: dict[str, str], parameter: float
) -> list[str]:
    ds, val = train_csvs(config.get("imbalanced_dataset_dir", ""), parameter)
    out = f"{config.get('results_dir', '')}/results_ncc/param={parameter}/"
    return (
        prefix(config, args)
        + train_base(config, ds, val, out)
        + [
            '--device="cpu"',
            '--model="ncc"',
            "--visualize",
            f"--class-names-path={config.get('class_names_path', '')}",
        ]
    )
