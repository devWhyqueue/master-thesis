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
    ds, val, test = _training_csvs(args, config, parameter, seed)
    out = _training_output(args, config, parameter, seed)
    return (
        prefix(config, args)
        + _train_base_with_test(config, ds, val, test, out)
        + _method_args(args.method)
        + [
            f"--training-method={args.method}",
            "--device=cpu",
            "--optimizer=adamw",
            "--learning-rate=0.001",
            "--weight-decay=0.0001",
            "--n-epochs=30",
            "--batch-size=256",
            "--dropout=0.1",
            f"--seed={seed}",
            "--visualize",
            f"--class-names-path={_class_names_path(args, config, parameter, seed)}",
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
            "--device=cpu",
            "--model=knn",
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
            "--device=cpu",
            "--model=ncc",
            "--visualize",
            f"--class-names-path={config.get('class_names_path', '')}",
        ]
    )


def _training_csvs(
    args: argparse.Namespace,
    config: dict[str, str],
    parameter: float,
    seed: int,
) -> tuple[str, str, str]:
    if not args.constructed:
        ds, val = train_csvs(config.get("imbalanced_dataset_dir", ""), parameter)
        return ds, val, ""
    stem = _constructed_stem(args, config, parameter, seed)
    return f"{stem}/train.csv", f"{stem}/validation.csv", f"{stem}/test.csv"


def _training_output(
    args: argparse.Namespace,
    config: dict[str, str],
    parameter: float,
    seed: int,
) -> str:
    method_root = f"{config.get('results_dir', '')}/results_{args.method}"
    if not args.constructed:
        return f"{method_root}/param={parameter}/seed={seed}"
    return f"{method_root}/order={args.class_order_name}/param={parameter}/seed={seed}"


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


def _class_names_path(
    args: argparse.Namespace,
    config: dict[str, str],
    parameter: float,
    seed: int,
) -> str:
    if not args.constructed:
        return config.get("class_names_path", "")
    return f"{_constructed_stem(args, config, parameter, seed)}/class_order.json"


def _method_args(method: str) -> list[str]:
    methods = {
        "ce": ["--loss=cross_entropy", "--alpha=uniform"],
        "weighted_ce": ["--loss=cross_entropy", "--alpha=inverse_class_frequency"],
        "balanced_sampler": [
            "--loss=cross_entropy",
            "--alpha=uniform",
            "--batch-balancing",
        ],
        "focal": ["--loss=focal_loss", "--alpha=uniform", "--gamma=2.0"],
        "ce_soft_f1": [
            "--loss=ce_soft_f1",
            "--alpha=uniform",
            "--batch-balancing",
        ],
        "ce_soft_mcc": [
            "--loss=ce_soft_mcc",
            "--alpha=uniform",
            "--batch-balancing",
        ],
        "cfal": [],
        "divide_conquer": [],
    }
    if method not in methods:
        raise ValueError(f"Unknown training method: {method}")
    return methods[method]


def _train_base_with_test(
    config: dict[str, str],
    dataset_path: str,
    validation_path: str,
    test_path: str,
    output_path: str,
) -> list[str]:
    args = train_base(config, dataset_path, validation_path, output_path)
    if test_path:
        args.append(f"--test-dataset-structure-path={test_path}")
    return args
