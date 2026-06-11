import argparse

from job_defs import Job, parameters, prefix


WSI_METHODS = (
    "mil_ce",
    "mil_weighted_ce",
    "mil_balanced_sampler_ce",
    "mil_focal",
    "rankmix_mil",
    "sc_mil",
    "mde_mil",
)


def train_wsi(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build constructed WSI-bag training jobs."""
    methods = WSI_METHODS if args.method == "all" else (args.method,)
    seeds = [0, 1, 2] if args.sweep else [args.seed]
    return [
        Job(
            _wsi_cmd(args, config, method, parameter, seed),
            "wsi_train",
            "logs/wsi/wsi%j.out",
            partition=config.get("wsi_partition", "gpu-2h"),
            gpus_per_node=1,
        )
        for method in methods
        for parameter in parameters(args)
        for seed in seeds
    ]


def _wsi_cmd(
    args: argparse.Namespace,
    config: dict[str, str],
    method: str,
    parameter: float,
    seed: int,
) -> list[str]:
    stem = _constructed_stem(args, config, parameter, seed)
    out = (
        f"{config.get('results_dir', '')}/results_{method}/"
        f"order={args.class_order_name}/param={parameter}/seed={seed}"
    )
    return (
        prefix(config, args)
        + [
            "-m",
            "tcga_ut_imbalanced.training.constructed_wsi",
            f"--manifest-path={stem}/manifest_splits.csv",
            f"--results-save-path={out}",
            f"--method={method}",
            f"--seed={seed}",
            f"--class-order-name={args.class_order_name}",
            f"--parameter={parameter}",
            "--device=auto",
            f"--epochs={args.epochs}",
            "--bag-batch-size=32",
            "--max-instances-per-bag=30",
            f"--bag-cache-dir={stem}/wsi_bag_cache",
        ]
        + _smoke_args(args)
        + _method_args(method)
    )


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


def _method_args(method: str) -> list[str]:
    selected = {
        "mil_weighted_ce": ["--weight-power=0.125"],
        "mil_focal": ["--focal-gamma=1.0"],
        "mil_balanced_sampler_ce": ["--sampler-power=0.75"],
        "rankmix_mil": ["--rankmix-alpha=32.0"],
        "sc_mil": ["--sc-mil-temperature=0.1"],
        "mde_mil": ["--mde-mil-consistency-weight=2.0"],
    }
    return selected.get(method, [])


def _smoke_args(args: argparse.Namespace) -> list[str]:
    if args.max_bags_per_class <= 0:
        return []
    return [f"--max-bags-per-class={args.max_bags_per_class}"]
