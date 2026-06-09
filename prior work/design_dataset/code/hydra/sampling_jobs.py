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


def _imbalanced_csv(imbalanced_dir: str, parameter: float) -> str:
    stem = f"TCGA-UT_imbalanced_parameter={parameter}_dataset_size=500_seed=0"
    return f"{imbalanced_dir}/{stem}/imbalanced_dataset.csv"
