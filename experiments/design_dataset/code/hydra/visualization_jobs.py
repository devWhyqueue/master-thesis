import argparse

from job_defs import Job, prefix


def visualize(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build visualization jobs."""
    cmd = (
        _standard_cmd(args, config)
        if args.type == "standard"
        else _point_plot_cmd(args, config)
    )
    return [Job(cmd, "viz", "logs/viz/viz%j.out")]


def _standard_cmd(args: argparse.Namespace, config: dict[str, str]) -> list[str]:
    res_dir = config.get("results_dir", "")
    return prefix(config, args) + [
        "-m",
        "tcga_ut_imbalanced.cli.visualize",
        "--plot-types",
        "scatter_accuracies_of_two_parameters",
        "difference_confusion_matrix",
        "confusion_matrix",
        "--results-paths",
        f"{res_dir}/results_batch_balancing/",
        f"{res_dir}/results_batch_balancing/",
        "--parameters",
        "1.0",
        "0.0",
        "--parameter-name",
        "param",
        "--visualization-save-path",
        f"{config.get('visualization_dir', '')}/",
    ]


def _point_plot_cmd(args: argparse.Namespace, config: dict[str, str]) -> list[str]:
    res_dir = config.get("results_dir", "")
    return prefix(config, args) + [
        "-m",
        "tcga_ut_imbalanced.cli.visualize",
        "--plot-types",
        "point_plot_compare_methods",
        "--results-paths",
        f"{res_dir}/results_cross_entropy_inverse_class_frequency",
        f"{res_dir}/results_batch_balancing/",
        f"{res_dir}/results_focal_loss_inverse_class_frequency/",
        f"{res_dir}/results_focal_loss_uniform/",
        f"{res_dir}/results_original_class_size_order/",
        f"{res_dir}/results_original_class_size_order/",
        "--parameters",
        "1.0",
        "1.0",
        "1.0",
        "1.0",
        "1.0",
        "0.0",
        "--parameter-name",
        "param",
        "--methods",
        "Weighted Cross Entropy",
        "Batch Balancing",
        "Weighted Focal Loss",
        "Unweighted Focal Loss",
        "Vanilla",
        "Balanced",
        "--visualization-save-path",
        f"{config.get('visualization_dir', '')}/",
    ]
