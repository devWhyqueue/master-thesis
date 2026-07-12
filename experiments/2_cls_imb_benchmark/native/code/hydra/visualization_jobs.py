import argparse

from job_defs import Job, prefix

_STANDARD_METHOD = "patch_feature_balanced_sampler_ce"

_POINT_PLOT_SPECS: tuple[tuple[str, float, str], ...] = (
    ("patch_feature_weighted_ce", 1.0, "Weighted Cross Entropy"),
    ("patch_feature_balanced_sampler_ce", 1.0, "Batch Balancing"),
    ("patch_feature_focal", 1.0, "Weighted Focal Loss"),
    ("patch_feature_ce", 1.0, "Cross Entropy"),
    ("patch_feature_ce", 0.0, "Balanced"),
)


def visualize(args: argparse.Namespace, config: dict[str, str]) -> list[Job]:
    """Build visualization jobs."""
    cmd = (
        _standard_cmd(args, config)
        if args.type == "standard"
        else _point_plot_cmd(args, config)
    )
    return [Job(cmd, "viz", "logs/viz/viz%j.out")]


def _class_order(args: argparse.Namespace) -> str:
    return getattr(args, "class_order_name", "native_prevalence")


def _method_root(results_dir: str, method: str, class_order: str) -> str:
    return f"{results_dir}/results_{method}/order={class_order}"


def _standard_cmd(args: argparse.Namespace, config: dict[str, str]) -> list[str]:
    res_dir = config.get("results_dir", "")
    class_order = _class_order(args)
    base = _method_root(res_dir, _STANDARD_METHOD, class_order)
    return prefix(config, args) + [
        "-m",
        "cli.visualize",
        "--plot-types",
        "scatter_accuracies_of_two_parameters",
        "difference_confusion_matrix",
        "confusion_matrix",
        "--results-paths",
        base,
        base,
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
    class_order = _class_order(args)
    paths = [
        _method_root(res_dir, method, class_order) for method, _, _ in _POINT_PLOT_SPECS
    ]
    parameters = [str(param) for _, param, _ in _POINT_PLOT_SPECS]
    methods = [label for _, _, label in _POINT_PLOT_SPECS]
    return prefix(config, args) + [
        "-m",
        "cli.visualize",
        "--plot-types",
        "point_plot_compare_methods",
        "--results-paths",
        *paths,
        "--parameters",
        *parameters,
        "--parameter-name",
        "param",
        "--methods",
        *methods,
        "--visualization-save-path",
        f"{config.get('visualization_dir', '')}/",
    ]
