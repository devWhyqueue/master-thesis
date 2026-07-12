import argparse
import logging
import os
from collections.abc import Sequence
from typing import Literal, cast, overload

from matplotlib.figure import Figure
import pandas as pd

from analysis.plotting.support.comparisons import (
    plot_scatter_accuracies_of_two_parameters,
    point_plot_compare_methods,
)
from analysis.plotting.support.matrices import (
    compute_average_confusion_matrix,
    number_of_slides_per_class_bar,
    plot_confusion_matrix,
    plot_difference_confusion_matrix,
)

logger = logging.getLogger(__name__)


@overload
def gather_results_across_seeds(
    path: str, return_n_seeds: Literal[False] = False
) -> list[str]:
    """Collect validation result paths below seed folders."""
    ...


@overload
def gather_results_across_seeds(
    path: str, return_n_seeds: Literal[True]
) -> tuple[list[str], int]:
    """Collect validation result paths and the seed count."""
    ...


def gather_results_across_seeds(
    path: str, return_n_seeds: bool = False
) -> list[str] | tuple[list[str], int]:
    """Collect validation result paths below seed folders."""
    result_paths = [_result_path(path, seed_folder) for seed_folder in os.listdir(path)]
    if return_n_seeds:
        return result_paths, len(result_paths)
    return result_paths


def _result_path(path: str, seed_folder: str) -> str:
    seed_path = os.path.join(path, seed_folder)
    children = os.listdir(seed_path)
    if len(children) == 1:
        return os.path.join(seed_path, children[0], "validation_results.json")
    return os.path.join(seed_path, "validation_results.json")


Meta = dict[str, float | str | None]

PLOT_TYPES = [
    "number_of_slides_per_class_bar",
    "extended_confusion_matrix",
    "confusion_matrix",
    "difference_confusion_matrix",
    "scatter_accuracies_of_two_parameters",
    "point_plot_compare_methods",
]


def get_args() -> argparse.Namespace:
    """Parse visualization command arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-types", choices=PLOT_TYPES, nargs="+", default=None)
    parser.add_argument("--results-paths", type=str, nargs="*", default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--parameters", type=float, nargs="*", default=None)
    parser.add_argument("--parameter-name", type=str, default=None)
    parser.add_argument("--methods", type=str, nargs="*", default=None)
    parser.add_argument("--visualization-save-path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    """Run requested visualizations."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = get_args()
    for plot_type in args.plot_types:
        fig, meta = _build_plot(plot_type, args)
        _save_plot(fig, args.visualization_save_path, plot_type, meta)


def _build_plot(plot_type: str, args: argparse.Namespace) -> tuple[Figure, Meta]:
    if plot_type == "number_of_slides_per_class_bar":
        return _slide_bar(args)
    if plot_type == "confusion_matrix":
        return _confusion(args)
    if plot_type == "difference_confusion_matrix":
        return _difference(args)
    if plot_type == "scatter_accuracies_of_two_parameters":
        return _scatter(args)
    if plot_type == "point_plot_compare_methods":
        return _point_plot(args)
    raise NotImplementedError("'plot_extended_confusion_matrix' not yet callable.")


def _slide_bar(args: argparse.Namespace) -> tuple[Figure, Meta]:
    _require(args.dataset_path, "dataset_path is required.")
    fig, _ = number_of_slides_per_class_bar(pd.read_csv(args.dataset_path))
    return fig, {"parameter_name": None, "parameter_1": None, "parameter_2": None}


def _confusion(args: argparse.Namespace) -> tuple[Figure, Meta]:
    _require(args.results_paths, "Need one results path.")
    _require(args.parameter_name, "Need --parameter-name.")
    _require(args.parameters, "Need one --parameters value.")
    res_paths = _seed_paths(
        args.results_paths[0], args.parameter_name, args.parameters[0]
    )
    matrix, _, class_names = compute_average_confusion_matrix(res_paths, normalize=True)
    fig, ax = plot_confusion_matrix(
        matrix, class_names, normalize=False, cell_fontsize=6.5
    )
    ax.set_title(f"Confusion Matrix for {args.parameter_name}={args.parameters[0]}")
    return fig, _meta(args.parameter_name, args.parameters[0], None)


def _difference(args: argparse.Namespace) -> tuple[Figure, Meta]:
    _require_pair(args)
    paths_1 = _seed_paths(
        args.results_paths[0], args.parameter_name, args.parameters[0]
    )
    paths_2 = _seed_paths(
        args.results_paths[1], args.parameter_name, args.parameters[1]
    )
    fig, _ = plot_difference_confusion_matrix(
        paths_1, paths_2, args.parameters[0], args.parameters[1], args.parameter_name
    )
    return fig, _meta(args.parameter_name, args.parameters[0], args.parameters[1])


def _scatter(args: argparse.Namespace) -> tuple[Figure, Meta]:
    _require_pair(args)
    paths_1 = _seed_paths(
        args.results_paths[0], args.parameter_name, args.parameters[0]
    )
    paths_2 = _seed_paths(
        args.results_paths[1], args.parameter_name, args.parameters[1]
    )
    fig, _ = plot_scatter_accuracies_of_two_parameters(
        paths_1, paths_2, args.parameters[0], args.parameters[1], args.parameter_name
    )
    return fig, _meta(args.parameter_name, args.parameters[0], args.parameters[1])


def _point_plot(args: argparse.Namespace) -> tuple[Figure, Meta]:
    _require(args.results_paths, "Need --results-paths.")
    _require(args.parameter_name, "Need --parameter-name.")
    _require(args.parameters, "Need --parameters.")
    paths, methods = _point_paths(args)
    fig, _ = point_plot_compare_methods(paths, methods)
    return fig, _meta(args.parameter_name, None, None)


def _point_paths(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    paths: list[str] = []
    methods: list[str] = []
    result_paths = cast(Sequence[str], args.results_paths)
    parameters = cast(Sequence[float], args.parameters)
    method_names = cast(Sequence[str], args.methods)
    for index, results_path in enumerate(result_paths):
        parameter = (
            parameters[index] if len(parameters) == len(result_paths) else parameters[0]
        )
        seed_paths, n_seeds = gather_results_across_seeds(
            _parameter_path(results_path, args.parameter_name, parameter), True
        )
        paths.extend(seed_paths)
        methods.extend([method_names[index]] * n_seeds)
    return paths, methods


def _require_pair(args: argparse.Namespace) -> None:
    _require(
        args.results_paths and len(args.results_paths) >= 2, "Need two result paths."
    )
    _require(args.parameter_name, "Need --parameter-name.")
    _require(args.parameters and len(args.parameters) >= 2, "Need two parameters.")


def _require(condition: object, message: str) -> None:
    if not condition:
        raise argparse.ArgumentError(None, message)


def _seed_paths(base_path: str, parameter_name: str, parameter: float) -> list[str]:
    return gather_results_across_seeds(
        _parameter_path(base_path, parameter_name, parameter)
    )


def _parameter_path(base_path: str, parameter_name: str, parameter: float) -> str:
    return os.path.join(base_path, f"{parameter_name}={parameter}")


def _meta(
    parameter_name: str | None, parameter_1: float | None, parameter_2: float | None
) -> Meta:
    return {
        "parameter_name": parameter_name,
        "parameter_1": parameter_1,
        "parameter_2": parameter_2,
    }


def _save_plot(fig: Figure, save_path: str, plot_type: str, meta: Meta) -> None:
    os.makedirs(save_path, exist_ok=True)
    filename = _plot_filename(plot_type, meta)
    fig.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
    logger.info("Stored %s in %s.", plot_type, os.path.join(save_path, filename))


def _plot_filename(plot_type: str, meta: Meta) -> str:
    title = plot_type
    if meta["parameter_name"] is not None and meta["parameter_1"] is not None:
        title += f"_{meta['parameter_name']}_1={meta['parameter_1']}"
    if meta["parameter_name"] is not None and meta["parameter_2"] is not None:
        title += f"_{meta['parameter_name']}_2={meta['parameter_2']}"
    return f"{title}.png"
