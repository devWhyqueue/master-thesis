import json
from collections.abc import Sequence
from typing import cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def calculate_recalls_of_results(
    results_paths: Sequence[str],
    class_order: np.ndarray | None = None,
    return_class_order: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Calculate per-class recalls for result JSON files."""
    recalls = []
    for results_path in results_paths:
        with open(results_path) as file:
            result = json.load(file)
        class_order = (
            np.array(cast(Sequence[int], result["class_order"]))
            if class_order is None
            else class_order
        )
        recalls.append(_recall(result, class_order))
    if return_class_order:
        if class_order is None:
            raise ValueError("At least one results path is required.")
        return np.array(recalls), class_order
    return np.array(recalls)


def plot_scatter_accuracies_of_two_parameters(
    results_paths_1: Sequence[str],
    results_paths_2: Sequence[str],
    parameter_1: float | None = None,
    parameter_2: float | None = None,
    parameter_name: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot class recalls for two result groups against each other."""
    recalls_1, class_order = cast(
        tuple[np.ndarray, np.ndarray],
        calculate_recalls_of_results(results_paths_1, return_class_order=True),
    )
    recalls_2 = cast(
        np.ndarray,
        calculate_recalls_of_results(results_paths_2, class_order=class_order),
    )
    fig, ax = plt.subplots()
    _draw_recall_scatter(ax, recalls_1, recalls_2, class_order)
    if parameter_1 is not None and parameter_2 is not None and parameter_name:
        ax.set_xlabel(f"{parameter_name}={parameter_2}")
        ax.set_ylabel(f"{parameter_name}={parameter_1}")
    return fig, ax


def point_plot_compare_methods(
    results_paths: Sequence[str],
    methods: Sequence[str] | None = None,
) -> tuple[Figure, Axes]:
    """Plot per-class recalls grouped by method."""
    df_res = pd.DataFrame(_method_results(results_paths, methods))
    fig, ax = plt.subplots()
    size_index = np.asarray(df_res["class_size_index"], dtype=float)
    norm = mcolors.Normalize(float(size_index.min()), float(size_index.max()))
    cmap = plt.get_cmap("cool")
    for key, group in df_res.groupby("class_name", sort=False):
        grouped = cast(
            pd.DataFrame,
            group.groupby("method", sort=False).mean(numeric_only=True),
        )
        group_size_index = np.asarray(grouped["class_size_index"], dtype=int)
        color = cmap(norm(group_size_index))[0]
        ax.plot(
            [str(label) for label in grouped.index.to_list()],
            np.asarray(grouped["recall"], dtype=float),
            marker="o",
            label=str(key),
            c=color,
        )
    _add_size_colorbar(fig, ax, norm, df_res)
    ax.set_ylabel("Recall")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig, ax


def _recall(result: dict[str, object], class_order: np.ndarray) -> np.ndarray:
    _, recall, _, _ = precision_recall_fscore_support(
        cast(Sequence[int], result["labels"]),
        cast(Sequence[int], result["preds"]),
        average=None,
        labels=class_order,
    )
    return cast(np.ndarray, recall)


def _draw_recall_scatter(
    ax: Axes,
    recalls_1: np.ndarray,
    recalls_2: np.ndarray,
    class_order: np.ndarray,
) -> None:
    avg_1, err_1 = _mean_and_error(recalls_1)
    avg_2, err_2 = _mean_and_error(recalls_2)
    ax.errorbar(
        avg_2, avg_1, xerr=err_2, yerr=err_1, fmt="none", ecolor="black", capsize=4
    )
    scatter = ax.scatter(avg_2, avg_1, c=np.arange(len(class_order)), cmap="cool")
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label("Number of datapoints")
    colorbar.set_ticks([class_order.min(), class_order.max()])
    colorbar.set_ticklabels(["Few", "Many"])
    ax.plot([0, 1], [0, 1], c="black")
    ax.set_title("Recall")


def _mean_and_error(recalls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = cast(np.ndarray, np.mean(recalls, axis=0))
    lower = cast(np.ndarray, np.abs(mean - np.min(recalls, axis=0)))
    upper = cast(np.ndarray, np.abs(np.max(recalls, axis=0) - mean))
    return mean, np.vstack([lower, upper])


def _method_results(
    results_paths: Sequence[str], methods: Sequence[str] | None
) -> dict[str, list[object]]:
    results: dict[str, list[object]] = {
        key: []
        for key in [
            "recall",
            "class_name",
            "class_label",
            "class_size_index",
            "method",
            "seed",
        ]
    }
    for index, result_path in enumerate(results_paths):
        with open(result_path) as file:
            result = json.load(file)
        method = (
            methods[index]
            if methods is not None
            else result_path.split("results_")[-1].split("/")[0]
        )
        _extend_method_results(results, result, method, result_path)
    return results


def _extend_method_results(
    results: dict[str, list[object]],
    result: dict[str, object],
    method: str,
    result_path: str,
) -> None:
    class_names = cast(list[str], result["class_names"])
    results["recall"].extend(cast(Sequence[object], result["recall_per_class"]))
    results["class_name"].extend(class_names)
    results["class_label"].extend(cast(Sequence[object], result["class_order"]))
    results["class_size_index"].extend(
        [class_names.index(name) for name in class_names]
    )
    results["method"].extend([method] * len(class_names))
    results["seed"].extend(
        [int(result_path.split("seed=")[-1].split("/")[0])] * len(class_names)
    )


def _add_size_colorbar(
    fig: Figure, ax: Axes, norm: mcolors.Normalize, df_res: pd.DataFrame
) -> None:
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("cool"))
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=ax)
    colorbar.set_label("Number of datapoints")
    size_index = np.asarray(df_res["class_size_index"], dtype=float)
    colorbar.set_ticks(
        [
            float(size_index.min()),
            float(size_index.max()),
        ]
    )
    colorbar.set_ticklabels(["Few", "Many"])
