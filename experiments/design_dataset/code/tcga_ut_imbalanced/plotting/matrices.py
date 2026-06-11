import json
from collections.abc import Sequence
from typing import cast

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def number_of_slides_per_class_bar(df: pd.DataFrame) -> tuple[Figure, Axes]:
    """Plot the number of slides per class."""
    counts = df["cancer_type"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(counts.index.to_list(), counts.to_numpy())
    ax.set_ylabel("Number of Slides")
    ax.set_xlabel("Class")
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    return fig, ax


def plot_extended_confusion_matrix(
    cm: np.ndarray,
    recall: np.ndarray,
    precision: np.ndarray,
    class_names: Sequence[str] | None = None,
    normalize: bool = True,
    figsize: tuple[int, int] = (8, 8),
    cell_fontsize: float = 7,
    tick_fontsize: float = 9,
) -> tuple[Figure, Axes]:
    """Plot a confusion matrix with recall and precision margins."""
    matrix = _normalize(cm) if normalize else cm
    extended = _extended_matrix(matrix, recall, precision)
    labels = _labels(class_names, cm.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(extended)
    _configure_extended_axes(ax, labels, tick_fontsize)
    _annotate_matrix(ax, extended, cell_fontsize)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str] | None = None,
    normalize: bool = True,
    figsize: tuple[int, int] = (8, 8),
    cell_fontsize: float = 8,
    tick_fontsize: float = 10,
) -> tuple[Figure, Axes]:
    """Plot a standard confusion matrix."""
    matrix = _normalize(cm) if normalize else cm
    labels = _labels(class_names, cm.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(matrix)
    _configure_matrix_axes(ax, labels, tick_fontsize)
    _annotate_matrix(ax, matrix, cell_fontsize)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


def compute_average_confusion_matrix(
    results_paths: Sequence[str],
    class_order: np.ndarray | None = None,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute the average confusion matrix across result JSON files."""
    matrices = []
    class_names: list[str] = []
    resolved_order = class_order
    for results_path in results_paths:
        with open(results_path) as file:
            result = json.load(file)
        resolved_order = _class_order(resolved_order, result)
        class_names = cast(list[str], result["class_names"])
        matrix = confusion_matrix(
            cast(Sequence[int], result["labels"]),
            cast(Sequence[int], result["preds"]),
            labels=resolved_order,
        )
        matrices.append(_normalize(matrix) if normalize else matrix)
    if resolved_order is None:
        raise ValueError("At least one results path is required.")
    return np.mean(np.array(matrices), axis=0), resolved_order, class_names


def plot_difference_confusion_matrix(
    results_paths_1: Sequence[str],
    results_paths_2: Sequence[str],
    parameter_1: float | None = None,
    parameter_2: float | None = None,
    parameter_name: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot the difference between two average confusion matrices."""
    cm1, class_order, class_names = compute_average_confusion_matrix(
        results_paths_1,
        normalize=True,
    )
    cm2, _, _ = compute_average_confusion_matrix(results_paths_2, class_order, True)
    fig, ax = plot_confusion_matrix(cm1 - cm2, class_names, False, cell_fontsize=6.5)
    if parameter_1 is not None and parameter_2 is not None and parameter_name:
        ax.set_title(
            f"Difference between {parameter_name}={parameter_1} and {parameter_name}={parameter_2}"
        )
    return fig, ax


def _normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix.astype(float) / matrix.sum(axis=1, keepdims=True)


def _extended_matrix(
    cm: np.ndarray, recall: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    size = cm.shape[0]
    extended = np.zeros((size + 1, size + 1))
    extended[:size, :size] = cm
    extended[:size, -1] = recall
    extended[-1, :size] = precision
    return extended


def _labels(class_names: Sequence[str] | None, size: int) -> list[str]:
    return (
        list(class_names)
        if class_names is not None
        else [str(index) for index in range(size)]
    )


def _configure_extended_axes(ax: Axes, labels: list[str], tick_fontsize: float) -> None:
    extended_labels = labels + ["Recall"]
    ax.set_xticks(np.arange(len(extended_labels)))
    ax.set_yticks(np.arange(len(extended_labels)))
    ax.set_xticklabels(extended_labels)
    ax.set_yticklabels(labels + ["Precision"])
    _set_common_axis_labels(
        ax, tick_fontsize, "Confusion Matrix with Precision & Recall"
    )


def _configure_matrix_axes(ax: Axes, labels: list[str], tick_fontsize: float) -> None:
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=tick_fontsize)
    ax.set_yticklabels(labels, fontsize=tick_fontsize)
    _set_common_axis_labels(ax, tick_fontsize, "Confusion Matrix")


def _set_common_axis_labels(ax: Axes, tick_fontsize: float, title: str) -> None:
    ax.set_xlabel("Predicted label", fontsize=tick_fontsize)
    ax.set_ylabel("True label", fontsize=tick_fontsize)
    ax.set_title(title, fontsize=tick_fontsize + 2)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def _annotate_matrix(ax: Axes, matrix: np.ndarray, fontsize: float) -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not np.isnan(value):
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )


def _class_order(
    class_order: np.ndarray | None, result: dict[str, object]
) -> np.ndarray:
    return (
        np.array(cast(Sequence[int], result["class_order"]))
        if class_order is None
        else class_order
    )
