from analysis.plotting.cli_runner import main
from analysis.plotting.comparisons import (
    calculate_recalls_of_results,
    plot_scatter_accuracies_of_two_parameters,
    point_plot_compare_methods,
)
from analysis.plotting.matrices import (
    compute_average_confusion_matrix,
    number_of_slides_per_class_bar,
    plot_confusion_matrix,
    plot_difference_confusion_matrix,
    plot_extended_confusion_matrix,
)
from analysis.plotting.results import gather_results_across_seeds

__all__ = [
    "calculate_recalls_of_results",
    "compute_average_confusion_matrix",
    "gather_results_across_seeds",
    "main",
    "number_of_slides_per_class_bar",
    "plot_confusion_matrix",
    "plot_difference_confusion_matrix",
    "plot_extended_confusion_matrix",
    "plot_scatter_accuracies_of_two_parameters",
    "point_plot_compare_methods",
]
