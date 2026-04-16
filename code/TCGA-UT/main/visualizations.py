import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()

    # plotting args
    parser.add_argument('--plot-types', choices=["number_of_slides_per_class_bar", "extended_confusion_matrix", "confusion_matrix", "difference_confusion_matrix", "scatter_accuracies_of_two_parameters", "point_plot_compare_methods"], nargs="+", default=None)

    # function args
    parser.add_argument('--results-paths', type=str, nargs="*", default=None)
    parser.add_argument('--dataset-path', type=str, default=None)
    parser.add_argument('--parameters', type=float, nargs="*", default=None)
    parser.add_argument('--parameter-name', type=str, default=None)
    parser.add_argument('--methods', type=str, nargs="*", default=None)

    # saving args
    parser.add_argument('--visualization-save-path', type=str, required=True)

    args = parser.parse_args()
    return args

def number_of_slides_per_class_bar(df):
    #n_slides_per_class = [(cls, len(val)) for cls, val in sorted(ds.items(), key=lambda item: len(item[1]))]
    n_slides_per_class = df["cancer_type"].value_counts()
    fig, ax = plt.subplots()
    #ax.bar([cls for cls, _ in n_slides_per_class], [n for _, n in n_slides_per_class])
    ax.bar(n_slides_per_class.index, n_slides_per_class.values)
    ax.set_ylabel("Number of Slides")
    ax.set_xlabel("Class")
    ax.set_xticklabels(n_slides_per_class.index, rotation=45, ha='right')

    return fig, ax

def plot_extended_confusion_matrix(
    cm,
    recall,
    precision,
    class_names=None,
    normalize=True,
    figsize=(8, 8),
    cell_fontsize=7,
    tick_fontsize=9
):
    """
    Plots confusion matrix with:
      - extra column: recall per class
      - extra row: precision per class
    """

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    num_classes = cm.shape[0]

    # Extended matrix
    extended_cm = np.zeros((num_classes + 1, num_classes + 1))
    extended_cm[:num_classes, :num_classes] = cm
    extended_cm[:num_classes, -1] = recall
    extended_cm[-1, :num_classes] = precision

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(extended_cm)

    # Ticks and labels
    tick_labels = class_names if class_names is not None else range(num_classes)

    ax.set_xticks(np.arange(num_classes + 1))
    ax.set_yticks(np.arange(num_classes + 1))

    ax.set_xticklabels(list(tick_labels) + ["Recall"])
    ax.set_yticklabels(list(tick_labels) + ["Precision"])

    ax.set_xlabel("Predicted label", fontsize=tick_fontsize)
    ax.set_ylabel("True label", fontsize=tick_fontsize)
    ax.set_title("Confusion Matrix with Precision & Recall")

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    for i in range(num_classes + 1):
        for j in range(num_classes + 1):
            value = extended_cm[i, j]
            if not np.isnan(value):
                ax.text(
                    j, i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize
                )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    return fig, ax

def plot_confusion_matrix(
    cm,
    class_names=None,
    normalize=True,
    figsize=(8, 8),
    cell_fontsize=8,
    tick_fontsize=10
):
    """
    Plots a (possibly normalized) confusion matrix.
    """

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    num_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm)

    # Tick labels
    if class_names is not None:
        tick_labels = class_names
    else:
        tick_labels = range(num_classes)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(tick_labels, fontsize=tick_fontsize)
    ax.set_yticklabels(tick_labels, fontsize=tick_fontsize)

    ax.set_xlabel("Predicted label", fontsize=tick_fontsize)
    ax.set_ylabel("True label", fontsize=tick_fontsize)
    ax.set_title("Confusion Matrix", fontsize=tick_fontsize + 2)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    for i in range(num_classes):
        for j in range(num_classes):
            value = cm[i, j]
            if not np.isnan(value):
                ax.text(
                    j, i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize
                )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    return fig, ax

def compute_average_confusion_matrix(results_paths, class_order=None, normalize=False):
    cms = []
    for results_path in results_paths:
        with open(results_path) as f:
            res = json.load(f)
        preds = np.array(res["preds"])
        labels = np.array(res["labels"])
        if class_order is None:
            class_order = np.array(res["class_order"])
        cm = confusion_matrix(labels, preds, labels=class_order)
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cms.append(cm)
    return np.mean(np.array(cms), axis=0), class_order, res["class_names"]

def plot_difference_confusion_matrix(results_paths_1, results_paths_2, parameter_1=None, parameter_2=None, parameter_name=None):
    
    cm1, class_order_1, class_names_1 = compute_average_confusion_matrix(results_paths_1, normalize=True)
    cm2, _, _ = compute_average_confusion_matrix(results_paths_2, class_order=class_order_1, normalize=True)
    
    cm_diff = cm1 - cm2
    fig, ax = plot_confusion_matrix(cm_diff, class_names=class_names_1, normalize=False, cell_fontsize=6.5)

    if parameter_1 is not None and parameter_2 is not None and parameter_name is not None:
        ax.set_title(f"Difference between {parameter_name}={parameter_1} and {parameter_name}={parameter_2}")

    return fig, ax

def calculate_recalls_of_results(results_paths, class_order=None, return_class_order=False):
    recalls = []
    for results_path in results_paths:
        with open(results_path) as f:
            res = json.load(f)
        preds = np.array(res["preds"])
        labels = np.array(res["labels"])
        if class_order is None:
            class_order = np.array(res["class_order"])
        _, recall, _, _ = precision_recall_fscore_support(labels, preds, average=None, labels=class_order)
        recalls.append(recall)
    if return_class_order:
        return np.array(recalls), class_order
    
    return np.array(recalls)

def plot_scatter_accuracies_of_two_parameters(results_paths_1, results_paths_2, parameter_1=None, parameter_2=None, parameter_name=None):
    
    recalls_1, class_order = calculate_recalls_of_results(results_paths_1, return_class_order=True)
    recalls_2 = calculate_recalls_of_results(results_paths_2, class_order=class_order)

    recalls_avg_1 = np.mean(recalls_1, axis=0)
    recalls_min_1 = np.min(recalls_1, axis=0)
    recalls_max_1 = np.max(recalls_1, axis=0)
    recalls_err_1 = np.vstack([np.abs(recalls_avg_1 - recalls_min_1), np.abs(recalls_max_1 - recalls_avg_1)]) # abs to prevent negative errors due to numerical instability

    recalls_avg_2 = np.mean(recalls_2, axis=0)
    recalls_min_2 = np.min(recalls_2, axis=0)
    recalls_max_2 = np.max(recalls_2, axis=0)
    recalls_err_2 = np.vstack([np.abs(recalls_avg_2 - recalls_min_2), np.abs(recalls_max_2 - recalls_avg_2)]) # abs to prevent negative errors due to numerical instability

    fig, ax = plt.subplots()
    ax.errorbar(recalls_avg_2, recalls_avg_1, xerr=recalls_err_2, yerr=recalls_err_1, fmt='none', ecolor='black', capsize=4)
    sc = ax.scatter(recalls_avg_2, recalls_avg_1, c=np.arange(len(class_order)), cmap="cool")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Number of datapoints")
    cbar.set_ticks([class_order.min(), class_order.max()])
    cbar.set_ticklabels(["Few", "Many"])
    ax.plot([0, 1], [0, 1], c="black")
    ax.set_title("Recall")

    if parameter_1 is not None and parameter_2 is not None and parameter_name is not None:
        ax.set_xlabel(f"{parameter_name}={parameter_2}")
        ax.set_ylabel(f"{parameter_name}={parameter_1}")

    return fig, ax

def point_plot_compare_methods(results_paths, methods=None):
    results = {"recall": [], "class_name": [], "class_label": [], "class_size_index": [], "method": [], "seed": []}
    for i, res_path in enumerate(results_paths):
        with open(res_path) as f:
            res_json = json.load(f)
        seed = int(res_path.split("seed=")[-1].split("/")[0])
        method = res_path.split("results_")[-1].split("/")[0]
        if methods is not None:
            method = methods[i]
        results["recall"].extend(res_json["recall_per_class"])
        results["class_name"].extend(res_json["class_names"])
        results["class_label"].extend(res_json["class_order"])
        results["class_size_index"].extend([res_json["class_names"].index(class_name) for class_name in res_json["class_names"]])
        results["method"].extend([method]*len(res_json["class_names"]))
        results["seed"].extend([seed]*len(res_json["class_names"]))
    df_res = pd.DataFrame(results)
    fig, ax = plt.subplots()
    grouped = df_res.groupby(["class_name", "method"], sort=False).mean()
    cmap = cm.cool
    norm = mcolors.Normalize(
        vmin=df_res["class_size_index"].min(),
        vmax=df_res["class_size_index"].max()
    )
    for key in grouped.index.levels[0]:  # A, B
        y = grouped.loc[key]["recall"]
        ax.plot(y.index, y.values, marker="o", label=key, c=cmap(norm(grouped.loc[key]["class_size_index"].astype("int")))[0])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for older matplotlib versions
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Number of datapoints")
    cbar.set_ticks([df_res["class_size_index"].min(), df_res["class_size_index"].max()])
    cbar.set_ticklabels(["Few", "Many"])
    ax.set_ylabel("Recall")

    return fig, ax

def gather_results_across_seeds(path, return_n_seeds=False):
    res_paths = []
    n_seeds = len(os.listdir(path))
    for seed_folder in os.listdir(path):
        if len(os.listdir(os.path.join(path, seed_folder))) == 1:
            # backward compatability with timestamps stores
            res_path = os.path.join(path, seed_folder, os.listdir(os.path.join(path, seed_folder))[0], "validation_results.json")
        else:
            res_path = os.path.join(path, seed_folder, "validation_results.json")
        res_paths.append(res_path)
    if return_n_seeds:
        return res_paths, n_seeds
    return res_paths

def main(): 
    args = get_args()

    for plot_type in args.plot_types:
        parameter_1 = None
        parameter_2 = None
        parameter_name = None
        if plot_type == "number_of_slides_per_class_bar":
            if args.args.dataset_path is None:
                raise argparse.ArgumentError("dataset_path is required for plotting 'number_of_slides_per_class_bar'.")
            with open(args.dataset_path) as f:
                ds = json.load(f)
            fig, _ = number_of_slides_per_class_bar(ds)
        elif plot_type == "extended_confusion_matrix":
            raise NotImplementedError("'plot_extended_confusion_matrix' not yet callable.")
        elif plot_type == "confusion_matrix":
            if args.results_paths is None:
                raise argparse.ArgumentError("You need to specify one results path in --results-paths for plotting 'confusion_matrix'.")
            if args.parameter_name is None:
                raise argparse.ArgumentError("Need to pass --parameter-name for 'confusion_matrix'.")
            if args.parameters is None:
                raise argparse.ArgumentError("Need to pass one parameter in --parameters for 'confusion_matrix'.")
            res_paths = gather_results_across_seeds(os.path.join(args.results_paths[0], f"{args.parameter_name}={args.parameters[0]}"))
            cm, _, class_names = compute_average_confusion_matrix(res_paths, normalize=True)
            parameter_1 = args.parameters[0]
            parameter_name = args.parameter_name
            fig, ax = plot_confusion_matrix(cm, class_names=class_names, normalize=False, cell_fontsize=6.5)

            if parameter_1 is not None and parameter_name is not None:
                ax.set_title(f"Confusion Matrix for {parameter_name}={parameter_1}")
        elif plot_type == "difference_confusion_matrix":
            if args.results_paths is None or len(args.results_paths) < 2:
                raise argparse.ArgumentError("You need to specify at least two results path in --results-paths for plotting 'difference_confusion_matrix'.")
            if args.parameter_name is None:
                raise argparse.ArgumentError("Need to pass --parameter-name for 'difference_confusion_matrix'.")
            if args.parameters is None or len(args.parameters) < 2:
                raise argparse.ArgumentError("Need to pass at least two parameters in --parameters for 'difference_confusion_matrix'.")
            parameter_1 = args.parameters[0]
            parameter_2 = args.parameters[1]
            parameter_name = args.parameter_name
            res_paths_1 = gather_results_across_seeds(os.path.join(args.results_paths[0], f"{parameter_name}={parameter_1}"))
            res_paths_2 = gather_results_across_seeds(os.path.join(args.results_paths[1], f"{parameter_name}={parameter_2}"))
            fig, _ = plot_difference_confusion_matrix(res_paths_1, res_paths_2, parameter_1=parameter_1, parameter_2=parameter_2, parameter_name=parameter_name)
        elif plot_type == "scatter_accuracies_of_two_parameters":
            if args.results_paths is None or len(args.results_paths) < 2:
                raise argparse.ArgumentError("You need to specify at least two results path in --results-paths for plotting 'scatter_accuracies_of_two_parameters'.")
            if args.parameter_name is None:
                raise argparse.ArgumentError("Need to pass --parameter-name for 'scatter_accuracies_of_two_parameters'.")
            if args.parameters is None or len(args.parameters) < 2:
                raise argparse.ArgumentError("Need to pass at least two parameters in --parameters for 'scatter_accuracies_of_two_parameters'.")
            parameter_1 = args.parameters[0]
            parameter_2 = args.parameters[1]
            parameter_name = args.parameter_name
            res_paths_1 = gather_results_across_seeds(os.path.join(args.results_paths[0], f"{parameter_name}={parameter_1}"))
            res_paths_2 = gather_results_across_seeds(os.path.join(args.results_paths[1], f"{parameter_name}={parameter_2}"))
            fig, _ = plot_scatter_accuracies_of_two_parameters(res_paths_1, res_paths_2, parameter_1=parameter_1, parameter_2=parameter_2, parameter_name=parameter_name)
        elif plot_type == "point_plot_compare_methods":
            if args.parameter_name is None:
                raise argparse.ArgumentError("Need to pass --parameter-name for 'point_plot_compare_methods'.")
            if args.parameters is None or len(args.parameters) < 1:
                raise argparse.ArgumentError("Need to pass one parameter in --parameters for 'point_plot_compare_methods'.")
            if args.results_paths is None:
                raise argparse.ArgumentError("Need to pass at least one results-path for 'point_plot_compare_methods'.")
            if args.methods is not None:
                if len(args.methods) != len(args.results_paths):
                    raise argparse.ArgumentError("If --methods are passed they need to be of the same length as --results-paths for 'point_plot_compare_methods'.")
            parameter_name = args.parameter_name
            res_paths = []
            methods = []
            for i, results_path in enumerate(args.results_paths):
                if len(args.parameters) == len(args.results_paths):
                    parameter = args.parameters[i]
                else:
                    parameter = args.parameters[0]
                paths, n_seeds = gather_results_across_seeds(os.path.join(results_path, f"{parameter_name}={parameter}"), return_n_seeds=True)
                res_paths.extend(paths)
                methods.extend([args.methods[i]]*n_seeds)
            fig, ax = point_plot_compare_methods(res_paths, methods=methods)

        os.makedirs(args.visualization_save_path, exist_ok=True)
        plot_title = f"{plot_type}"
        if parameter_name is not None and parameter_1 is not None:
            plot_title += f"_{parameter_name}_1={parameter_1}"
        if parameter_name is not None and parameter_2 is not None:
            plot_title += f"_{parameter_name}_2={parameter_2}"
        plot_title += ".png"
        fig.savefig(os.path.join(args.visualization_save_path, plot_title), dpi=300, bbox_inches="tight")
        print(f"Stored {plot_type} in {os.path.join(args.visualization_save_path, plot_title)}.", flush=True)

if __name__ == '__main__':
    main()
