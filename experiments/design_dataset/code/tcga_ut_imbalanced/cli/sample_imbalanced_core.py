import argparse
import ast
import json
import logging
import os
from typing import cast

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from tcga_ut_imbalanced.data.sampling import (
    patch_sort_key,
    sample_balanced_from_dataset_structure,
)
from tcga_ut_imbalanced.plotting.plots import number_of_slides_per_class_bar

logger = logging.getLogger(__name__)


def sample_imbalanced(args: argparse.Namespace) -> None:
    """Create and store an imbalanced dataset split."""
    df = _load_balanced_dataset(args)
    out_path = _output_path(args)
    os.makedirs(out_path, exist_ok=True)
    df, df_balanced = _maybe_sample_validation(args, df, out_path)
    sampled = _sample_imbalanced_slides(args, df)
    _save_outputs(args, sampled, df_balanced, out_path)
    logger.info(
        "Stored imbalanced TCGA-UT dataset in %s.",
        os.path.join(out_path, "imbalanced_dataset.csv"),
    )


def make_index_array(
    groups: int, patches: int, total_patches_per_group: int
) -> np.ndarray:
    """Return flat patch indices for leading regions."""
    return (
        np.arange(groups)[:, None] * total_patches_per_group + np.arange(patches)
    ).ravel()


def _load_balanced_dataset(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.balanced_dataset_path)
    df = _sort_classes(args, df)
    if isinstance(df["patch_ids"].iloc[0], str):
        df["patch_ids"] = df["patch_ids"].apply(ast.literal_eval)
    return df


def _sort_classes(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    if args.class_order_file is None:
        return df.sort_values(
            ["original_class_size", "cancer_type"], ascending=[False, True]
        )
    with open(args.class_order_file) as file:
        class_order = json.load(file)
    logger.info("class order: %s", class_order)
    order_index = {value: index for index, value in enumerate(class_order)}
    return df.sort_values(["cancer_type"], key=lambda series: series.map(order_index))


def _maybe_sample_validation(
    args: argparse.Namespace,
    df: pd.DataFrame,
    out_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not args.sample_balanced_validation:
        return df, None
    balanced = sample_balanced_from_dataset_structure(
        df,
        args.n_slides_per_class,
        n_patches=None,
        seed=args.seed,
        store_original_class_sizes=False,
    )
    balanced.to_csv(_validation_path(args, out_path), index=False)
    remaining = df.loc[~df["slide_id"].isin(balanced["slide_id"].tolist())].copy()
    return remaining, balanced


def _sample_imbalanced_slides(
    args: argparse.Namespace, df: pd.DataFrame
) -> pd.DataFrame:
    targets = _target_counts(args, df)
    sampled = _floor_sample(args, df, targets)
    sampled = _fill_fractional_remainders(args, df, sampled, targets)
    sampled = _select_patch_regions(args, sampled)
    return sampled.sort_values(
        ["original_class_size", "cancer_type"], ascending=[False, True]
    )


def _target_counts(args: argparse.Namespace, df: pd.DataFrame) -> np.ndarray:
    n_classes = len(df["cancer_type"].unique())
    max_size = (
        len(df[df["cancer_type"] == df["cancer_type"][0]]) - args.n_slides_per_class
    )
    targets = _power_law_targets(args.parameter, args.dataset_size, n_classes)
    if args.overflow_strategy == "none":
        assert targets[0] <= max_size, "Not enough samples for the largest class."
        return targets
    return _redistribute_overflow(args.parameter, args.dataset_size, targets, max_size)


def _power_law_targets(
    parameter: float, dataset_size: int, n_classes: int
) -> np.ndarray:
    weights = np.array([np.power(index + 1, -parameter) for index in range(n_classes)])
    return weights / np.sum(weights) * dataset_size


def _redistribute_overflow(
    parameter: float,
    dataset_size: int,
    targets: np.ndarray,
    max_size: int,
) -> np.ndarray:
    capped = targets > max_size
    targets[capped] = max_size
    remaining = dataset_size - targets[capped].sum()
    while True:
        targets[~capped] = _power_law_targets(
            parameter, int(remaining), int((~capped).sum())
        )
        newly_capped = targets > max_size
        capped[newly_capped] = True
        targets[capped] = max_size
        remaining = remaining - targets[newly_capped].sum()
        if newly_capped.sum() == 0:
            return targets


def _floor_sample(
    args: argparse.Namespace, df: pd.DataFrame, targets: np.ndarray
) -> pd.DataFrame:
    floors = dict(zip(df["cancer_type"].unique(), np.floor(targets)))
    generator = np.random.default_rng(args.seed)
    sampled = df.groupby("cancer_type", group_keys=False, sort=False).apply(
        lambda group: group.sample(n=int(floors[group.name]), random_state=generator),
    )
    return cast(pd.DataFrame, sampled)


def _fill_fractional_remainders(
    args: argparse.Namespace,
    df: pd.DataFrame,
    sampled: pd.DataFrame,
    targets: np.ndarray,
) -> pd.DataFrame:
    remainders = dict(zip(df["cancer_type"].unique(), targets - np.floor(targets)))
    remaining = df.loc[~df["slide_id"].isin(sampled["slide_id"])].copy()
    generator = np.random.default_rng(args.seed)
    for index, class_name in enumerate(_largest_remainders(remainders)):
        if index >= args.dataset_size - len(sampled):
            break
        class_remaining = remaining.loc[remaining["cancer_type"] == class_name]
        sampled = pd.concat(
            [
                sampled,
                class_remaining.sample(random_state=generator),
            ],
            ignore_index=True,
        )
    return sampled


def _largest_remainders(remainders: dict[str, float]) -> list[str]:
    return [
        class_name
        for class_name, _ in sorted(
            remainders.items(), key=lambda item: item[1], reverse=True
        )
    ]


def _select_patch_regions(
    args: argparse.Namespace, sampled: pd.DataFrame
) -> pd.DataFrame:
    requested = args.n_regions_per_slide * args.n_patches_per_region
    assert np.all(sampled["patch_ids"].apply(len) >= requested), (
        "Not enough patches in original dataset."
    )
    sampled["patch_ids"] = sampled["patch_ids"].apply(
        lambda patches: sorted(patches, key=patch_sort_key)
    )
    indices = make_index_array(args.n_regions_per_slide, args.n_patches_per_region, 10)
    sampled["patch_ids"] = sampled["patch_ids"].apply(
        lambda patches: np.array(patches)[indices].tolist()
    )
    return sampled


def _save_outputs(
    args: argparse.Namespace,
    sampled: pd.DataFrame,
    balanced: pd.DataFrame | None,
    out_path: str,
) -> None:
    sampled.to_csv(os.path.join(out_path, "imbalanced_dataset.csv"), index=False)
    _save_json(os.path.join(out_path, "args.json"), vars(args))
    if args.store_class_names:
        _save_json(
            os.path.join(out_path, "class_names.json"),
            sampled["cancer_type"].unique()[::-1].tolist(),
        )
    if args.visualize:
        _save_visualizations(sampled, balanced, out_path)


def _save_visualizations(
    sampled: pd.DataFrame, balanced: pd.DataFrame | None, out_path: str
) -> None:
    viz_path = os.path.join(out_path, "visualizations")
    os.makedirs(viz_path, exist_ok=True)
    _save_bar(
        sampled, os.path.join(viz_path, "number_of_slides_per_class_bar_imbalanced.png")
    )
    if balanced is not None:
        _save_bar(
            balanced,
            os.path.join(
                viz_path, "number_of_slides_per_class_bar_balanced_validation.png"
            ),
        )


def _save_bar(df: pd.DataFrame, path: str) -> None:
    figure, _ = number_of_slides_per_class_bar(df)
    cast(Figure, figure).savefig(path, dpi=300, bbox_inches="tight")


def _output_path(args: argparse.Namespace) -> str:
    name = f"TCGA-UT_imbalanced_parameter={args.parameter}_dataset_size={args.dataset_size}_seed={args.seed}"
    return os.path.join(args.file_save_path, name)


def _validation_path(args: argparse.Namespace, out_path: str) -> str:
    name = f"balanced_validation_n_slides_per_class={args.n_slides_per_class}_seed={args.seed}.csv"
    return os.path.join(out_path, name)


def _save_json(path: str, data: object) -> None:
    with open(path, "w") as file:
        json.dump(data, file)
