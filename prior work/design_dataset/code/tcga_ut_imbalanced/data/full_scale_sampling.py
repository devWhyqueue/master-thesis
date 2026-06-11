import ast
import argparse
import json
import os
from typing import cast

import numpy as np
import pandas as pd

from tcga_ut_imbalanced.data.full_scale_rows import (
    cap_rows_per_slide,
    rows_for_slides,
    slide_frame,
)
from tcga_ut_imbalanced.data.full_scale_targets import power_law_counts, target_counts
from tcga_ut_imbalanced.data.sampling import patch_sort_key

__all__ = ["power_law_counts"]


def load_manifest(path: str) -> pd.DataFrame:
    """Load a slide manifest with parsed patch identifiers."""
    frame = pd.read_csv(path)
    if "patch_ids" in frame.columns and isinstance(frame["patch_ids"].iloc[0], str):
        frame["patch_ids"] = frame["patch_ids"].apply(ast.literal_eval)
    return frame


def attach_splits(
    manifest: pd.DataFrame,
    split_path: str | None,
    split_column: str,
) -> pd.DataFrame:
    """Return a manifest containing one split column."""
    if split_column in manifest.columns:
        return manifest
    if split_path is None:
        raise ValueError("A split assignment path is required.")
    splits = pd.read_csv(split_path)
    required = {"slide_id", split_column}
    if not required.issubset(splits.columns):
        raise ValueError(f"Split assignments require columns {sorted(required)}.")
    return cast(pd.DataFrame, manifest.merge(splits, on="slide_id", how="inner"))


def class_order(
    manifest: pd.DataFrame,
    order_file: str | None,
) -> list[str]:
    """Return the requested class order from frequent to rare."""
    if order_file is not None:
        with open(order_file) as file:
            return list(json.load(file))
    counts = cast(pd.Series, manifest.groupby("cancer_type")["slide_id"].nunique())
    ordered = counts.sort_values(ascending=False).index.tolist()
    return [str(class_name) for class_name in ordered]


def construct_training_split(
    train_frame: pd.DataFrame,
    ordered_classes: list[str],
    parameter: float,
    seed: int,
    overflow_strategy: str = "redistribute",
) -> pd.DataFrame:
    """Sample a full-size constructed training split."""
    slides = slide_frame(train_frame)
    available = cast(pd.Series, slides.groupby("cancer_type")["slide_id"].nunique())
    targets = target_counts(
        available, ordered_classes, parameter, len(slides), overflow_strategy
    )
    rng = np.random.default_rng(seed)
    parts = [
        _sample_class(
            slides,
            class_name,
            targets[class_name],
            rng,
            replace=overflow_strategy == "replacement",
        )
        for class_name in ordered_classes
    ]
    sampled_slides = cast(pd.DataFrame, pd.concat(parts, ignore_index=True))
    return rows_for_slides(train_frame, sampled_slides["slide_id"].tolist())


def cap_patches(frame: pd.DataFrame, n_patches: int) -> pd.DataFrame:
    """Keep a deterministic per-slide patch budget."""
    if "patch_ids" not in frame.columns:
        return cap_rows_per_slide(frame, n_patches)
    capped = frame.copy()
    capped["patch_ids"] = capped["patch_ids"].apply(
        lambda patches: sorted(patches, key=patch_sort_key)[:n_patches]
    )
    return capped


def write_constructed_outputs(
    frame_by_split: dict[str, pd.DataFrame],
    targets: dict[str, int],
    ordered_classes: list[str],
    output_dir: str,
    metadata: dict[str, object],
) -> None:
    """Write constructed manifests and provenance files."""
    os.makedirs(output_dir, exist_ok=True)
    for split, frame in frame_by_split.items():
        frame.to_csv(os.path.join(output_dir, f"{split}.csv"), index=False)
    _combined_manifest(frame_by_split).to_csv(
        os.path.join(output_dir, "manifest_splits.csv"), index=False
    )
    _write_json(os.path.join(output_dir, "class_order.json"), ordered_classes)
    _write_json(os.path.join(output_dir, "target_counts.json"), targets)
    _write_json(os.path.join(output_dir, "args.json"), metadata)


def split_frames(
    args: argparse.Namespace,
    manifest: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return train, validation, and test frames by configured split names."""
    names = [args.train_name, args.validation_name, args.test_name]
    frames = cast(
        dict[str, pd.DataFrame],
        {name: manifest[manifest[args.split_column] == name].copy() for name in names},
    )
    assert_case_disjoint(frames)
    return frames


def assert_case_disjoint(frame_by_split: dict[str, pd.DataFrame]) -> None:
    """Raise if a TCGA case appears in more than one split."""
    if not all("case_id" in frame.columns for frame in frame_by_split.values()):
        return
    cases_by_split = {
        split: set(frame["case_id"].dropna().astype(str))
        for split, frame in frame_by_split.items()
    }
    split_names = list(cases_by_split)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            overlap = cases_by_split[first] & cases_by_split[second]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise ValueError(
                    f"Case leakage between {first} and {second}: {examples}"
                )


def constructed_payload(
    args: argparse.Namespace,
    splits: dict[str, pd.DataFrame],
    ordered_classes: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Return capped split frames and target class counts."""
    train = construct_training_split(
        splits[args.train_name],
        ordered_classes,
        args.parameter,
        args.seed,
        args.overflow_strategy,
    )
    frames = {
        "train": cap_patches(train, args.n_patches_per_slide),
        "validation": cap_patches(
            splits[args.validation_name], args.n_patches_per_slide
        ),
        "test": cap_patches(splits[args.test_name], args.n_patches_per_slide),
    }
    targets = target_counts(
        _available_training_slides(args, splits),
        ordered_classes,
        args.parameter,
        int(splits[args.train_name]["slide_id"].nunique()),
        args.overflow_strategy,
    )
    return frames, targets


def output_dir_for_args(args: argparse.Namespace) -> str:
    """Return the output directory for one constructed split."""
    name = (
        f"constructed_order={args.class_order_name}_"
        f"parameter={args.parameter}_seed={args.seed}"
    )
    return os.path.join(args.file_save_path, name)


def _combined_manifest(frame_by_split: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for split, frame in frame_by_split.items():
        tagged = frame.copy()
        tagged["split"] = split
        frames.append(tagged)
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))


def _sample_class(
    frame: pd.DataFrame,
    class_name: str,
    count: int,
    rng: np.random.Generator,
    replace: bool = False,
) -> pd.DataFrame:
    class_frame = frame[frame["cancer_type"] == class_name]
    return cast(
        pd.DataFrame,
        class_frame.sample(
            n=count, replace=replace and count > len(class_frame), random_state=rng
        ),
    )


def _available_training_slides(
    args: argparse.Namespace,
    splits: dict[str, pd.DataFrame],
) -> pd.Series:
    return cast(
        pd.Series,
        splits[args.train_name].groupby("cancer_type")["slide_id"].nunique(),
    )


def _write_json(path: str, data: object) -> None:
    with open(path, "w") as file:
        json.dump(data, file, indent=2)
