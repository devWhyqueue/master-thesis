import ast
import os
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd


def patch_sort_key(item: str) -> tuple[int, int]:
    """Sort patch identifiers by region and patch index."""
    return int(item.split("_")[0]), int(item.split("_")[1])


def get_dataset_structure(
    path: str,
    slides_to_exclude: Sequence[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Read the nested class/slide/patch dataset structure."""
    excluded = set(slides_to_exclude or [])
    return {
        class_name: _class_slides(path, class_name, excluded)
        for class_name in os.listdir(path)
        if not class_name.startswith(".")
    }


def sample_balanced_from_dataset_structure(
    df: pd.DataFrame,
    n_slides: int,
    n_patches: int | None = None,
    seed: int = 0,
    store_original_class_sizes: bool = True,
) -> pd.DataFrame:
    """Sample a balanced slide-level dataset structure."""
    df = _parse_patch_lists(df)
    original_sizes = df["cancer_type"].value_counts()
    filtered = _filter_available_slides(df, n_slides, n_patches)
    generator = np.random.default_rng(seed)
    sampled = filtered.groupby("cancer_type", group_keys=False).sample(
        n=n_slides,
        random_state=generator,
    )
    sampled = cast(pd.DataFrame, sampled.copy())
    sampled["patch_ids"] = sampled["patch_ids"].apply(
        lambda patches: _select_patches(patches, n_patches),
    )
    if store_original_class_sizes:
        sampled["original_class_size"] = sampled["cancer_type"].replace(original_sizes)
    return sampled


def _class_slides(
    path: str, class_name: str, excluded: set[str]
) -> dict[str, list[str]]:
    class_path = os.path.join(path, class_name)
    return {
        slide_id: _patches_for_slide(class_path, slide_id)
        for slide_id in os.listdir(class_path)
        if not slide_id.startswith(".") and slide_id not in excluded
    }


def _patches_for_slide(class_path: str, slide_id: str) -> list[str]:
    slide_path = os.path.join(class_path, slide_id)
    return [
        patch_name
        for patch_name in sorted(os.listdir(slide_path), key=patch_sort_key)
        if patch_name.endswith(".jpg")
    ]


def _parse_patch_lists(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df["patch_ids"].iloc[0], str):
        return df
    parsed = df.copy()
    parsed["patch_ids"] = parsed["patch_ids"].apply(ast.literal_eval)
    return parsed


def _filter_available_slides(
    df: pd.DataFrame,
    n_slides: int,
    n_patches: int | None,
) -> pd.DataFrame:
    filtered = df if n_patches is None else df[df["patch_ids"].apply(len) >= n_patches]
    filtered = filtered[
        filtered.groupby("cancer_type")["cancer_type"].transform("size") >= n_slides
    ]
    if n_patches is not None:
        assert len(filtered) != 0, f"Not enough slides with {n_patches} patches."
    return cast(pd.DataFrame, filtered)


def _select_patches(patches: Sequence[str], n_patches: int | None) -> list[str]:
    sorted_patches = sorted(patches, key=patch_sort_key)
    return sorted_patches if n_patches is None else sorted_patches[:n_patches]
