from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from scripts.training.split import _slice_split_rows

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class GanSettings:
    image_size: int
    latent_dim: int
    batch_size: int
    epochs: int
    learning_rate: float
    beta1: float
    max_real_patches_per_class: int
    generated_patches_per_class: int
    tail_class_quantile: float
    max_classes: int | None


def gan_settings(config: dict, smoke: bool) -> GanSettings:
    """Resolve image-GAN settings, shrinking them for smoke runs."""
    raw = dict(config.get("synthetic_image_gan", {}))
    max_classes = raw.get("max_classes")
    if smoke:
        raw["epochs"] = min(int(raw.get("epochs", 1)), 1)
        raw["max_real_patches_per_class"] = min(
            int(raw.get("max_real_patches_per_class", 16)), 16
        )
        raw["generated_patches_per_class"] = min(
            int(raw.get("generated_patches_per_class", 8)), 8
        )
        max_classes = min(int(max_classes or 1), 1)
    return GanSettings(
        image_size=int(raw.get("image_size", 64)),
        latent_dim=int(raw.get("latent_dim", 128)),
        batch_size=int(raw.get("batch_size", 64)),
        epochs=int(raw.get("epochs", 25)),
        learning_rate=float(raw.get("learning_rate", 0.0002)),
        beta1=float(raw.get("beta1", 0.5)),
        max_real_patches_per_class=int(raw.get("max_real_patches_per_class", 2048)),
        generated_patches_per_class=int(raw.get("generated_patches_per_class", 256)),
        tail_class_quantile=float(raw.get("tail_class_quantile", 0.5)),
        max_classes=max_classes if max_classes is None else int(max_classes),
    )


def load_train_frame(paths: dict[str, Path], seed: int, config: dict) -> pd.DataFrame:
    """Load the configured train split for one seed."""
    frame = pd.read_csv(paths["data"] / f"manifest_splits_seed={seed}.csv")
    max_train = config["training"].get("max_train_rows")
    max_eval = config["training"].get("max_eval_rows")
    sliced = _slice_split_rows(frame, max_train, max_eval)
    return cast(pd.DataFrame, sliced[sliced["split"] == "train"].copy())


def tail_classes(train_frame: pd.DataFrame, settings: GanSettings) -> list[str]:
    """Select low-support classes for synthetic image generation."""
    counts = cast(
        pd.Series,
        train_frame.groupby("cancer_type")["slide_id"].nunique(),
    ).sort_values()
    cutoff = counts.quantile(settings.tail_class_quantile)
    selected_counts = cast(pd.Series, counts.loc[counts <= cutoff])
    selected = [str(class_name) for class_name in selected_counts.index.tolist()]
    if settings.max_classes is not None:
        selected = selected[: settings.max_classes]
    return selected


def patch_paths_for_class(
    raw_root: Path,
    class_name: str,
    train_slides: set[str],
    limit: int,
    seed: int,
) -> list[Path]:
    """Collect patch image paths for one class, restricted to train slides."""
    class_root = raw_root / class_name
    paths = [
        path
        for split_root in sorted(class_root.iterdir())
        if split_root.is_dir()
        for slide_root in sorted(split_root.iterdir())
        if slide_root.is_dir() and slide_root.name in train_slides
        for path in sorted(slide_root.iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:limit]
