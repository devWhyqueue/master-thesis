from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

from scripts.analysis.report.progan_diagnostics.metrics import (
    _REFERENCE_FINAL_DEPTH_EPOCHS,
    load_summary,
    pretty_class_name,
    summary_path,
)
from scripts.data.staging.io import resolve_raw_image_path


def _feature_cache_frame(cache_dir: Path) -> tuple[pd.DataFrame, np.memmap]:
    return pd.read_csv(cache_dir / "manifest.csv", low_memory=False), np.load(
        cache_dir / "features.npy", mmap_mode="r"
    )


def _class_feature_groups(
    manifest: pd.DataFrame,
    features: np.memmap,
    class_name: str,
    variant: int = _REFERENCE_FINAL_DEPTH_EPOCHS,
) -> tuple[np.ndarray, np.ndarray]:
    frame = cast(pd.DataFrame, manifest[manifest["cancer_type"] == class_name])
    real = frame[~frame["is_synthetic"].astype(bool)]
    syn_all = frame[frame["is_synthetic"].astype(bool)]
    syn = (
        cast(
            pd.DataFrame, syn_all[syn_all["final_depth_epochs"].astype(int) == variant]
        )
        if "final_depth_epochs" in syn_all.columns
        else syn_all
    )
    real_idx, syn_idx = (
        np.asarray(real["feature_index"], dtype=np.int64),
        np.asarray(syn["feature_index"], dtype=np.int64),
    )
    return np.asarray(features[real_idx], dtype=np.float32), np.asarray(
        features[syn_idx], dtype=np.float32
    )


def _nearest_real_index(real_features: np.ndarray, synthetic_vector: np.ndarray) -> int:
    return int(np.linalg.norm(real_features - synthetic_vector, axis=1).argmin())


def _synthetic_image_root(
    paths: dict, seed: int, summary: dict, var: int = _REFERENCE_FINAL_DEPTH_EPOCHS
) -> Path:
    root = paths["root"]
    candidates = [
        Path(str(summary["image_root"])),
        root / "synthetic_patch_images" / f"seed={seed}" / f"epochs={var}",
        root / "outputs" / "synthetic_patch_images" / f"seed={seed}",
        root / "synthetic_patch_images" / f"seed={seed}",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def _resolve_image_path(path: Path, fallback_dir: Path | None = None) -> Path:
    if path.exists():
        return path
    if fallback_dir and (fb := fallback_dir / path.name).exists():
        return fb
    raise FileNotFoundError(f"Image not found: {path}")


def _canonical_real_path(r: pd.Series, raw_root: Path) -> Path:
    p = raw_root / str(r["cancer_type"]) / str(r["resolution"]) / str(r["slide_id"])
    return p / Path(str(r["image_path"])).name


def _nearest_real_row(
    real_frame: pd.DataFrame, real_features: np.ndarray, synthetic_vector: np.ndarray
) -> pd.Series:
    sample_size = min(len(real_features), 512)
    if len(real_features) > sample_size:
        sample_idx = np.random.default_rng(0).choice(
            len(real_features), size=sample_size, replace=False
        )
        sample_features = real_features[sample_idx]
    else:
        sample_idx = np.arange(len(real_features))
        sample_features = real_features
    local_best = _nearest_real_index(sample_features, synthetic_vector)
    return cast(pd.Series, real_frame.iloc[int(sample_idx[local_best])])


def select_example_classes(frame: pd.DataFrame, configured: str) -> list[str]:
    """Choose representative tail classes for the example figure."""
    requested = [name.strip() for name in configured.split(",") if name.strip()]
    available = set(frame["class_name"].astype(str))
    selected = [name for name in requested if name in available]
    if selected:
        return selected
    return (
        frame.sort_values("generated_patches", ascending=False)["class_name"]
        .astype(str)
        .head(3)
        .tolist()
    )


def _plot_class_examples(
    axes: np.ndarray,
    row_idx: int,
    class_name: str,
    manifest: pd.DataFrame,
    features: np.memmap,
    frame: pd.DataFrame,
    image_root: Path,
    raw_root: Path,
    examples_per_class: int,
    variant: int = _REFERENCE_FINAL_DEPTH_EPOCHS,
) -> str:
    real_frame = cast(
        pd.DataFrame,
        manifest[
            (manifest["cancer_type"] == class_name)
            & (~manifest["is_synthetic"].astype(bool))
        ],
    )
    synthetic_all = cast(
        pd.DataFrame,
        manifest[
            (manifest["cancer_type"] == class_name)
            & (manifest["is_synthetic"].astype(bool))
        ],
    )
    if "final_depth_epochs" in synthetic_all.columns:
        synthetic_frame = cast(
            pd.DataFrame,
            synthetic_all[synthetic_all["final_depth_epochs"].astype(int) == variant],
        )
    else:
        synthetic_frame = synthetic_all
    real_features, _ = _class_feature_groups(
        manifest, features, class_name, variant=variant
    )
    for col_idx, (_, synthetic_row) in enumerate(
        synthetic_frame.head(examples_per_class).iterrows()
    ):
        synthetic_path = _resolve_image_path(
            Path(str(synthetic_row["image_path"])),
            image_root / class_name,
        )
        synthetic_vector = np.asarray(
            features[int(synthetic_row["feature_index"])], dtype=np.float32
        )
        real_row = _nearest_real_row(real_frame, real_features, synthetic_vector)
        real_path = resolve_raw_image_path(
            _canonical_real_path(real_row, raw_root), raw_root
        )
        synthetic_ax = axes[row_idx, col_idx * 2]
        real_ax = axes[row_idx, col_idx * 2 + 1]
        synthetic_ax.imshow(np.asarray(Image.open(synthetic_path).convert("RGB")))
        synthetic_ax.set_title("Synthetic", fontsize=8)
        real_ax.imshow(np.asarray(Image.open(real_path).convert("RGB")))
        real_ax.set_title("Nearest real", fontsize=8)
        for axis in (synthetic_ax, real_ax):
            axis.axis("off")
    class_metrics = frame.loc[frame["class_name"] == class_name].iloc[0]
    row_label = (
        f"{pretty_class_name(class_name)}; "
        f"FID={class_metrics['inception_fid']:.0f}; "
        f"Virchow NN={class_metrics['virchow_mean_nn_distance']:.2f}"
    )
    return row_label


def _save_example_figure(
    fig: Figure, axes: np.ndarray, row_labels: list[str], seed: int, output_path: Path
) -> None:
    fig.suptitle(
        f"ProGAN synthetic patches and nearest real training neighbors (seed {seed}, Virchow2 embedding space)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95), h_pad=2.0)
    _add_row_labels(fig, axes, row_labels)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _add_row_labels(fig: Figure, axes: np.ndarray, row_labels: list[str]) -> None:
    for row_idx, label in enumerate(row_labels):
        row_axes = axes[row_idx]
        left = min(axis.get_position().x0 for axis in row_axes)
        right = max(axis.get_position().x1 for axis in row_axes)
        bottom = min(axis.get_position().y0 for axis in row_axes)
        fig.text(
            (left + right) / 2, bottom - 0.012, label, ha="center", va="top", fontsize=8
        )


def _example_axes(
    classes: list[str], examples_per_class: int
) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(
        len(classes),
        examples_per_class * 2,
        figsize=(examples_per_class * 2.05, len(classes) * 1.95),
    )
    return fig, (np.expand_dims(axes, axis=0) if len(classes) == 1 else axes)


def plot_examples(
    paths: dict[str, Path],
    config: dict[str, Any],
    seed: int,
    cache_dir: Path,
    frame: pd.DataFrame,
    example_classes: list[str],
    examples_per_class: int,
    output_path: Path,
    variant: int = _REFERENCE_FINAL_DEPTH_EPOCHS,
) -> None:
    """Plot synthetic patches beside nearest real training neighbors."""
    manifest, features = _feature_cache_frame(cache_dir)
    summary = load_summary(summary_path(paths, seed, variant))
    image_root = _synthetic_image_root(paths, seed, summary, variant)
    raw_root = Path(str(config["paths"]["raw_root"]))
    fig, axes = _example_axes(example_classes, examples_per_class)
    row_labels = [
        _plot_class_examples(
            axes,
            idx,
            name,
            manifest,
            features,
            frame,
            image_root,
            raw_root,
            examples_per_class,
            variant=variant,
        )
        for idx, name in enumerate(example_classes)
    ]
    _save_example_figure(fig, axes, row_labels, seed, output_path)
