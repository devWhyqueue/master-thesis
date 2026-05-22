from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

from scripts.report.progan_diagnostics_metrics import (
    load_summary,
    pretty_class_name,
    summary_path,
)
from scripts.staging.io import resolve_raw_image_path


def _feature_cache_frame(cache_dir: Path) -> tuple[pd.DataFrame, np.memmap]:
    manifest = pd.read_csv(cache_dir / "manifest.csv", low_memory=False)
    features = np.load(cache_dir / "features.npy", mmap_mode="r")
    return manifest, features


def _class_feature_groups(
    manifest: pd.DataFrame, features: np.memmap, class_name: str
) -> tuple[np.ndarray, np.ndarray]:
    frame = cast(pd.DataFrame, manifest[manifest["cancer_type"] == class_name])
    real = frame[~frame["is_synthetic"].astype(bool)]
    synthetic = frame[frame["is_synthetic"].astype(bool)]
    real_idx = np.asarray(real["feature_index"], dtype=np.int64)
    synthetic_idx = np.asarray(synthetic["feature_index"], dtype=np.int64)
    real_features = np.asarray(features[real_idx], dtype=np.float32)
    synthetic_features = np.asarray(features[synthetic_idx], dtype=np.float32)
    return real_features, synthetic_features


def _nearest_real_index(real_features: np.ndarray, synthetic_vector: np.ndarray) -> int:
    distances = np.linalg.norm(real_features - synthetic_vector, axis=1)
    return int(distances.argmin())


def _synthetic_image_root(
    paths: dict[str, Path], seed: int, summary: dict[str, Any]
) -> Path:
    candidates = [
        paths["root"] / "outputs" / "synthetic_patch_images" / f"seed={seed}",
        paths["root"] / "synthetic_patch_images" / f"seed={seed}",
        Path(str(summary["image_root"])),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _resolve_image_path(path: Path, fallback_dir: Path | None = None) -> Path:
    if path.exists():
        return path
    if fallback_dir is not None:
        candidate = fallback_dir / path.name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found: {path}")


def _canonical_real_path(row: pd.Series, raw_root: Path) -> Path:
    filename = Path(str(row["image_path"])).name
    return (
        raw_root
        / str(row["cancer_type"])
        / str(row["resolution"])
        / str(row["slide_id"])
        / filename
    )


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
) -> str:
    real_frame = cast(
        pd.DataFrame,
        manifest[
            (manifest["cancer_type"] == class_name)
            & (~manifest["is_synthetic"].astype(bool))
        ],
    )
    synthetic_frame = cast(
        pd.DataFrame,
        manifest[
            (manifest["cancer_type"] == class_name)
            & (manifest["is_synthetic"].astype(bool))
        ],
    )
    real_features, _ = _class_feature_groups(manifest, features, class_name)
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
        "ProGAN synthetic patches and nearest real training neighbors (seed "
        f"{seed}, Virchow2 embedding space)",
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
            (left + right) / 2,
            bottom - 0.012,
            label,
            ha="center",
            va="top",
            fontsize=8,
        )


def _example_axes(
    example_classes: list[str], examples_per_class: int
) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(
        len(example_classes),
        examples_per_class * 2,
        figsize=(examples_per_class * 2.05, len(example_classes) * 1.95),
    )
    if len(example_classes) == 1:
        axes = np.expand_dims(axes, axis=0)
    return fig, axes


def plot_examples(
    paths: dict[str, Path],
    config: dict[str, Any],
    seed: int,
    cache_dir: Path,
    frame: pd.DataFrame,
    example_classes: list[str],
    examples_per_class: int,
    output_path: Path,
) -> None:
    """Plot synthetic patches beside nearest real training neighbors."""
    manifest, features = _feature_cache_frame(cache_dir)
    summary = load_summary(summary_path(paths, seed))
    image_root = _synthetic_image_root(paths, seed, summary)
    raw_root = Path(str(config["paths"]["raw_root"]))

    fig, axes = _example_axes(example_classes, examples_per_class)

    row_labels = []
    for row_idx, class_name in enumerate(example_classes):
        row_labels.append(
            _plot_class_examples(
                axes,
                row_idx,
                class_name,
                manifest,
                features,
                frame,
                image_root,
                raw_root,
                examples_per_class,
            )
        )

    _save_example_figure(fig, axes, row_labels, seed, output_path)
