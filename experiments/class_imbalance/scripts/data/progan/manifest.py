from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from scripts.common import ensure_dirs
from scripts.data.progan.core import ProGanSettings
from scripts.data.progan.storage import (
    class_is_complete,
    collect_rows,
    diagnostics_path,
    fid_payload,
    generated_counts_match,
    load_class_diagnostics,
    load_diagnostics,
    save_class_diagnostics,
    synthetic_output_root,
    variant_output_root,
    write_combined_manifest,
    write_variant_manifest,
)
from scripts.data.progan.train import train_class_progan, write_generated_images
from scripts.data.staging.io import resolve_raw_image_path
from scripts.modeling.training.support import _resolve_device


_REFERENCE_FINAL_DEPTH_EPOCHS = 25
"""Default final-depth epoch count used by the image-level patch_progan_aug path."""


def progan_settings(config: dict, smoke: bool = False) -> ProGanSettings:
    """Build ProGAN settings from experiment config."""
    raw = dict(config["patch_synthetic_progan"])
    grid: tuple[int, ...] = tuple(
        int(v) for v in raw.get("final_depth_epoch_grid", (10, 25, 50))
    )
    if smoke:
        raw["image_size"] = 8
        raw["epochs_per_depth"] = 1
        raw["max_real_patches_per_class"] = min(
            int(raw["max_real_patches_per_class"]), 16
        )
        raw["max_classes"] = 1
        grid = (1, 2, 3)
    return ProGanSettings(
        image_size=int(raw["image_size"]),
        latent_dim=int(raw["latent_dim"]),
        epochs_per_depth=int(raw["epochs_per_depth"]),
        learning_rate=float(raw["learning_rate"]),
        beta1=float(raw["beta1"]),
        max_real_patches_per_class=int(raw["max_real_patches_per_class"]),
        balance_target=str(raw["balance_target"]),
        max_classes=raw.get("max_classes"),
        fade_in_fraction=float(raw["fade_in_fraction"]),
        base_channels=int(raw["base_channels"]),
        final_depth_epoch_grid=grid,
    )


def patch_seeds(config: dict, smoke: bool) -> list[int]:
    """Return patch benchmark seeds for the current run mode."""
    seeds = [int(seed) for seed in config["patch_training"]["seeds"]]
    return [seeds[0]] if smoke else seeds


def train_frame_for_seed(config: dict, seed: int) -> pd.DataFrame:
    """Load the training split of a patch manifest."""
    paths = ensure_dirs(config)
    frame = pd.read_csv(paths["data"] / f"patch_manifest_seed={seed}.csv")
    return cast(pd.DataFrame, frame[frame["split"] == "train"])


def balance_target(train_frame: pd.DataFrame, settings: ProGanSettings) -> int:
    """Resolve the post-augmentation patch count target."""
    counts = train_frame["cancer_type"].value_counts()
    if settings.balance_target == "max_train_class_count":
        return int(counts.max())
    raise ValueError(f"Unknown ProGAN balance target: {settings.balance_target}")


def tail_classes(train_frame: pd.DataFrame, settings: ProGanSettings) -> list[str]:
    """Return minority classes that require synthetic augmentation."""
    counts = cast(pd.Series, train_frame["cancer_type"].value_counts().sort_values())
    target = balance_target(train_frame, settings)
    selected = [str(name) for name in counts.loc[counts < target].index.tolist()]
    if settings.max_classes is None:
        return selected
    return selected[: int(settings.max_classes)]


def expected_generated_counts(
    train_frame: pd.DataFrame, settings: ProGanSettings
) -> dict[str, int]:
    """Return required synthetic patch counts per tail class."""
    target = balance_target(train_frame, settings)
    return {
        class_name: max(
            0, target - int((train_frame["cancer_type"] == class_name).sum())
        )
        for class_name in tail_classes(train_frame, settings)
    }


def output_root_for_seed(config: dict, seed: int) -> Path:
    """Return the seed-level synthetic output directory (parent of per-variant dirs)."""
    paths = ensure_dirs(config)
    return synthetic_output_root(paths["root"], seed)


def _variant_root_for_seed(config: dict, seed: int, variant: int) -> Path:
    """Return the per-variant synthetic output directory for one seed."""
    paths = ensure_dirs(config)
    return variant_output_root(paths["root"], seed, variant)


def progan_array_upper_bound(config: dict, smoke: bool = False) -> int:
    """Return the inclusive upper bound for the parallel ProGAN SLURM array."""
    settings = progan_settings(config, smoke)
    max_tail = max(
        len(tail_classes(train_frame_for_seed(config, seed), settings))
        for seed in patch_seeds(config, smoke)
    )
    return max_tail * len(patch_seeds(config, smoke)) - 1


def decode_progan_array_task(
    config: dict, task_id: int, smoke: bool = False
) -> tuple[int, str] | None:
    """Map a SLURM array task id to a seed and tail class name."""
    seeds = patch_seeds(config, smoke)
    seed = seeds[task_id % len(seeds)]
    class_idx = task_id // len(seeds)
    settings = progan_settings(config, smoke)
    classes = tail_classes(train_frame_for_seed(config, seed), settings)
    if class_idx >= len(classes):
        return None
    return seed, classes[class_idx]


def _progan_subsample_seed(benchmark_seed: int, class_name: str) -> int:
    """Return a stable RNG seed for subsampling real patches per benchmark seed and class."""
    digest = hashlib.sha256(f"{benchmark_seed}:{class_name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _class_image_paths(
    train_frame: pd.DataFrame,
    class_name: str,
    settings: ProGanSettings,
    raw_root: Path,
    benchmark_seed: int,
) -> list[Path]:
    values = train_frame.loc[train_frame["cancer_type"] == class_name, "image_path"]
    paths = [resolve_raw_image_path(Path(path), raw_root) for path in values.tolist()]
    limit = settings.max_real_patches_per_class
    if len(paths) <= limit:
        return paths
    rng = np.random.default_rng(_progan_subsample_seed(benchmark_seed, class_name))
    picked = rng.choice(len(paths), size=limit, replace=False)
    return [paths[int(index)] for index in picked]


def _train_and_write_class(
    train_frame: pd.DataFrame,
    settings: ProGanSettings,
    seed_root: Path,
    class_name: str,
    seed: int,
    raw_root: Path,
) -> dict[int, dict[str, object]]:
    """Train one class GAN, write one synthetic directory per grid variant.

    Returns a mapping from final_depth_epochs to per-class diagnostics.
    """
    image_paths = _class_image_paths(
        train_frame, class_name, settings, raw_root, benchmark_seed=seed
    )
    n_real = int((train_frame["cancer_type"] == class_name).sum())
    expected = expected_generated_counts(train_frame, settings)[class_name]
    device = _resolve_device("auto")
    snapshots, training = train_class_progan(image_paths, settings, device, seed)
    per_variant: dict[int, dict[str, object]] = {}
    for variant, generator in snapshots.items():
        variant_dir = seed_root / f"epochs={variant}"
        generated = write_generated_images(
            generator, variant_dir / class_name, class_name, settings, device, expected
        )
        diag: dict[str, object] = {
            "balance_target": balance_target(train_frame, settings),
            "class_name": class_name,
            "final_depth_epochs": variant,
            "fid": fid_payload(image_paths, generated, device),
            "generated_patches": expected,
            "real_train_patches": n_real,
            "training": training,
        }
        save_class_diagnostics(variant_dir, diag)
        per_variant[variant] = diag
    return per_variant


def _all_variants_complete(
    seed_root: Path, settings: ProGanSettings, expected: dict[str, int]
) -> bool:
    """Return True if every variant directory has the expected patch count for every class."""
    for variant in settings.final_depth_epoch_grid:
        variant_dir = seed_root / f"epochs={variant}"
        for class_name, count in expected.items():
            if not class_is_complete(variant_dir / class_name, count):
                return False
    return True


def generate_class_progan(
    config: dict, seed: int, class_name: str, smoke: bool = False
) -> dict[int, dict[str, object]]:
    """Train one class-specific ProGAN and write synthetic patches for each grid variant."""
    settings = progan_settings(config, smoke)
    train_frame = train_frame_for_seed(config, seed)
    seed_root = output_root_for_seed(config, seed)
    expected = expected_generated_counts(train_frame, settings).get(class_name, 0)
    # Check if all variant directories are already complete.
    all_done = all(
        class_is_complete(seed_root / f"epochs={v}" / class_name, expected)
        and diagnostics_path(seed_root / f"epochs={v}", class_name).exists()
        for v in settings.final_depth_epoch_grid
    )
    if all_done:
        return {
            v: load_class_diagnostics(diagnostics_path(seed_root / f"epochs={v}", class_name))
            for v in settings.final_depth_epoch_grid
        }
    # Remove any partial variant directories for this class before retraining.
    for v in settings.final_depth_epoch_grid:
        class_dir = seed_root / f"epochs={v}" / class_name
        if class_dir.exists():
            shutil.rmtree(class_dir)
    raw_root = Path(config["paths"]["raw_root"])
    return _train_and_write_class(
        train_frame, settings, seed_root, class_name, seed, raw_root
    )


def merge_patch_gan_manifest(config: dict, seed: int, smoke: bool = False) -> Path:
    """Merge per-class synthetic outputs into per-variant and combined manifests.

    Returns the path to the combined manifest (all variants, final_depth_epochs tagged)
    which is used by Virchow2 feature extraction.  A per-variant manifest is also written
    for the image-level patch_progan_aug path (reference variant).
    """
    settings = progan_settings(config, smoke)
    train_frame = train_frame_for_seed(config, seed)
    expected = expected_generated_counts(train_frame, settings)
    seed_root = output_root_for_seed(config, seed)

    for variant in settings.final_depth_epoch_grid:
        variant_dir = seed_root / f"epochs={variant}"
        missing = {
            class_name: count
            for class_name, count in expected.items()
            if not class_is_complete(variant_dir / class_name, count)
        }
        if missing:
            missing_text = ", ".join(f"{name}={count}" for name, count in missing.items())
            raise RuntimeError(
                f"ProGAN classes incomplete for seed={seed} epochs={variant}: {missing_text}"
            )
        rows = collect_rows(variant_dir)
        if not smoke and expected and not rows:
            raise RuntimeError(
                f"ProGAN augmentation produced no synthetic patches for seed={seed} epochs={variant}."
            )
        write_variant_manifest(
            variant_dir, rows, load_diagnostics(variant_dir), seed, variant, settings
        )

    return write_combined_manifest(seed_root, settings)


def merge_patch_gan_manifest_reference(
    config: dict, seed: int, smoke: bool = False
) -> Path:
    """Return the reference-variant manifest for the image-level patch_progan_aug path."""
    settings = progan_settings(config, smoke)
    ref = _REFERENCE_FINAL_DEPTH_EPOCHS
    if ref not in settings.final_depth_epoch_grid:
        ref = max(settings.final_depth_epoch_grid)
    seed_root = output_root_for_seed(config, seed)
    return seed_root / f"epochs={ref}" / "synthetic_patch_manifest.csv"


def generate_patch_gan_manifest(config: dict, seed: int, smoke: bool = False) -> Path:
    """Train all class-specific GANs sequentially and write the combined manifest."""
    settings = progan_settings(config, smoke)
    train_frame = train_frame_for_seed(config, seed)
    seed_root = output_root_for_seed(config, seed)
    expected = expected_generated_counts(train_frame, settings)

    if _all_variants_complete(seed_root, settings, expected):
        return merge_patch_gan_manifest(config, seed, smoke)

    for class_name in tail_classes(train_frame, settings):
        generate_class_progan(config, seed, class_name, smoke)
    return merge_patch_gan_manifest(config, seed, smoke)
