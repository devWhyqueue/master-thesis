"""Build constructed ProGAN synthetic manifests and row-level feature caches."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

import pandas as pd
import torch

from common_code.progan import (
    ProGanSettings,
    collect_rows,
    load_diagnostics,
    train_class_progan,
    write_combined_manifest,
    write_generated_images,
    write_variant_manifest,
)
from tcga_ut_imbalanced.progan.config import (
    parse_args,
    settings_from_args,
    validate_manifest,
)
from tcga_ut_imbalanced.progan.features import build_feature_payload, real_only_payload
from tcga_ut_imbalanced.progan.manifest import (
    class_image_paths,
    combined_manifest,
    synthetic_manifest_rows,
    tail_classes,
    train_rows,
    with_real_metadata,
)
from tcga_ut_imbalanced.progan.reporting import write_class_diagnostics, write_summary

logger = logging.getLogger(__name__)


def main() -> None:
    """Generate synthetic patches and a combined ProGAN feature cache."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_progan_cache(parse_args())


def build_progan_cache(args: argparse.Namespace) -> None:
    """Build the augmented manifest and cache for one constructed regime."""
    manifest = pd.read_csv(args.manifest_path)
    validate_manifest(manifest, args.manifest_path)
    settings = settings_from_args(args)
    combined = _combined_manifest(args, manifest, settings)
    combined.to_csv(args.manifest_save_path, index=False)
    payload = _feature_payload(args, combined)
    torch.save(payload, args.file_save_path)
    write_summary(args, manifest, combined, settings)


def _combined_manifest(
    args, manifest: pd.DataFrame, settings: ProGanSettings
) -> pd.DataFrame:
    classes = tail_classes(manifest, settings.max_classes)
    if not classes:
        logger.info(
            "No tail classes require ProGAN augmentation: %s", args.manifest_path
        )
        return with_real_metadata(manifest)
    generated = generate_synthetic_rows(
        manifest,
        settings,
        Path(args.synthetic_root),
        Path(args.raw_root),
        str(args.raw_resolution),
        args.seed,
        torch.device(args.device),
    )
    return combined_manifest(manifest, generated, args.file_save_path)


def _feature_payload(args, combined: pd.DataFrame) -> dict[str, object]:
    has_synthetic = bool(combined["is_synthetic"].astype(bool).to_numpy().any())
    if not has_synthetic:
        return real_only_payload(combined)
    return build_feature_payload(
        combined,
        model_name=str(args.feature_model_name),
        batch_size=int(args.feature_batch_size),
        dtype=str(args.feature_dtype),
        device=torch.device(args.device),
    )


def generate_synthetic_rows(
    manifest: pd.DataFrame,
    settings: ProGanSettings,
    synthetic_root: Path,
    raw_root: Path,
    raw_resolution: str,
    seed: int,
    device: torch.device,
) -> dict[int, pd.DataFrame]:
    """Train class-specific ProGANs and return per-variant synthetic rows."""
    train = train_rows(manifest)
    target = int(train["cancer_type"].value_counts().max())
    rows = _variant_row_buckets(settings)
    seed_root = synthetic_root / f"seed={seed}"
    _populate_variant_rows(
        rows,
        manifest,
        train,
        target,
        settings,
        seed_root,
        raw_root,
        raw_resolution,
        seed,
        device,
    )
    _write_variant_manifests(seed_root, settings, seed)
    return {
        variant: pd.DataFrame(variant_rows)
        for variant, variant_rows in rows.items()
        if variant_rows
    }


def _populate_variant_rows(
    rows: dict[int, list[dict[str, object]]],
    manifest: pd.DataFrame,
    train: pd.DataFrame,
    target: int,
    settings: ProGanSettings,
    seed_root: Path,
    raw_root: Path,
    raw_resolution: str,
    seed: int,
    device: torch.device,
) -> None:
    for class_name in tail_classes(manifest, settings.max_classes):
        class_rows = cast(
            pd.DataFrame, train.loc[train["cancer_type"] == class_name].copy()
        )
        _generate_class_rows(
            rows,
            class_rows,
            class_name,
            target,
            settings,
            seed_root,
            raw_root,
            raw_resolution,
            seed,
            device,
        )


def _generate_class_rows(
    rows: dict[int, list[dict[str, object]]],
    class_rows: pd.DataFrame,
    class_name: str,
    target: int,
    settings: ProGanSettings,
    seed_root: Path,
    raw_root: Path,
    raw_resolution: str,
    seed: int,
    device: torch.device,
) -> None:
    image_paths = class_image_paths(
        class_rows, raw_root, raw_resolution, settings.max_real_patches_per_class, seed
    )
    snapshots, training = train_class_progan(image_paths, settings, device, seed)
    n_generate = target - len(class_rows)
    for variant, generator in snapshots.items():
        variant_dir = seed_root / f"epochs={variant}"
        generated = write_generated_images(
            generator,
            variant_dir / class_name,
            class_name,
            settings,
            device,
            n_generate,
        )
        rows[variant].extend(synthetic_manifest_rows(generated, variant))
        write_class_diagnostics(
            variant_dir, class_name, variant, n_generate, len(class_rows), training
        )


def _write_variant_manifests(
    seed_root: Path, settings: ProGanSettings, seed: int
) -> None:
    for variant in settings.final_depth_epoch_grid:
        variant_dir = seed_root / f"epochs={variant}"
        write_variant_manifest(
            variant_dir,
            collect_rows(variant_dir),
            load_diagnostics(variant_dir),
            seed,
            variant,
            settings,
        )
    write_combined_manifest(seed_root, settings)


def _variant_row_buckets(
    settings: ProGanSettings,
) -> dict[int, list[dict[str, object]]]:
    return {variant: [] for variant in settings.final_depth_epoch_grid}


if __name__ == "__main__":
    main()
