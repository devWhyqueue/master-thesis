from __future__ import annotations

import argparse

import pandas as pd

from common_code.progan import ProGanSettings


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for constructed ProGAN augmentation."""
    parser = argparse.ArgumentParser()
    _add_io_args(parser)
    _add_progan_args(parser)
    _add_feature_args(parser)
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> ProGanSettings:
    """Build shared ProGAN settings from CLI arguments."""
    return ProGanSettings(
        image_size=int(args.image_size),
        latent_dim=int(args.latent_dim),
        epochs_per_depth=int(args.epochs_per_depth),
        learning_rate=float(args.learning_rate),
        beta1=float(args.beta1),
        max_real_patches_per_class=int(args.max_real_patches_per_class),
        balance_target="max_train_class_count",
        max_classes=int(args.max_classes) if args.max_classes is not None else None,
        fade_in_fraction=float(args.fade_in_fraction),
        base_channels=int(args.base_channels),
        final_depth_epoch_grid=tuple(int(value) for value in args.final_depth_epochs),
    )


def validate_manifest(manifest: pd.DataFrame, path: str) -> None:
    """Require the row-level columns needed for ProGAN cache building."""
    required = {
        "split",
        "slide_id",
        "cancer_type",
        "patch_id",
        "feature_path",
        "feature_index",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest {path} missing required columns: {missing}")


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--manifest-save-path", required=True)
    parser.add_argument("--file-save-path", required=True)
    parser.add_argument("--synthetic-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--raw-resolution", default="0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--class-shard",
        default=None,
        metavar="I/N",
        help="Process only the I-th round-robin shard of tail classes (0-indexed).",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train and generate images only; skip feature extraction and manifest writing.",
    )


def _add_progan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--epochs-per-depth", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--max-real-patches-per-class", type=int, default=2048)
    parser.add_argument("--fade-in-fraction", type=float, default=0.5)
    parser.add_argument("--base-channels", type=int, default=256)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument(
        "--final-depth-epochs", nargs="+", type=int, default=[10, 25, 50]
    )


def _add_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--feature-model-name", default="hf-hub:paige-ai/Virchow2")
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument(
        "--feature-dtype", choices=["float16", "float32"], default="float16"
    )
