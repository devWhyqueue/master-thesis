from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

import pandas as pd

from scripts.common import ensure_dirs, load_config, write_json
from scripts.progan.fid import fid_for_paths
from scripts.progan.core import ProGanSettings
from scripts.progan.train import train_class_progan, write_generated_images
from scripts.training.support import _resolve_device

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse GAN-generation CLI arguments for the patch benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def _settings(config: dict, smoke: bool = False) -> ProGanSettings:
    raw = dict(config["patch_synthetic_progan"])
    if smoke:
        raw["image_size"] = 8
        raw["epochs_per_depth"] = 1
        raw["max_real_patches_per_class"] = min(
            int(raw["max_real_patches_per_class"]), 16
        )
        raw["max_classes"] = 1
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
    )


def _tail_classes(train_frame: pd.DataFrame, settings: ProGanSettings) -> list[str]:
    counts = cast(pd.Series, train_frame["cancer_type"].value_counts().sort_values())
    selected_counts = cast(
        pd.Series, counts.loc[counts < _balance_target(train_frame, settings)]
    )
    selected = [str(name) for name in selected_counts.index.tolist()]
    return (
        selected
        if settings.max_classes is None
        else selected[: int(settings.max_classes)]
    )


def _generated_rows(
    train_frame: pd.DataFrame,
    settings: ProGanSettings,
    output_root: Path,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Generate minority-class images and return manifest rows."""
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    device = _resolve_device("auto")
    target = _balance_target(train_frame, settings)
    for class_name in _tail_classes(train_frame, settings):
        image_paths = _class_image_paths(train_frame, class_name, settings)
        n_real = int((train_frame["cancer_type"] == class_name).sum())
        n_generate = max(0, target - n_real)
        generator = train_class_progan(image_paths, settings, device, seed)
        generated = write_generated_images(
            generator,
            output_root / class_name,
            class_name,
            settings,
            device,
            n_generate,
        )
        rows.extend(generated)
        diagnostics.append(
            {
                "class_name": class_name,
                "real_train_patches": n_real,
                "generated_patches": n_generate,
                "balance_target": target,
                "fid": fid_for_paths(
                    image_paths,
                    [Path(str(row["image_path"])) for row in generated],
                    device,
                ),
            }
        )
    return rows, diagnostics


def _class_image_paths(
    train_frame: pd.DataFrame, class_name: str, settings: ProGanSettings
) -> list[Path]:
    """Return capped real-image paths for one class."""
    values = train_frame.loc[train_frame["cancer_type"] == class_name, "image_path"]
    return [Path(path) for path in values.tolist()][
        : settings.max_real_patches_per_class
    ]


def _write_manifest(
    output_root: Path,
    rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    seed: int,
) -> Path:
    """Write synthetic patch rows and summary files."""
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        manifest = pd.DataFrame(
            {"cancer_type": pd.Series(dtype=str), "image_path": pd.Series(dtype=str)}
        )
    manifest["split"] = "train"
    manifest["slide_id"] = manifest["image_path"].map(lambda path: Path(str(path)).stem)
    manifest["resolution"] = "synthetic"
    path = output_root / "synthetic_patch_manifest.csv"
    manifest.to_csv(path, index=False)
    write_json(
        output_root / "synthetic_patch_summary.json",
        {
            "seed": seed,
            "n_patches": int(len(manifest)),
            "counts_by_class": manifest["cancer_type"].value_counts().to_dict(),
            "per_class": diagnostics,
        },
    )
    return path


def generate_patch_gan_manifest(config: dict, seed: int, smoke: bool = False) -> Path:
    """Train class-specific GANs and write a synthetic patch manifest."""
    paths = ensure_dirs(config)
    frame = pd.read_csv(paths["data"] / f"patch_manifest_seed={seed}.csv")
    train_frame = cast(pd.DataFrame, frame[frame["split"] == "train"])
    settings = _settings(config, smoke)
    output_root = paths["root"] / "outputs" / "synthetic_patch_images" / f"seed={seed}"
    existing = _existing_generated_rows(output_root)
    if existing:
        diagnostics = _existing_diagnostics(existing, train_frame, settings)
        return _write_manifest(output_root, existing, diagnostics, seed)
    rows, diagnostics = _generated_rows(train_frame, settings, output_root, seed)
    return _write_manifest(output_root, rows, diagnostics, seed)


def _existing_generated_rows(output_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not output_root.exists():
        return rows
    for path in sorted(output_root.glob("*/*.jpg")):
        rows.append({"cancer_type": path.parent.name, "image_path": str(path)})
    return rows


def _existing_diagnostics(
    rows: list[dict[str, object]], train_frame: pd.DataFrame, settings: ProGanSettings
) -> list[dict[str, object]]:
    counts = pd.DataFrame(rows)["cancer_type"].value_counts()
    target = _balance_target(train_frame, settings)
    diagnostics: list[dict[str, object]] = []
    for class_name, generated in sorted(counts.to_dict().items()):
        n_real = int((train_frame["cancer_type"] == class_name).sum())
        diagnostics.append(
            {
                "class_name": str(class_name),
                "real_train_patches": n_real,
                "generated_patches": int(generated),
                "balance_target": target,
                "fid": None,
            }
        )
    return diagnostics


def _balance_target(train_frame: pd.DataFrame, settings: ProGanSettings) -> int:
    """Resolve the post-augmentation patch count target."""
    counts = train_frame["cancer_type"].value_counts()
    if settings.balance_target == "max_train_class_count":
        return int(counts.max())
    raise ValueError(f"Unknown ProGAN balance target: {settings.balance_target}")


def main() -> None:
    """Generate patch-level GAN augmentations for one seed."""
    args = parse_args()
    config = load_config(args.config)
    path = generate_patch_gan_manifest(config, args.seed)
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
