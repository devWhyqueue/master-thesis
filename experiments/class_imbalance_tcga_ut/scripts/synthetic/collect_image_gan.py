from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from scripts.common import ensure_dirs, load_config, write_json, write_progress
from scripts.synthetic.image_gan_data import (
    GanSettings,
    gan_settings,
    load_train_frame,
    tail_classes,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for image-GAN manifest collection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def _class_manifest(output_root: Path, class_name: str) -> Path:
    return output_root / class_name / "synthetic_image_manifest.csv"


def _read_class_manifests(
    output_root: Path, selected_classes: list[str]
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for class_name in selected_classes:
        manifest = _class_manifest(output_root, class_name)
        if manifest.exists():
            frames.append(pd.read_csv(manifest))
        else:
            missing.append(str(manifest))
    if missing:
        raise FileNotFoundError("Missing class GAN manifests: " + ", ".join(missing))
    return frames


def _write_outputs(
    output_root: Path,
    seed: int,
    selected_classes: list[str],
    rows: pd.DataFrame,
    settings: GanSettings,
) -> None:
    manifest_path = output_root / "synthetic_image_manifest.csv"
    rows.to_csv(manifest_path, index=False)
    write_json(
        output_root / "synthetic_image_gan_config.json",
        {
            "seed": seed,
            "settings": settings.__dict__,
            "n_generated": int(len(rows)),
            "classes": selected_classes,
            "execution": "seed-class-array",
            "note": (
                "Generated patch images are image-level augmentation artifacts. "
                "They require a feature-extraction pass before use in frozen "
                "Virchow2 WSI feature-bag experiments."
            ),
        },
    )
    write_progress(
        output_root / "progress.json",
        {
            "seed": seed,
            "status": "completed",
            "n_generated": int(len(rows)),
            "manifest": str(manifest_path),
        },
    )


def main() -> None:
    """Collect per-class image-GAN manifests into one seed-level manifest."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    settings = gan_settings(config, smoke=False)
    train_frame = load_train_frame(paths, args.seed, config)
    selected_classes = tail_classes(train_frame, settings)
    output_root = paths["root"] / "outputs" / "synthetic_images" / f"seed={args.seed}"
    frames = _read_class_manifests(output_root, selected_classes)
    rows = pd.concat(frames, ignore_index=True)
    _write_outputs(output_root, args.seed, selected_classes, rows, settings)
    logger.info("Collected %s generated images for seed %s", len(rows), args.seed)


if __name__ == "__main__":
    main()
