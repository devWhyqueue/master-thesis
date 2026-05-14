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
    patch_paths_for_class,
    tail_classes,
)
from scripts.synthetic.image_gan_models import (
    train_class_gan,
    write_generated_images,
)
from scripts.training.support import _resolve_device

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for image GAN generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-index", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _generate_for_class(
    raw_root: Path,
    output_root: Path,
    train_frame: pd.DataFrame,
    class_name: str,
    settings: GanSettings,
    seed: int,
) -> list[dict[str, object]]:
    """Train and export one class-specific image GAN."""
    train_slides = set(
        train_frame.loc[train_frame["cancer_type"] == class_name, "slide_id"]
    )
    image_paths = patch_paths_for_class(
        raw_root,
        class_name,
        train_slides,
        settings.max_real_patches_per_class,
        seed,
    )
    if len(image_paths) < 2:
        logger.warning("Skipping %s with only %s patches", class_name, len(image_paths))
        return []
    device = _resolve_device("auto")
    generator = train_class_gan(image_paths, settings, device, seed)
    return write_generated_images(
        generator,
        output_root / class_name,
        class_name,
        settings,
        device,
    )


def _write_class_progress(
    output_root: Path, seed: int, class_name: str, class_idx: int, n_classes: int
) -> None:
    write_progress(
        output_root / "progress.json",
        {
            "seed": seed,
            "status": "running",
            "class": class_name,
            "class_index": class_idx,
            "n_classes": n_classes,
        },
    )


def _write_outputs(
    output_root: Path,
    seed: int,
    rows: list[dict[str, object]],
    selected_classes: list[str],
    settings: GanSettings,
) -> None:
    manifest = pd.DataFrame(rows)
    manifest_path = output_root / "synthetic_image_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    write_json(
        output_root / "synthetic_image_gan_config.json",
        {
            "seed": seed,
            "settings": settings.__dict__,
            "n_generated": len(rows),
            "classes": selected_classes,
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
            "n_generated": len(rows),
            "manifest": str(manifest_path),
        },
    )


def _write_class_outputs(
    output_root: Path,
    seed: int,
    rows: list[dict[str, object]],
    class_name: str,
    class_idx: int,
    n_classes: int,
) -> None:
    manifest_path = output_root / class_name / "synthetic_image_manifest.csv"
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        frame = pd.DataFrame({"cancer_type": [], "image_path": []})
    frame.to_csv(manifest_path, index=False)
    write_progress(
        output_root / f"progress_class_{class_idx:02d}.json",
        {
            "seed": seed,
            "status": "completed",
            "class": class_name,
            "class_index": class_idx + 1,
            "n_classes": n_classes,
            "n_generated": len(rows),
            "manifest": str(manifest_path),
        },
    )


def main() -> None:
    """Train per-tail-class image GANs and export synthetic patch images."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    settings = gan_settings(config, args.smoke)
    train_frame = load_train_frame(paths, args.seed, config)
    raw_root = Path(config["paths"]["raw_root"])
    output_root = paths["root"] / "outputs" / "synthetic_images" / f"seed={args.seed}"
    selected_classes = tail_classes(train_frame, settings)
    if args.class_index is not None:
        _run_single_class(
            raw_root, output_root, train_frame, selected_classes, settings, args
        )
        return
    _run_all_classes(
        raw_root, output_root, train_frame, selected_classes, settings, args
    )


def _run_single_class(
    raw_root: Path,
    output_root: Path,
    train_frame: pd.DataFrame,
    selected_classes: list[str],
    settings: GanSettings,
    args: argparse.Namespace,
) -> None:
    class_name = _indexed_class(selected_classes, args.class_index)
    _write_class_progress(
        output_root, args.seed, class_name, args.class_index + 1, len(selected_classes)
    )
    rows = _generate_for_class(
        raw_root, output_root, train_frame, class_name, settings, args.seed
    )
    _write_class_outputs(
        output_root,
        args.seed,
        rows,
        class_name,
        args.class_index,
        len(selected_classes),
    )
    logger.info("Wrote %s generated images for %s", len(rows), class_name)


def _run_all_classes(
    raw_root: Path,
    output_root: Path,
    train_frame: pd.DataFrame,
    selected_classes: list[str],
    settings: GanSettings,
    args: argparse.Namespace,
) -> None:
    rows = _generate_all_classes(
        raw_root, output_root, train_frame, selected_classes, settings, args.seed
    )
    _write_outputs(output_root, args.seed, rows, selected_classes, settings)
    logger.info("Wrote %s generated images to %s", len(rows), output_root)


def _indexed_class(selected_classes: list[str], class_index: int) -> str:
    if class_index < 0 or class_index >= len(selected_classes):
        raise ValueError(
            f"class-index {class_index} is outside 0..{len(selected_classes) - 1}"
        )
    return selected_classes[class_index]


def _generate_all_classes(
    raw_root: Path,
    output_root: Path,
    train_frame: pd.DataFrame,
    selected_classes: list[str],
    settings: GanSettings,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_idx, class_name in enumerate(selected_classes, start=1):
        _write_class_progress(
            output_root, seed, class_name, class_idx, len(selected_classes)
        )
        rows.extend(
            _generate_for_class(
                raw_root, output_root, train_frame, class_name, settings, seed
            )
        )
    return rows


if __name__ == "__main__":
    main()
