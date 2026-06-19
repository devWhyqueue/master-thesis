from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from code.common import ensure_dirs, load_config, write_json

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for patch-manifest generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_patch_manifest(
    raw_root: Path,
    slide_splits: pd.DataFrame,
    resolution: str,
    patches_per_slide: int,
) -> pd.DataFrame:
    """Build a deterministic patch manifest from one fixed resolution."""
    rows: list[dict[str, object]] = []
    for slide in slide_splits.to_dict("records"):
        class_name = str(slide["cancer_type"])
        slide_id = str(slide["slide_id"])
        slide_root = raw_root / class_name / resolution / slide_id
        if not slide_root.exists():
            continue
        image_paths = sorted(
            path for path in slide_root.iterdir() if path.suffix == ".jpg"
        )
        for image_path in image_paths[:patches_per_slide]:
            rows.append(
                {
                    "slide_id": slide_id,
                    "cancer_type": class_name,
                    "split": str(slide["split"]),
                    "resolution": resolution,
                    "image_path": str(image_path),
                }
            )
    return pd.DataFrame(rows)


def _write_outputs(paths: dict[str, Path], seed: int, frame: pd.DataFrame) -> None:
    """Persist the controlled patch manifest and summary report."""
    path = paths["data"] / f"patch_manifest_seed={seed}.csv"
    frame.to_csv(path, index=False)
    write_json(
        paths["data"] / f"patch_manifest_report_seed={seed}.json",
        {
            "seed": seed,
            "n_patches": int(len(frame)),
            "n_slides": int(frame["slide_id"].nunique()) if not frame.empty else 0,
            "patches_by_split": frame["split"].value_counts().to_dict(),
        },
    )
    logger.info("Wrote %s", path)


def main() -> None:
    """Create one leakage-safe controlled patch manifest from slide splits."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    slide_splits = pd.read_csv(paths["data"] / f"slide_splits_seed={args.seed}.csv")
    frame = build_patch_manifest(
        Path(config["paths"]["raw_root"]),
        slide_splits,
        str(config["data"]["patch_resolution"]),
        int(config["data"]["patches_per_slide"]),
    )
    _write_outputs(paths, args.seed, frame)


if __name__ == "__main__":
    main()
