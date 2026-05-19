"""Export per-seed ProGAN synthetic patches for SquashFS creation on Hydra."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from scripts.common import ensure_dirs, load_config, write_json
from scripts.progan.storage import synthetic_output_root

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse synthetic SquashFS export arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def _export_root(seed: int) -> Path:
    configured = os.environ.get("PATCH_SYNTHETIC_SQFS_INPUT")
    if configured:
        return Path(configured)
    tmp = os.environ.get("SLURM_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
    return Path(tmp) / f"tcga_ut_synthetic_seed={seed}"


def _link_into_tree(source: Path, export_root: Path, image_root: Path) -> bool:
    destination = export_root / source.relative_to(image_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return False
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return True


def export_synthetic_sqfs_input(config: dict, seed: int) -> Path:
    """Hardlink one seed's synthetic JPEG tree for SquashFS packaging."""
    paths = ensure_dirs(config)
    image_root = synthetic_output_root(paths["root"], seed)
    manifest_path = image_root / "synthetic_patch_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Expected synthetic manifest for seed={seed}: {manifest_path}"
        )
    export_root = _export_root(seed)
    if export_root.exists():
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest_path)
    unique_paths = sorted(frame["image_path"].astype(str).unique())
    workers = int(os.environ.get("PATCH_SQFS_EXPORT_WORKERS", "16"))
    linked = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_link_into_tree, Path(path), export_root, image_root)
            for path in unique_paths
        ]
        for future in as_completed(futures):
            if future.result():
                linked += 1
    write_json(
        paths["data"] / f"synthetic_sqfs_export_report_seed={seed}.json",
        {
            "seed": seed,
            "export_root": str(export_root),
            "image_root": str(image_root),
            "n_unique_images": len(unique_paths),
            "n_linked": linked,
        },
    )
    logger.info(
        "Exported %s synthetic images for seed=%s to %s",
        len(unique_paths),
        seed,
        export_root,
    )
    return export_root


def main() -> None:
    """Export synthetic patch images for one seed."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    export_synthetic_sqfs_input(load_config(args.config), args.seed)


if __name__ == "__main__":
    main()
