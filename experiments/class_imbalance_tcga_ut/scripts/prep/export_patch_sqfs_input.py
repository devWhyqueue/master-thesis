"""Export a deduplicated patch tree for one-time SquashFS creation on Hydra."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.common import ensure_dirs, load_config, write_json
from scripts.staging.patch import load_seed_manifest

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse export arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def _link_into_tree(source: Path, export_root: Path, raw_root: Path) -> bool:
    destination = export_root / source.relative_to(raw_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return False
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return True


def _export_root(config: dict) -> Path:
    configured = os.environ.get("PATCH_SQFS_INPUT") or config.get("paths", {}).get(
        "patch_sqfs_input"
    )
    if configured:
        return Path(str(configured))
    paths = ensure_dirs(config)
    return paths["data"] / "patch_sqfs_input"


def export_sqfs_input(config: dict) -> Path:
    """Hardlink all controlled patch images into one directory tree."""
    paths = ensure_dirs(config)
    raw_root = Path(config["paths"]["raw_root"])
    export_root = _export_root(config)
    if export_root.exists():
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in config["patch_training"]["seeds"]]
    frames = [load_seed_manifest(paths, seed) for seed in seeds]
    unique_paths = sorted(
        {str(path) for frame in frames for path in frame["image_path"].astype(str)}
    )
    workers = int(os.environ.get("PATCH_SQFS_EXPORT_WORKERS", "16"))
    linked = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_link_into_tree, Path(source_text), export_root, raw_root)
            for source_text in unique_paths
        ]
        for future in as_completed(futures):
            if future.result():
                linked += 1
    write_json(
        paths["data"] / "patch_sqfs_export_report.json",
        {
            "export_root": str(export_root),
            "n_unique_images": len(unique_paths),
            "n_linked": linked,
            "seeds": seeds,
        },
    )
    logger.info("Exported %s unique images to %s", len(unique_paths), export_root)
    return export_root


def main() -> None:
    """Export the controlled patch image tree for SquashFS packaging."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    export_sqfs_input(load_config(parse_args().config))


if __name__ == "__main__":
    main()
