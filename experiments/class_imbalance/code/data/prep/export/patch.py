"""Export a deduplicated patch tree for one-time SquashFS creation on Hydra."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.common import ensure_dirs, load_config, write_json
from scripts.data.staging.patch import load_seed_manifest

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse export arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--file-list",
        type=Path,
        help="Write controlled patch paths relative to raw_root for tar/mksquashfs.",
    )
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


def _seed_frames(config: dict) -> tuple[dict[str, Path], list[int], list]:
    paths = ensure_dirs(config)
    seeds = [int(seed) for seed in config["patch_training"]["seeds"]]
    frames = [load_seed_manifest(paths, seed) for seed in seeds]
    return paths, seeds, frames


def _unique_patch_paths(
    config: dict,
) -> tuple[dict[str, Path], Path, list[int], list[str]]:
    paths, seeds, frames = _seed_frames(config)
    raw_root = Path(config["paths"]["raw_root"])
    unique_paths = sorted(
        {str(path) for frame in frames for path in frame["image_path"].astype(str)}
    )
    return paths, raw_root, seeds, unique_paths


def _link_paths(unique_paths: list[str], export_root: Path, raw_root: Path) -> int:
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
    return linked


def export_sqfs_input(config: dict) -> Path:
    """Hardlink all controlled patch images into one directory tree."""
    paths, raw_root, seeds, unique_paths = _unique_patch_paths(config)
    export_root = _export_root(config)
    if export_root.exists():
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    linked = _link_paths(unique_paths, export_root, raw_root)
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


def _relative_patch_paths(raw_root: Path, unique_paths: list[str]) -> list[str]:
    relative_paths = []
    for source_text in unique_paths:
        source = Path(source_text)
        try:
            relative_paths.append(source.relative_to(raw_root).as_posix())
        except ValueError as error:
            raise ValueError(f"Patch path is not under raw_root: {source}") from error
    return sorted(relative_paths)


def write_sqfs_file_list(config: dict, output_path: Path) -> Path:
    """Write unique controlled patch paths relative to raw_root."""
    paths, raw_root, seeds, unique_paths = _unique_patch_paths(config)
    relative_paths = _relative_patch_paths(raw_root, unique_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(relative_paths) + "\n", encoding="utf-8")
    write_json(
        paths["data"] / "patch_sqfs_export_report.json",
        {
            "export_mode": "file_list",
            "file_list": str(output_path),
            "n_unique_images": len(relative_paths),
            "raw_root": str(raw_root),
            "seeds": seeds,
        },
    )
    logger.info("Wrote %s patch paths to %s", len(relative_paths), output_path)
    return raw_root


def main() -> None:
    """Export the controlled patch image tree for SquashFS packaging."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    config = load_config(args.config)
    if args.file_list is not None:
        sys.stdout.write(f"{write_sqfs_file_list(config, args.file_list)}\n")
    else:
        export_sqfs_input(config)


if __name__ == "__main__":
    main()
