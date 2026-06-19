"""Export per-seed ProGAN synthetic patches for SquashFS creation on Hydra."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from scripts.common import ensure_dirs, load_config, write_json
from scripts.data.progan.storage import synthetic_output_root

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse synthetic SquashFS export arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--file-list",
        type=Path,
        help="Write synthetic patch paths relative to the seed image root.",
    )
    parser.add_argument(
        "--epoch-ref",
        type=int,
        default=None,
        help="If set, include only images under epochs=EPOCH_REF/ (e.g. 25).",
    )
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


def _synthetic_manifest(config: dict, seed: int) -> tuple[dict[str, Path], Path, Path]:
    paths = ensure_dirs(config)
    image_root = synthetic_output_root(paths["root"], seed)
    manifest_path = image_root / "synthetic_patch_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Expected synthetic manifest for seed={seed}: {manifest_path}"
        )
    return paths, image_root, manifest_path


def _unique_synthetic_paths(
    manifest_path: Path, epoch_ref: int | None = None
) -> list[str]:
    frame = pd.read_csv(manifest_path)
    paths = frame["image_path"].astype(str)
    if epoch_ref is not None:
        paths = paths[paths.str.contains(f"/epochs={epoch_ref}/", regex=False)]
    return sorted(set(paths.tolist()))


def _link_synthetic_paths(
    unique_paths: list[str], export_root: Path, image_root: Path
) -> int:
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
    return linked


def export_synthetic_sqfs_input(
    config: dict, seed: int, epoch_ref: int | None = None
) -> Path:
    """Hardlink one seed's synthetic JPEG tree for SquashFS packaging."""
    paths, image_root, manifest_path = _synthetic_manifest(config, seed)
    export_root = _export_root(seed)
    if export_root.exists():
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)
    unique_paths = _unique_synthetic_paths(manifest_path, epoch_ref)
    linked = _link_synthetic_paths(unique_paths, export_root, image_root)
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


def _relative_synthetic_paths(
    image_root: Path, manifest_path: Path, epoch_ref: int | None = None
) -> list[str]:
    relative_paths = []
    for source_text in _unique_synthetic_paths(manifest_path, epoch_ref):
        source = Path(source_text)
        try:
            relative_paths.append(source.relative_to(image_root).as_posix())
        except ValueError as error:
            raise ValueError(
                f"Synthetic patch path is not under image_root: {source}"
            ) from error
    return sorted(relative_paths)


def write_synthetic_sqfs_file_list(
    config: dict, seed: int, output_path: Path, epoch_ref: int | None = None
) -> Path:
    """Write unique synthetic patch paths relative to the seed image root."""
    paths, image_root, manifest_path = _synthetic_manifest(config, seed)
    relative_paths = _relative_synthetic_paths(image_root, manifest_path, epoch_ref)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(relative_paths) + "\n", encoding="utf-8")
    write_json(
        paths["data"] / f"synthetic_sqfs_export_report_seed={seed}.json",
        {
            "seed": seed,
            "export_mode": "file_list",
            "file_list": str(output_path),
            "image_root": str(image_root),
            "n_unique_images": len(relative_paths),
        },
    )
    logger.info(
        "Wrote %s synthetic paths for seed=%s to %s",
        len(relative_paths),
        seed,
        output_path,
    )
    return image_root


def main() -> None:
    """Export synthetic patch images for one seed."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    config = load_config(args.config)
    if args.file_list is not None:
        sys.stdout.write(
            f"{write_synthetic_sqfs_file_list(config, args.seed, args.file_list, args.epoch_ref)}\n"
        )
    else:
        export_synthetic_sqfs_input(config, args.seed, args.epoch_ref)


if __name__ == "__main__":
    main()
