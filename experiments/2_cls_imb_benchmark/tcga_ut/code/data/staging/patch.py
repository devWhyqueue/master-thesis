"""Stage patch images to node-local storage for Hydra training jobs."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from code.common import ensure_dirs, load_config, write_json
from code.data.progan.storage import synthetic_output_root
from code.data.staging.io import _stage_with_copy, _stage_with_sqfs, _unmount_sqfs

logger = logging.getLogger(__name__)


def stage_root(seed: int) -> Path:
    """Return the per-job local staging directory."""
    base = Path(os.environ.get("PATCH_STAGE_DIR", ""))
    if str(base) == "":
        tmp = os.environ.get("SLURM_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
        base = Path(tmp) / f"tcga_ut_patch_seed={seed}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _staging_settings(config: dict) -> dict:
    return dict(config.get("staging", {}))


def _raw_root(config: dict) -> Path:
    return Path(config["paths"]["raw_root"])


def _sqfs_path(config: dict) -> Path | None:
    configured = config.get("paths", {}).get("patch_sqfs")
    if not configured:
        return None
    path = Path(str(configured))
    return path if path.exists() else None


def _synthetic_sqfs_path(config: dict, seed: int) -> Path | None:
    configured = config.get("paths", {}).get("patch_synthetic_sqfs")
    if not configured:
        return None
    path = Path(str(configured).format(seed=seed))
    return path if path.exists() else None


def load_seed_manifest(paths: dict[str, Path], seed: int) -> pd.DataFrame:
    """Load the controlled patch manifest for one seed."""
    return pd.read_csv(paths["data"] / f"patch_manifest_seed={seed}.csv")


def _synthetic_manifest_path(config: dict, seed: int) -> Path:
    paths = ensure_dirs(config)
    root = synthetic_output_root(paths["root"], seed)
    epoch_ref = os.environ.get("PATCH_SYNTHETIC_SQFS_EPOCH_REF")
    if epoch_ref:
        return root / f"epochs={epoch_ref}" / "synthetic_patch_manifest.csv"
    return root / "synthetic_patch_manifest.csv"


def combined_training_frame(
    paths: dict[str, Path], config: dict, seed: int, include_synthetic: bool
) -> pd.DataFrame:
    """Build the patch frame used for staging (optionally with synthetic train rows)."""
    frame = load_seed_manifest(paths, seed)
    frame["is_synthetic"] = False
    if not include_synthetic:
        return frame
    synthetic_path = _synthetic_manifest_path(config, seed)
    if not synthetic_path.exists():
        raise FileNotFoundError(
            f"Expected synthetic manifest for seed={seed}: {synthetic_path}"
        )
    synthetic = pd.read_csv(synthetic_path)
    synthetic["split"] = "train"
    synthetic["is_synthetic"] = True
    return pd.concat([frame, synthetic], ignore_index=True)


def _write_staged_manifest(
    target_dir: Path,
    staged_frame: pd.DataFrame,
    seed: int,
    mode: str,
    include_synthetic: bool,
    sqfs: Path | None,
    synthetic_sqfs: Path | None = None,
) -> Path:
    manifest_path = target_dir / "patch_manifest.csv"
    staged_frame.to_csv(manifest_path, index=False)
    os.environ["PATCH_STAGE_DIR"] = str(target_dir)
    os.environ["PATCH_STAGED_MANIFEST"] = str(manifest_path)
    write_json(
        target_dir / "staging_report.json",
        {
            "seed": seed,
            "mode": mode,
            "include_synthetic": include_synthetic,
            "n_rows": int(len(staged_frame)),
            "n_unique_images": int(staged_frame["image_path"].nunique()),
            "manifest_path": str(manifest_path),
            "sqfs_source": str(sqfs) if sqfs is not None else None,
            "synthetic_sqfs_source": (
                str(synthetic_sqfs) if synthetic_sqfs is not None else None
            ),
        },
    )
    return manifest_path


def stage_patch_manifest(
    config: dict,
    seed: int,
    include_synthetic: bool = False,
    stage_dir: Path | None = None,
) -> Path:
    """Stage manifest images to fast local storage and write a local manifest CSV."""
    paths = ensure_dirs(config)
    raw_root = _raw_root(config)
    target_dir = stage_dir or stage_root(seed)
    target_dir.mkdir(parents=True, exist_ok=True)
    frame = combined_training_frame(paths, config, seed, include_synthetic)
    sqfs = _sqfs_path(config)
    synthetic_sqfs = _synthetic_sqfs_path(config, seed) if include_synthetic else None
    staged_frame, mode = _stage_frame(
        config, paths, frame, target_dir, raw_root, sqfs, seed, include_synthetic
    )
    return _write_staged_manifest(
        target_dir, staged_frame, seed, mode, include_synthetic, sqfs, synthetic_sqfs
    )


def _stage_frame(
    config: dict,
    paths: dict[str, Path],
    frame: pd.DataFrame,
    target_dir: Path,
    raw_root: Path,
    sqfs: Path | None,
    seed: int,
    include_synthetic: bool,
) -> tuple[pd.DataFrame, str]:
    settings = _staging_settings(config)
    synthetic_root = _synthetic_root(paths, seed, include_synthetic)
    copy_workers = int(settings.get("copy_workers", 8))
    mount_point = target_dir / "sqfs_mount"
    try:
        return _stage_frame_inner(
            frame, target_dir, raw_root, sqfs, copy_workers, settings, synthetic_root
        )
    except (OSError, subprocess.CalledProcessError):
        _unmount_sqfs(mount_point)
        raise


def _synthetic_root(
    paths: dict[str, Path], seed: int, include_synthetic: bool
) -> Path | None:
    if not include_synthetic:
        return None
    return synthetic_output_root(paths["root"], seed)


def _stage_frame_inner(
    frame: pd.DataFrame,
    target_dir: Path,
    raw_root: Path,
    sqfs: Path | None,
    copy_workers: int,
    settings: dict,
    synthetic_root: Path | None,
) -> tuple[pd.DataFrame, str]:
    use_sqfs = bool(settings.get("prefer_sqfs", True)) and sqfs is not None
    if not use_sqfs or sqfs is None:
        return _stage_with_copy(frame, target_dir, raw_root, copy_workers, sqfs)
    pre_mount = os.environ.get("PATCH_SQFS_MOUNT")
    if _is_live_mount(pre_mount):
        return _stage_with_sqfs(
            frame,
            target_dir,
            raw_root,
            sqfs,
            copy_workers,
            mount=Path(str(pre_mount)),
            synthetic_root=synthetic_root,
        )
    try:
        return _stage_with_sqfs(
            frame,
            target_dir,
            raw_root,
            sqfs,
            copy_workers,
            synthetic_root=synthetic_root,
        )
    except FileNotFoundError:
        logger.warning("squashfuse not available; falling back to image copy")
        return _stage_with_copy(frame, target_dir, raw_root, copy_workers, sqfs)


def _is_live_mount(path_text: str | None) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    return path.is_dir() and any(path.iterdir())


def parse_args() -> argparse.Namespace:
    """Parse patch staging CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Also stage ProGAN synthetic training patches for this seed.",
    )
    return parser.parse_args()


def main() -> None:
    """Stage patch images to node-local storage and print the manifest path."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    manifest_path = stage_patch_manifest(
        load_config(args.config), args.seed, include_synthetic=args.include_synthetic
    )
    sys.stdout.write(f"{manifest_path}\n")


if __name__ == "__main__":
    main()
