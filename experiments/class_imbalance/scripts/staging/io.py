"""Copy and SquashFS helpers for patch staging."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def resolve_raw_image_path(path: Path, raw_root: Path) -> Path:
    """Map a BeeGFS raw path to a host-mounted SquashFS path when available."""
    mount = os.environ.get("PATCH_SQFS_MOUNT")
    if not mount:
        return path
    try:
        relative = path.relative_to(raw_root)
    except ValueError:
        return path
    resolved = Path(mount) / relative
    return resolved if resolved.exists() else path


def _is_under_raw(image_path: str, raw_root: Path) -> bool:
    try:
        Path(image_path).relative_to(raw_root)
    except ValueError:
        return False
    return True


def _split_raw_and_synthetic(
    frame: pd.DataFrame, raw_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = (
        frame["image_path"].astype(str).map(lambda path: _is_under_raw(path, raw_root))
    )
    return frame[mask].copy(), frame[~mask].copy()


def _remap_via_sqfs_mount(
    frame: pd.DataFrame, image_root: Path, mount_point: Path
) -> pd.DataFrame:
    staged = frame.copy()
    remapped: list[str] = []
    for source in staged["image_path"].astype(str):
        path = Path(source)
        relative = path.relative_to(image_root)
        target = mount_point / relative
        if not target.exists():
            raise FileNotFoundError(
                f"Staged image missing on SquashFS mount: {target}"
            )
        remapped.append(str(target))
    staged["image_path"] = remapped
    return staged


def stage_destination(source: Path, stage_dir: Path, raw_root: Path) -> Path:
    """Map a source patch path to its node-local staging destination."""
    try:
        relative = source.relative_to(raw_root)
        return stage_dir / "raw" / relative
    except ValueError:
        return stage_dir / "synthetic" / source.parent.name / source.name


def _copy_one(source: Path, stage_dir: Path, raw_root: Path) -> str:
    destination = stage_destination(source, stage_dir, raw_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return str(destination)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return str(destination)


def _stage_via_copy(
    frame: pd.DataFrame, stage_dir: Path, raw_root: Path, copy_workers: int
) -> pd.DataFrame:
    staged = frame.copy()
    sources = [Path(str(path)) for path in staged["image_path"].unique()]
    mapping: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=copy_workers) as pool:
        futures = {
            pool.submit(_copy_one, source, stage_dir, raw_root): source
            for source in sources
        }
        for future in as_completed(futures):
            source = futures[future]
            mapping[str(source)] = future.result()
    staged["image_path"] = staged["image_path"].astype(str).map(mapping)
    return staged


def _mount_sqfs(sqfs_source: Path, stage_dir: Path) -> Path:
    """Copy SquashFS to node-local disk and mount with squashfuse."""
    local_sqfs = stage_dir / "patches.sqfs"
    mount_point = stage_dir / "sqfs_mount"
    if mount_point.exists() and any(mount_point.iterdir()):
        return mount_point
    mount_point.mkdir(parents=True, exist_ok=True)
    if not local_sqfs.exists():
        shutil.copy2(sqfs_source, local_sqfs)
    subprocess.run(
        ["squashfuse", str(local_sqfs), str(mount_point)],
        check=True,
    )
    return mount_point


def _unmount_sqfs(mount_point: Path) -> None:
    if not mount_point.exists():
        return
    try:
        subprocess.run(["fusermount", "-u", str(mount_point)], check=False)
    except FileNotFoundError:
        return


def _resolve_sqfs_mount(sqfs: Path, target_dir: Path) -> Path:
    """Use a host-mounted SquashFS when available, otherwise mount locally."""
    configured = os.environ.get("PATCH_SQFS_MOUNT")
    if configured:
        mount = Path(configured)
        if mount.is_dir() and any(mount.iterdir()):
            return mount
    return _mount_sqfs(sqfs, target_dir)


def _stage_with_sqfs(
    frame: pd.DataFrame,
    target_dir: Path,
    raw_root: Path,
    sqfs: Path,
    copy_workers: int,
    mount: Path | None = None,
    synthetic_root: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    logger.info("Staging via SquashFS mount from %s", sqfs)
    resolved_mount = mount or _resolve_sqfs_mount(sqfs, target_dir)
    os.environ["PATCH_SQFS_MOUNT"] = str(resolved_mount)
    raw_frame, synthetic_frame = _split_raw_and_synthetic(frame, raw_root)
    staged_frame = _remap_via_sqfs_mount(raw_frame, raw_root, resolved_mount)
    if synthetic_frame.empty:
        return staged_frame, "sqfs"
    synthetic_mount_env = os.environ.get("PATCH_SYNTHETIC_SQFS_MOUNT")
    synthetic_mount = (
        Path(synthetic_mount_env)
        if synthetic_mount_env and Path(synthetic_mount_env).is_dir()
        else None
    )
    if synthetic_mount and synthetic_root is not None:
        staged_synthetic = _remap_via_sqfs_mount(
            synthetic_frame, synthetic_root, synthetic_mount
        )
        return (
            pd.concat([staged_frame, staged_synthetic], ignore_index=True),
            "sqfs+synthetic_sqfs",
        )
    logger.info("Synthetic SquashFS unavailable; copying synthetic patches")
    staged_synthetic = _stage_via_copy(
        synthetic_frame, target_dir, raw_root, copy_workers=copy_workers
    )
    return (
        pd.concat([staged_frame, staged_synthetic], ignore_index=True),
        "sqfs+copy",
    )


def _stage_with_copy(
    frame: pd.DataFrame,
    target_dir: Path,
    raw_root: Path,
    copy_workers: int,
    sqfs: Path | None,
) -> tuple[pd.DataFrame, str]:
    if sqfs is None:
        logger.info("SquashFS not configured; copying images to %s", target_dir)
    else:
        logger.info("SquashFS missing on disk; copying images to %s", target_dir)
    return (
        _stage_via_copy(frame, target_dir, raw_root, copy_workers=copy_workers),
        "copy",
    )
