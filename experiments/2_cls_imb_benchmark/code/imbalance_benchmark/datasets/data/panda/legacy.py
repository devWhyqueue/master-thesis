"""PANDA legacy-tile resolution: map eligible coordinates to on-disk JPEGs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

# (position, patch_id, legacy_image_path) for one resolved eligible coordinate.
Resolved = tuple[int, str, Path]
Resolver = Callable[[int, int, int], "Resolved | None"]


def legacy_resolver(
    directory: Path, manifest_path: Path | None
) -> tuple[Resolver, int]:
    """Return a ``(k, x, y) -> resolved`` lookup plus its total legacy-row count."""
    if manifest_path is not None:
        return _manifest_resolver(manifest_path)
    return _glob_resolver(directory)


def _manifest_resolver(manifest_path: Path) -> tuple[Resolver, int]:
    records = _manifest_records(manifest_path)
    by_xy = {
        (int(str(record["x"])), int(str(record["y"]))): (
            index,
            str(record["patch_id"]),
            Path(str(record["image_path"])),
        )
        for index, record in enumerate(records)
    }

    def _resolve(k: int, x: int, y: int) -> Resolved | None:
        del k
        return by_xy.get((x, y))

    return _resolve, len(records)


def _glob_resolver(directory: Path) -> tuple[Resolver, int]:
    all_jpgs = list(directory.glob("*.jpg"))
    paths = {int(path.stem): path for path in all_jpgs if path.stem.isdigit()}
    if len(paths) != len(all_jpgs):
        raise ValueError(f"PANDA legacy tiles have non-numeric names: {directory}")

    def _resolve(k: int, x: int, y: int) -> Resolved | None:
        del x, y
        path = paths.get(k)
        return None if path is None else (k, str(k), path)

    return _resolve, len(paths)


def _manifest_records(path: Path) -> list[dict[str, object]]:
    required = {"patch_id", "x", "y", "image_path"}
    frame = pd.read_csv(path)
    if (
        required - set(frame)
        or frame.duplicated(["patch_id"]).any()
        or frame.duplicated(["x", "y"]).any()
    ):
        raise ValueError(f"PANDA legacy tile manifest is invalid: {path}")
    return [dict(record) for record in frame.to_dict(orient="records")]
