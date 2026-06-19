from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


def tail_classes(manifest: pd.DataFrame, max_classes: int | None = None) -> list[str]:
    """Return minority train classes that need synthetic augmentation."""
    counts = cast(pd.Series, train_rows(manifest)["cancer_type"].value_counts())
    target = int(counts.max())
    classes = [str(name) for name, count in counts.items() if int(count) < target]
    return classes if max_classes is None else classes[:max_classes]


def train_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return the training rows of one constructed manifest."""
    return cast(pd.DataFrame, manifest[manifest["split"] == "train"].copy())


def with_real_metadata(manifest: pd.DataFrame) -> pd.DataFrame:
    """Add explicit metadata columns for real rows."""
    real = manifest.copy()
    real["is_synthetic"] = False
    real["final_depth_epochs"] = 0
    return real


def combined_manifest(
    manifest: pd.DataFrame,
    generated_by_variant: dict[int, pd.DataFrame],
    cache_path: str,
) -> pd.DataFrame:
    """Append per-variant synthetic train rows to the real manifest."""
    real = with_real_metadata(manifest)
    synthetic = _synthetic_manifest(generated_by_variant, cache_path)
    if synthetic.empty:
        return real
    return cast(pd.DataFrame, pd.concat([real, synthetic], ignore_index=True))


def _synthetic_manifest(
    generated_by_variant: dict[int, pd.DataFrame], cache_path: str
) -> pd.DataFrame:
    next_index = 0
    frames: list[pd.DataFrame] = []
    for variant in sorted(generated_by_variant):
        frame = generated_by_variant[variant].copy()
        frame["feature_path"] = cache_path
        frame["feature_index"] = np.arange(
            next_index, next_index + len(frame), dtype=int
        )
        next_index += len(frame)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))


def synthetic_manifest_rows(
    rows: list[dict[str, object]],
    variant: int,
) -> list[dict[str, object]]:
    """Map generated JPEG rows into the constructed patch-manifest schema."""
    out: list[dict[str, object]] = []
    for row in rows:
        image_path = str(row["image_path"])
        patch_id = Path(image_path).stem
        out.append(
            {
                "split": "train",
                "slide_id": patch_id,
                "cancer_type": str(row["cancer_type"]),
                "patch_id": patch_id,
                "image_path": image_path,
                "is_synthetic": True,
                "final_depth_epochs": variant,
            }
        )
    return out


def resolve_real_patch_path(
    raw_root: Path,
    raw_resolution: str,
    cancer_type: str,
    slide_id: str,
    patch_id: str,
) -> Path:
    """Resolve one real patch image path from constructed manifest columns."""
    path = raw_root / cancer_type / raw_resolution / slide_id / f"{patch_id}.jpg"
    if not path.is_file():
        raise FileNotFoundError(f"Missing raw patch image: {path}")
    return path


def class_image_paths(
    frame: pd.DataFrame,
    raw_root: Path,
    raw_resolution: str,
    limit: int,
    seed: int,
) -> list[Path]:
    """Return deterministic real image paths for one train class."""
    paths = [
        resolve_real_patch_path(
            raw_root,
            raw_resolution,
            str(row["cancer_type"]),
            str(row["slide_id"]),
            str(row["patch_id"]),
        )
        for _, row in frame.iterrows()
    ]
    if len(paths) <= limit:
        return paths
    rng = np.random.default_rng(subsample_seed(seed, str(frame.iloc[0]["cancer_type"])))
    selected = rng.choice(len(paths), size=limit, replace=False)
    return [paths[int(index)] for index in selected]


def subsample_seed(seed: int, class_name: str) -> int:
    """Return a stable class-specific seed for real-patch subsampling."""
    digest = hashlib.sha256(f"{seed}:{class_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")
