"""WSI bag memmap cache builders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common_code.features import feature_to_bag


def write_split_cache(frame: pd.DataFrame, cache_dir: Path, split: str) -> None:
    paths = [str(path) for path in frame["feature_path"].tolist()]
    lengths, feature_dim = _scan(paths)
    offsets = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
    features = np.lib.format.open_memmap(
        cache_dir / f"{split}_features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(int(offsets[-1]), feature_dim),
    )
    cursor = 0
    for path in paths:
        bag = feature_to_bag(path, None).numpy().astype(np.float32, copy=False)
        features[cursor : cursor + len(bag)] = bag
        cursor += len(bag)
    np.save(cache_dir / f"{split}_offsets.npy", offsets)
    _write_meta(cache_dir, split, len(paths), int(offsets[-1]), feature_dim)


def _scan(paths: list[str]) -> tuple[np.ndarray, int]:
    lengths: list[int] = []
    feature_dim = 0
    for path in paths:
        bag = feature_to_bag(path, None)
        lengths.append(int(len(bag)))
        feature_dim = int(bag.shape[-1])
    return np.asarray(lengths, dtype=np.int64), feature_dim


def _write_meta(
    cache_dir: Path, split: str, n_bags: int, n_instances: int, feature_dim: int
) -> None:
    payload = {
        "split": split,
        "n_bags": n_bags,
        "n_instances": n_instances,
        "feature_dim": feature_dim,
        "dtype": "float32",
    }
    (cache_dir / f"{split}_meta.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
