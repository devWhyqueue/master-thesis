"""Pack constructed WSI manifests into per-split memmap bag caches."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from modeling.training.constructed_wsi_data import bag_rows, load_bag_rows

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse constructed WSI bag-cache arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    return parser.parse_args()


def main() -> None:
    """Write memmap bag caches for train, validation, and test splits."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    frame = pd.read_csv(args.manifest_path)
    frame = frame.copy()
    frame["split"] = frame["split"].replace({"validation": "val"})
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        split_frame = cast(pd.DataFrame, frame[frame["split"] == split])
        if split_frame.empty:
            continue
        if _split_cache_exists(cache_dir, split):
            logger.info("Skipping existing cache for split=%s in %s", split, cache_dir)
            continue
        _write_split_cache(split_frame, cache_dir, split)


def _split_cache_exists(cache_dir: Path, split: str) -> bool:
    return (
        (cache_dir / f"{split}_features.npy").exists()
        and (cache_dir / f"{split}_offsets.npy").exists()
        and (cache_dir / f"{split}_meta.json").exists()
    )


def _write_split_cache(frame: pd.DataFrame, cache_dir: Path, split: str) -> None:
    bags = bag_rows(frame)
    loaded: list[np.ndarray] = []
    feature_dim = 0
    for index, (_, row) in enumerate(bags.iterrows(), start=1):
        bag = load_bag_rows(row).numpy().astype(np.float32, copy=False)
        loaded.append(bag)
        feature_dim = int(bag.shape[-1])
        if index % 1000 == 0:
            logger.info("loaded split=%s bags=%s", split, index)
    lengths = [int(len(bag)) for bag in loaded]
    offsets = np.concatenate([[0], np.cumsum(np.asarray(lengths, dtype=np.int64))])
    features = np.lib.format.open_memmap(
        cache_dir / f"{split}_features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(int(offsets[-1]), feature_dim),
    )
    cursor = 0
    for index, bag in enumerate(loaded, start=1):
        features[cursor : cursor + len(bag)] = bag
        cursor += len(bag)
        if index % 1000 == 0:
            logger.info("cached split=%s bags=%s", split, index)
    np.save(cache_dir / f"{split}_offsets.npy", offsets)
    _write_meta(cache_dir, split, len(bags), int(offsets[-1]), feature_dim)


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
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
