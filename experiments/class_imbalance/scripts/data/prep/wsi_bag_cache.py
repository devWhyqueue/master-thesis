from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from scripts.common import ensure_dirs, load_config
from scripts.modeling.mil.bag.dataset import _feature_to_bag

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse WSI-bag cache arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    """Pack WSI feature bags into large per-split memmap arrays."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    frame = pd.read_csv(paths["data"] / f"manifest_splits_seed={args.seed}.csv")
    cache_dir = paths["data"] / "wsi_bag_cache" / f"seed={args.seed}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        split_frame = cast(pd.DataFrame, frame[frame["split"] == split])
        _write_split_cache(split_frame, cache_dir, split)


def _write_split_cache(frame: pd.DataFrame, cache_dir: Path, split: str) -> None:
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
    for index, path in enumerate(paths, start=1):
        bag = _feature_to_bag(path, None).numpy().astype(np.float32, copy=False)
        features[cursor : cursor + len(bag)] = bag
        cursor += len(bag)
        if index % 1000 == 0:
            logger.info("cached split=%s bags=%s", split, index)
    np.save(cache_dir / f"{split}_offsets.npy", offsets)
    _write_meta(cache_dir, split, len(paths), int(offsets[-1]), feature_dim)


def _scan(paths: list[str]) -> tuple[np.ndarray, int]:
    lengths: list[int] = []
    feature_dim = 0
    for index, path in enumerate(paths, start=1):
        bag = _feature_to_bag(path, None)
        lengths.append(int(len(bag)))
        feature_dim = int(bag.shape[-1])
        if index % 1000 == 0:
            logger.info("scanned bags=%s", index)
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


if __name__ == "__main__":
    main()
