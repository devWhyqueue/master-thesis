from __future__ import annotations

import argparse
import logging
from typing import cast

import pandas as pd

from scripts.common import ensure_dirs, load_config
from common_code.wsi.bag_cache import write_split_cache

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
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
        write_split_cache(split_frame, cache_dir, split)
        logger.info("cached split=%s bags=%s", split, len(split_frame))


if __name__ == "__main__":
    main()
