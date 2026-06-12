"""Build a row-level patch feature cache from a manifest CSV."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import torch

from tcga_ut_imbalanced.data.feature_store import load_feature_row

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse feature-cache arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--file-save-path", required=True)
    return parser.parse_args()


def main() -> None:
    """Build a row-level feature cache from a manifest with feature_path columns."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = get_args()
    if Path(args.file_save_path).is_file():
        logger.info("Skipping existing patch feature cache: %s", args.file_save_path)
        return
    manifest = pd.read_csv(args.manifest_path)
    os.makedirs(os.path.dirname(args.file_save_path) or ".", exist_ok=True)
    torch.save(_build_payload(manifest), args.file_save_path)
    logger.info("Stored feature cache in %s.", args.file_save_path)


def _build_payload(manifest: pd.DataFrame) -> dict[str, object]:
    has_index = "feature_index" in manifest.columns
    features = [
        load_feature_row(
            str(row["feature_path"]),
            int(row["feature_index"]) if has_index else 0,
        )
        for _, row in manifest.iterrows()
    ]
    payload: dict[str, object] = {
        "feature_paths": manifest["feature_path"].astype(str).tolist(),
        "features": torch.stack(features),
    }
    if has_index:
        payload["feature_indices"] = manifest["feature_index"].astype(int).tolist()
    return payload


if __name__ == "__main__":
    main()
