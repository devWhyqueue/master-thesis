import argparse
import logging
import os

import pandas as pd
import torch

from tcga_ut_imbalanced.data.dataset import _load_feature_tensor

logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse feature-cache arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--file-save-path", required=True)
    parser.add_argument("--feature-path-column", default="feature_path")
    return parser.parse_args()


def main() -> None:
    """Build a feature cache from a row-level feature manifest."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = get_args()
    manifest = pd.read_csv(args.manifest_path)
    feature_paths = sorted(manifest[args.feature_path_column].drop_duplicates())
    logger.info("Caching %s feature tensors.", len(feature_paths))
    features = [_load_feature_tensor(str(path)) for path in feature_paths]
    os.makedirs(os.path.dirname(args.file_save_path), exist_ok=True)
    torch.save(
        {"feature_paths": feature_paths, "features": torch.stack(features)},
        args.file_save_path,
    )
    logger.info("Stored feature cache in %s.", args.file_save_path)


if __name__ == "__main__":
    main()
