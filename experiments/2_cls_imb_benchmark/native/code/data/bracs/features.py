"""Extract frozen Virchow2 features for prepared BRACS ROI-tile manifests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from data.progan.features import _embed_batch, _load_feature_model

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse feature extraction arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--feature-path", required=True)
    parser.add_argument("--model-name", default="hf-hub:paige-ai/Virchow2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Extract features and rewrite BRACS manifests with feature references."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    manifest_dir = Path(args.manifest_dir)
    manifest_path = manifest_dir / "manifest_splits.csv"
    frame = pd.read_csv(manifest_path)
    feature_path = Path(args.feature_path)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    if not feature_path.exists():
        features = extract_features(
            frame["image_path"].astype(str).tolist(),
            str(args.model_name),
            int(args.batch_size),
            str(args.dtype),
            torch.device(str(args.device)),
        )
        torch.save(features, feature_path)
        logger.info("Wrote %s features to %s", len(features), feature_path)
    enriched = frame.copy()
    enriched["feature_path"] = str(feature_path)
    enriched["feature_index"] = range(len(enriched))
    _write_manifests(manifest_dir, enriched)


def extract_features(
    image_paths: list[str],
    model_name: str,
    batch_size: int,
    dtype: str,
    device: torch.device,
) -> torch.Tensor:
    """Return stacked Virchow2 features for image paths."""
    model, transforms = _load_feature_model(model_name, device)
    loader = DataLoader(
        ImagePathDataset(image_paths, transforms),
        batch_size=batch_size,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    rows: list[torch.Tensor] = []
    for index, images in enumerate(loader, start=1):
        rows.extend(_embed_batch(model, images.to(device), dtype, device))
        if index % 25 == 0:
            logger.info("embedded batches=%s images=%s", index, len(rows))
    return torch.stack(rows) if rows else torch.empty((0, 0))


class ImagePathDataset(Dataset):  # type: ignore[type-arg]
    """Small image-path dataset for feature extraction."""

    def __init__(self, paths: list[str], transforms) -> None:
        self.paths = paths
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        return cast(torch.Tensor, self.transforms(image))


def _write_manifests(manifest_dir: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(manifest_dir / "manifest_splits.csv", index=False)
    for split in ("train", "validation", "test"):
        split_frame = frame[frame["split"] == split].copy()
        split_frame.to_csv(manifest_dir / f"{split}.csv", index=False)


if __name__ == "__main__":
    main()
