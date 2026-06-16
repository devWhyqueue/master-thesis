from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler

from common_code.sampling import uses_balanced_sampler

class PatchImageDataset(Dataset):
    """Image dataset for controlled TCGA-UT patch manifests."""

    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_idx: dict[str, int],
        image_size: int,
    ) -> None:
        self.rows = frame.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows.iloc[idx]
        image = Image.open(Path(str(row["image_path"]))).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor, self.class_to_idx[str(row["cancer_type"])]


def patch_loader(
    dataset: PatchImageDataset,
    labels: np.ndarray,
    method: str,
    batch_size: int,
    seed: int,
    num_workers: int,
    samples_per_epoch: int | None = None,
) -> DataLoader:
    """Build deterministic patch training loaders."""
    generator = torch.Generator().manual_seed(seed)
    if samples_per_epoch is not None:
        sampler = RandomSampler(
            dataset,
            replacement=False,
            num_samples=min(samples_per_epoch, len(dataset)),
            generator=generator,
        )
        return _data_loader(dataset, batch_size, num_workers, sampler=sampler)
    if not uses_balanced_sampler(method):
        return _data_loader(
            dataset, batch_size, num_workers, shuffle=True, generator=generator
        )
    sampler = _balanced_sampler(labels, generator)
    return _data_loader(dataset, batch_size, num_workers, sampler=sampler)


def _data_loader(
    dataset: PatchImageDataset,
    batch_size: int,
    num_workers: int,
    sampler: RandomSampler | WeightedRandomSampler | None = None,
    shuffle: bool | None = None,
    generator: torch.Generator | None = None,
) -> DataLoader:
    persistent_workers = os.environ.get("PATCH_TRAINING_PERSISTENT_WORKERS", "1") != "0"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0 and persistent_workers,
    )


def _balanced_sampler(
    labels: np.ndarray, generator: torch.Generator
) -> WeightedRandomSampler:
    counts = np.bincount(labels)
    sample_weights = [float(1.0 / counts[label]) for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), True, generator)


def _load_manifest(
    paths: dict[str, Path],
    seed: int,
    smoke: bool,
    config: dict,
    staged_manifest: str | None = None,
) -> pd.DataFrame:
    manifest_path = staged_manifest or str(
        paths["data"] / f"patch_manifest_seed={seed}.csv"
    )
    frame = pd.read_csv(manifest_path)
    training = config["patch_training"]
    if not smoke:
        return frame
    limits = {
        "train": int(training.get("max_train_rows") or 64),
        "val": int(training.get("max_eval_rows") or 32),
        "test": int(training.get("max_eval_rows") or 32),
    }
    parts = [part.head(limits[str(name)]) for name, part in frame.groupby("split")]
    return pd.concat(parts, ignore_index=True)


def _labels(dataset: PatchImageDataset) -> np.ndarray:
    return np.asarray(
        [dataset.class_to_idx[str(name)] for name in dataset.rows["cancer_type"]],
        dtype=np.int64,
    )


def _split_datasets(
    frame: pd.DataFrame, config: dict
) -> tuple[PatchImageDataset, PatchImageDataset, PatchImageDataset, list[str]]:
    """Split a patch manifest into train/val/test datasets."""
    class_names = sorted(frame["cancer_type"].unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    image_size = int(config["patch_training"]["image_size"])
    datasets = []
    for split in ["train", "val", "test"]:
        datasets.append(
            PatchImageDataset(
                cast(pd.DataFrame, frame[frame["split"] == split]),
                class_to_idx,
                image_size,
            )
        )
    return datasets[0], datasets[1], datasets[2], class_names
