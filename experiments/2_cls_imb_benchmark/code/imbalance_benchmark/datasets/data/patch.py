from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from imbalance_benchmark.datasets.data.common import (
    class_names as resolve_class_names,
    validate_class_names,
)
from imbalance_benchmark.datasets.features import load_feature_row
from imbalance_benchmark.datasets.features.cache import bank_index, feature_rows


class ImbalanceDataset(Dataset):
    """Frozen patch features indexed without per-sample DataFrame construction."""

    def __init__(
        self,
        manifest_path: str | Path,
        split_name: str | None = None,
        device: str | torch.device = "cpu",
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        del device
        frame = cast(pd.DataFrame, pd.read_csv(manifest_path))
        capacity_hint = len(frame)
        if split_name is not None and "split" in frame.columns:
            frame = cast(
                pd.DataFrame, frame[frame["split"] == split_name].reset_index(drop=True)
            )
        self.df = frame
        self.classes = (
            list(class_names)
            if class_names is not None
            else resolve_class_names(cast(pd.Series, frame["cancer_type"]))
        )
        validate_class_names(frame, self.classes)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.feature_paths = frame["feature_path"].astype(str).tolist()
        self.feature_indices = self._feature_indices(frame)
        self.target_names = frame["cancer_type"].astype(str).tolist()
        self.slide_ids = frame["slide_id"].tolist()
        self.patch_ids = (
            frame["patch_id"].tolist() if "patch_id" in frame else [None] * len(frame)
        )
        self.rows = feature_rows(
            self.feature_paths, self.feature_indices, capacity_hint
        )
        self.targets = torch.tensor(self.get_int_targets(), dtype=torch.long)

    @staticmethod
    def _feature_indices(frame: pd.DataFrame) -> list[int | None]:
        if "feature_index" not in frame:
            return [None] * len(frame)
        return [
            int(value) if pd.notna(value) else None for value in frame["feature_index"]
        ]

    def get_n_classes(self) -> int:
        """Return the locked target size."""
        return len(self.classes)

    def get_int_targets(self) -> np.ndarray:
        """Return targets in manifest order."""
        return np.asarray(
            [self.class_to_idx[name] for name in self.target_names], dtype=int
        )

    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        target_name = self.target_names[index]
        patch_id = self.patch_ids[index] or f"{self.slide_ids[index]}_{index}"
        return {
            "slide_id": self.slide_ids[index],
            "features": load_feature_row(
                self.feature_paths[index], self.feature_indices[index]
            ),
            "target_str": target_name,
            "target": self.class_to_idx[target_name],
            "patch_id": patch_id,
        }

    def __getitems__(self, indices: list[int]) -> dict[str, Any]:
        """Return an already-collated batch; pair with :func:`patch_collate`.

        ``DataLoader`` calls this instead of ``__getitem__`` + per-item collate
        whenever it is present, so this is the fast path every patch-regime
        loader must use.
        """
        return {
            "features": bank_index(self.rows[indices]),
            "target": self.targets[indices],
        }


def patch_collate(batch: dict[str, Any]) -> dict[str, Any]:
    """Identity collate for :class:`ImbalanceDataset`: ``__getitems__`` already batches."""
    return batch
