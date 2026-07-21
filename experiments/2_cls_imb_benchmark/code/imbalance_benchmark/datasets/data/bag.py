from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from imbalance_benchmark.datasets.data.common import (
    class_names as resolve_class_names,
    slide_level_identity,
    validate_class_names,
)
from imbalance_benchmark.datasets.features import load_slide_features


class BagFeatureDataset(Dataset):
    """Multi-class WSI bags backed by frozen feature tensors."""

    def __init__(
        self,
        manifest_path: str | Path,
        split_name: str | None = None,
        max_instances: int | None = None,
        instance_selection_seed: int = 0,
        device: str | torch.device = "cpu",
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        del device
        self.max_instances = max_instances or None
        self.instance_selection_seed = instance_selection_seed
        frame = cast(pd.DataFrame, pd.read_csv(manifest_path))
        if split_name is not None and "split" in frame.columns:
            frame = cast(
                pd.DataFrame, frame[frame["split"] == split_name].reset_index(drop=True)
            )
        slide_rows = slide_level_identity(frame)
        self.df = cast(
            pd.DataFrame,
            frame.groupby("slide_id", sort=False)
            .agg({"case_id": "first", "cancer_type": "first", "feature_path": list})
            .reset_index(),
        )
        self.classes = (
            list(class_names)
            if class_names is not None
            else resolve_class_names(cast(pd.Series, slide_rows["cancer_type"]))
        )
        validate_class_names(self.df, self.classes)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}

    def get_n_classes(self) -> int:
        """Return the locked target size."""
        return len(self.classes)

    def get_int_targets(self) -> np.ndarray:
        """Return targets in slide order."""
        return np.asarray(
            [self.class_to_idx[name] for name in self.df["cancer_type"]], dtype=int
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        paths = list(dict.fromkeys(str(path) for path in row["feature_path"]))
        features = torch.cat([load_slide_features(path) for path in paths], dim=0)
        if self.max_instances is not None and len(features) > self.max_instances:
            features = features.index_select(
                0, self._selected_indices(row, len(features))
            )
        return features, self.class_to_idx[str(row["cancer_type"])]

    def _selected_indices(self, row: pd.Series, count: int) -> torch.Tensor:
        if self.instance_selection_seed == 0:
            return torch.linspace(0, count - 1, self.max_instances).long()
        digest = hashlib.sha256(
            f"{self.instance_selection_seed}:{row['slide_id']}".encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        selected = np.sort(rng.choice(count, self.max_instances, replace=False))
        return torch.as_tensor(selected, dtype=torch.long)


def bag_collate(
    items: list[tuple[torch.Tensor, int]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Collate variable-size bags without padding."""
    bags, labels = zip(*items, strict=False)
    return list(bags), torch.tensor(labels, dtype=torch.long)
