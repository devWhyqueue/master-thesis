from __future__ import annotations

from typing import cast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


from common_code.features import feature_to_bag as _feature_to_bag

class BagFeatureDataset(Dataset):
    """Feature-bag dataset that preserves variable-size WSI tensors."""

    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_idx: dict[str, int],
        max_instances: int | None,
        cache_dir: Path | None = None,
        split: str | None = None,
    ) -> None:
        rows = frame.reset_index(drop=True)
        self.feature_paths = [str(path) for path in rows["feature_path"].tolist()]
        self.max_instances = max_instances
        self.labels = torch.tensor(
            [class_to_idx[str(name)] for name in rows["cancer_type"].tolist()],
            dtype=torch.long,
        )
        self.cache = _BagCache(cache_dir, split) if cache_dir and split else None
        if self.cache and len(self.cache) != len(self.labels):
            raise ValueError(f"Cache row count does not match manifest split: {split}")

    def __len__(self) -> int:
        return len(self.feature_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        bag = (
            self.cache.bag(idx, self.max_instances)
            if self.cache
            else _feature_to_bag(self.feature_paths[idx], self.max_instances)
        )
        return bag, int(self.labels[idx].item())


class _BagCache:
    """Memory-mapped feature-bag cache backed by large contiguous arrays."""

    def __init__(self, cache_dir: Path, split: str) -> None:
        features_path = cache_dir / f"{split}_features.npy"
        offsets_path = cache_dir / f"{split}_offsets.npy"
        if not features_path.exists() or not offsets_path.exists():
            raise FileNotFoundError(f"Incomplete WSI bag cache for split: {split}")
        self.features = np.load(features_path, mmap_mode="r")
        self.offsets = np.load(offsets_path)

    def __len__(self) -> int:
        return int(len(self.offsets) - 1)

    def bag(self, idx: int, max_instances: int | None) -> torch.Tensor:
        """Return one cached bag as a CPU tensor."""
        start, end = int(self.offsets[idx]), int(self.offsets[idx + 1])
        array = np.asarray(self.features[start:end], dtype=np.float32)
        bag = torch.from_numpy(array.copy())
        if max_instances and len(bag) > max_instances:
            indices = torch.linspace(0, len(bag) - 1, max_instances).long()
            bag = bag.index_select(0, indices)
        return bag


def bag_collate(
    items: list[tuple[torch.Tensor, int]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Collate variable-size feature bags without padding."""
    bags, labels = zip(*items, strict=False)
    return list(bags), torch.tensor(labels, dtype=torch.long)


class AttentionMil(nn.Module):
    """Attention-based MIL classifier for frozen WSI feature bags."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward_bags(
        self, bags: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Classify feature bags and return logits, embeddings, and attention."""
        embeddings: list[torch.Tensor] = []
        attention_weights: list[torch.Tensor] = []
        for bag in bags:
            encoded = self.instance_encoder(bag)
            weights = torch.softmax(self.attention(encoded).squeeze(1), dim=0)
            embeddings.append(torch.sum(encoded * weights.unsqueeze(1), dim=0))
            attention_weights.append(weights)
        bag_embeddings = torch.stack(embeddings)
        return self.classifier(bag_embeddings), bag_embeddings, attention_weights

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # noqa
        """Classify already pooled feature vectors."""
        encoded = self.instance_encoder(features)
        return self.classifier(encoded)

    def rank_scores(self, bag: torch.Tensor, class_id: int) -> torch.Tensor:
        """Return teacher pseudo-label scores for one class across a bag."""
        encoded = self.instance_encoder(bag)
        return torch.softmax(self.classifier(encoded), dim=1)[:, class_id]

    def project_bag_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project and normalize bag embeddings for supervised contrastive loss."""
        return nn.functional.normalize(self.projector(embeddings), dim=1)


def _expert_head(hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class DualExpertMil(nn.Module):
    """Shared attention MIL aggregator with dual distribution-specific experts."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.expert_u = _expert_head(hidden_dim, output_dim, dropout)
        self.expert_b = _expert_head(hidden_dim, output_dim, dropout)

    def aggregate(self, bags: list[torch.Tensor]) -> torch.Tensor:
        """Pool instance features into one embedding per bag."""
        embeddings: list[torch.Tensor] = []
        for bag in bags:
            encoded = self.instance_encoder(bag)
            weights = torch.softmax(self.attention(encoded).squeeze(1), dim=0)
            embeddings.append(torch.sum(encoded * weights.unsqueeze(1), dim=0))
        return torch.stack(embeddings)

    def logits_u(self, bag_embeddings: torch.Tensor) -> torch.Tensor:
        return self.expert_u(bag_embeddings)

    def logits_b(self, bag_embeddings: torch.Tensor) -> torch.Tensor:
        return self.expert_b(bag_embeddings)

    def forward_ensemble(self, bags: list[torch.Tensor]) -> torch.Tensor:
        """Return mean expert logits for held-out evaluation."""
        embeddings = self.aggregate(bags)
        return (self.logits_u(embeddings) + self.logits_b(embeddings)) * 0.5


def class_weights(
    labels: np.ndarray, n_classes: int, beta: float = 0.999, power: float = 1.0
) -> torch.Tensor:
    """Compute effective-number class weights."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    effective = (1.0 - np.power(beta, np.maximum(counts, 1.0))) / (1.0 - beta)
    weights = np.power(1.0 / effective, power)
    return torch.tensor(weights * (n_classes / weights.sum()), dtype=torch.float32)


def infer_input_dim(dataset: BagFeatureDataset) -> int:
    """Read the feature dimension from the first bag."""
    first_bag = cast(torch.Tensor, dataset[0][0])
    return int(first_bag.shape[-1])
