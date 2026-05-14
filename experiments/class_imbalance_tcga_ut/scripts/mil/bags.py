from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


def _feature_to_bag(path: str, max_instances: int | None) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, dict):
        tensor = next(value for value in tensor.values() if torch.is_tensor(value))
    features = tensor.float()
    if features.ndim == 1:
        features = features.unsqueeze(0)
    if features.ndim > 2:
        features = features.reshape(-1, features.shape[-1])
    if max_instances and len(features) > max_instances:
        indices = torch.linspace(0, len(features) - 1, max_instances).long()
        features = features.index_select(0, indices)
    return features


class BagFeatureDataset(Dataset):
    """Feature-bag dataset that preserves variable-size WSI tensors."""

    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_idx: dict[str, int],
        max_instances: int | None,
    ) -> None:
        rows = frame.reset_index(drop=True)
        self.bags = [
            _feature_to_bag(str(path), max_instances)
            for path in rows["feature_path"].tolist()
        ]
        self.labels = torch.tensor(
            [class_to_idx[str(name)] for name in rows["cancer_type"].tolist()],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.bags[idx], int(self.labels[idx].item())


class SyntheticBagFeatureDataset(BagFeatureDataset):
    """Feature-bag dataset with optional generated-image feature bags."""

    def append_rows(
        self,
        frame: pd.DataFrame,
        class_to_idx: dict[str, int],
        max_instances: int | None,
    ) -> int:
        """Append synthetic feature rows and return how many were added."""
        rows = frame.reset_index(drop=True)
        if rows.empty:
            return 0
        extra_bags = [
            _feature_to_bag(str(path), max_instances)
            for path in rows["feature_path"].tolist()
        ]
        extra_labels = torch.tensor(
            [class_to_idx[str(name)] for name in rows["cancer_type"].tolist()],
            dtype=torch.long,
        )
        self.bags.extend(extra_bags)
        self.labels = torch.cat([self.labels, extra_labels])
        return len(extra_bags)


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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Classify already pooled feature vectors."""
        encoded = self.instance_encoder(features)
        return self.classifier(encoded)


class MdeMil(nn.Module):
    """Two-expert MIL classifier for original and rebalanced objectives."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.original = AttentionMil(input_dim, hidden_dim, output_dim, dropout)
        self.rebalanced = AttentionMil(input_dim, hidden_dim, output_dim, dropout)

    def forward_bags(
        self, bags: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Classify feature bags with both experts."""
        original_logits, original_embeddings, _ = self.original.forward_bags(bags)
        rebalanced_logits, rebalanced_embeddings, _ = self.rebalanced.forward_bags(bags)
        logits = 0.5 * (original_logits + rebalanced_logits)
        embeddings = 0.5 * (original_embeddings + rebalanced_embeddings)
        return logits, embeddings, original_logits, rebalanced_logits


def class_weights(
    labels: np.ndarray, n_classes: int, beta: float = 0.999
) -> torch.Tensor:
    """Compute effective-number class weights."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    effective = (1.0 - np.power(beta, np.maximum(counts, 1.0))) / (1.0 - beta)
    weights = 1.0 / effective
    return torch.tensor(weights * (n_classes / weights.sum()), dtype=torch.float32)


def infer_input_dim(dataset: BagFeatureDataset) -> int:
    """Read the feature dimension from the first bag."""
    first_bag = cast(torch.Tensor, dataset[0][0])
    return int(first_bag.shape[-1])
