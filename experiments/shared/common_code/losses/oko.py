"""Odd-k-out (OKO) set learning for frozen patch-feature classifiers.

Muttenthaler et al.; ICLR 2024. Implements the hard-loss main-text variant:
k=1, pair-class prediction via main head, odd-class prediction via auxiliary
head that is discarded at inference time.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class OkoClassifier(nn.Module):
    """Shared trunk with a main head and an auxiliary odd-class head.

    At inference the auxiliary head is discarded; forward() returns main-head
    logits identical in shape to any standard nn.Sequential classifier, so
    existing evaluation code needs no changes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.main_head = nn.Linear(hidden_dim, n_classes)
        self.odd_head = nn.Linear(hidden_dim, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return trunk embeddings for input features."""
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return main-head logits for single-example inference."""
        return self.main_head(self.encode(x))


def _build_class_index(labels: np.ndarray) -> dict[int, list[int]]:
    """Build a mapping from class label to list of dataset indices."""
    index: dict[int, list[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        index.setdefault(int(label), []).append(idx)
    return index


def _sample_odd_classes_vec(
    pair_classes: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one odd class per set uniformly from [C] \\ {pair_class}."""
    raw = rng.integers(n_classes - 1, size=len(pair_classes))
    return np.where(raw < pair_classes, raw, raw + 1)


def _fill_pair_members(
    class_index: dict[int, list[int]],
    pair_classes: np.ndarray,
    set_indices: np.ndarray,
    n_classes: int,
    rng: np.random.Generator,
) -> None:
    """Fill columns 0 and 1 of set_indices with two samples from each pair class."""
    for c in range(n_classes):
        mask = pair_classes == c
        count = int(mask.sum())
        if count == 0:
            continue
        pool = np.array(class_index[c])
        drawn = rng.integers(len(pool), size=count * 2)
        set_indices[mask, 0] = pool[drawn[:count]]
        set_indices[mask, 1] = pool[drawn[count:]]


def _fill_odd_member(
    class_index: dict[int, list[int]],
    odd_classes: np.ndarray,
    set_indices: np.ndarray,
    column: int,
    n_classes: int,
    rng: np.random.Generator,
) -> None:
    """Fill one column of set_indices with one sample per assigned odd class."""
    for c in range(n_classes):
        mask = odd_classes == c
        count = int(mask.sum())
        if count == 0:
            continue
        pool = np.array(class_index[c])
        drawn = rng.integers(len(pool), size=count)
        set_indices[mask, column] = pool[drawn]


def _sample_sets(
    class_index: dict[int, list[int]],
    n_classes: int,
    n_sets: int,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample n_sets OKO sets following Algorithm 1 of Muttenthaler et al., 2024.

    Returns (pair_classes, set_indices of shape (n_sets, k+2), first_odd_classes).
    For k>1 the auxiliary loss uses only the first odd class; each slot samples
    independently from [C]\\{pair_class} so duplicates are possible when k>1.
    """
    pair_classes = rng.integers(n_classes, size=n_sets)
    set_indices = np.empty((n_sets, k + 2), dtype=np.int64)
    _fill_pair_members(class_index, pair_classes, set_indices, n_classes, rng)
    first_odd = _sample_odd_classes_vec(pair_classes, n_classes, rng)
    _fill_odd_member(class_index, first_odd, set_indices, 2, n_classes, rng)
    for slot in range(1, k):
        odd_col = _sample_odd_classes_vec(pair_classes, n_classes, rng)
        _fill_odd_member(class_index, odd_col, set_indices, 2 + slot, n_classes, rng)
    return pair_classes, set_indices, first_odd


def _gather_features(train_set: object, flat_indices: np.ndarray) -> torch.Tensor:
    """Gather feature vectors for flat dataset indices."""
    feat_rows = train_set.indices[flat_indices]  # type: ignore[attr-defined]
    arr = np.asarray(train_set.features[feat_rows], dtype=np.float32)  # type: ignore[attr-defined]
    return torch.from_numpy(arr.copy())


def _oko_loss(
    model: OkoClassifier,
    features: torch.Tensor,
    batch_n: int,
    set_size: int,
    pair_labels: torch.Tensor,
    odd_labels: torch.Tensor,
) -> torch.Tensor:
    """Compute OKO hard loss (pair-class CE) plus auxiliary odd-class CE."""
    summed = model.encode(features).view(batch_n, set_size, -1).sum(dim=1)
    return nn.functional.cross_entropy(
        model.main_head(summed), pair_labels
    ) + nn.functional.cross_entropy(model.odd_head(summed), odd_labels)


def _train_epoch(
    model: OkoClassifier,
    train_set: object,
    pair_classes: np.ndarray,
    set_indices: np.ndarray,
    odd_classes: np.ndarray,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    k: int,
) -> None:
    """Train one epoch over pre-sampled OKO sets."""
    set_size = k + 2
    model.train()
    for start in range(0, len(pair_classes), batch_size):
        end = min(start + batch_size, len(pair_classes))
        flat_idx = set_indices[start:end].reshape(-1)
        features = _gather_features(train_set, flat_idx).to(device)
        pair_t = torch.from_numpy(pair_classes[start:end]).long().to(device)
        odd_t = torch.from_numpy(odd_classes[start:end]).long().to(device)
        loss = _oko_loss(model, features, end - start, set_size, pair_t, odd_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _build_oko_model(
    train_set: object,
    n_classes: int,
    settings: dict,
    device: torch.device,
    tuning_params: dict[str, float],
) -> tuple[OkoClassifier, torch.optim.Optimizer, int]:
    """Build OkoClassifier, AdamW optimizer, and resolved k for one training run."""
    sample, _ = train_set[0]  # type: ignore[index]
    k = int(tuning_params.get("oko_k", float(settings["oko_k"])))
    model = OkoClassifier(
        input_dim=int(sample.numel()),
        hidden_dim=int(settings["hidden_dim"]),
        n_classes=n_classes,
        dropout=float(settings["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    return model, optimizer, k


def train_oko_model(
    train_set: object,
    n_classes: int,
    settings: dict,
    device: torch.device,
    seed: int,
    tuning_params: dict[str, float],
) -> OkoClassifier:
    """Train one OKO patch-feature classifier."""
    labels = train_set.labels.cpu().numpy()  # type: ignore[attr-defined]
    model, optimizer, k = _build_oko_model(
        train_set, n_classes, settings, device, tuning_params
    )
    class_index = _build_class_index(labels)
    rng = np.random.default_rng(seed)
    epochs = int(settings["epochs"])
    batch_size = int(settings["batch_size"])
    for _ in range(epochs):
        pair_classes, set_indices, odd_classes = _sample_sets(
            class_index, n_classes, len(labels), k, rng
        )
        _train_epoch(
            model,
            train_set,
            pair_classes,
            set_indices,
            odd_classes,
            optimizer,
            batch_size,
            device,
            k,
        )
    return model
