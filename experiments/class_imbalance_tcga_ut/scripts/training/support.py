import logging
from typing import cast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


def _feature_to_vector(path: str) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, dict):
        tensor = next(value for value in tensor.values() if torch.is_tensor(value))
    features = tensor.float()
    if features.ndim > 1:
        features = features.reshape(-1, features.shape[-1]).mean(dim=0)
    return features.flatten()


def _labels_to_indices(frame: pd.DataFrame, class_to_idx: dict[str, int]) -> np.ndarray:
    return np.asarray(
        [class_to_idx[str(class_name)] for class_name in frame["cancer_type"]],
        dtype=np.int64,
    )


class FeatureDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, class_to_idx: dict[str, int]) -> None:
        frame = frame.reset_index(drop=True)
        self.features = torch.stack(
            [_feature_to_vector(str(path)) for path in frame["feature_path"].tolist()]
        )
        self.labels = torch.tensor(
            [
                class_to_idx[str(class_name)]
                for class_name in frame["cancer_type"].tolist()
            ],
            dtype=torch.long,
        )
        logger.info("Loaded %s feature tensors into memory", len(self.features))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.features[idx], int(self.labels[idx].item())


class Mlp(nn.Module):
    """Simple MLP classifier for frozen feature vectors."""

    def __init__(
        self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run classifier forward pass."""
        return self.model(features)


class FocalLoss(nn.Module):
    """Multi-class focal loss."""

    def __init__(self, gamma: float, weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weights = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss value."""
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.weights is not None:
            loss = loss * self.weights[targets]
        return loss.mean()


def _class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    normalized = weights * (n_classes / weights.sum())
    return torch.tensor(normalized, dtype=torch.float32)


def _resolve_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured)


def _build_criterion(
    method: str,
    labels: np.ndarray,
    n_classes: int,
    gamma: float,
    device: torch.device,
) -> nn.Module:
    weights = _class_weights(labels, n_classes).to(device)
    if method == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights)
    if method == "focal":
        return FocalLoss(gamma=gamma)
    if method in {"weighted_focal", "balanced_sampler_weighted_focal"}:
        return FocalLoss(gamma=gamma, weights=weights)
    return nn.CrossEntropyLoss()


def _make_loader(
    dataset: FeatureDataset,
    labels: np.ndarray,
    method: str,
    batch_size: int,
    seed: int,
) -> DataLoader:
    if method not in {
        "balanced_sampler_ce",
        "balanced_sampler_weighted_focal",
        "oversampling_ce",
    }:
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=generator
        )
    counts = np.bincount(labels)
    sample_weights = np.array(
        [1.0 / counts[label] for label in labels], dtype=np.float64
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _metric_payload(
    y_true: list[int],
    y_pred: list[int],
    probabilities: list[list[float]],
    class_names: list[str],
) -> dict[str, object]:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=cast(str, 0)
    )
    precision = cast(np.ndarray, precision)
    recall = cast(np.ndarray, recall)
    f1 = cast(np.ndarray, f1)
    support = cast(np.ndarray, support)
    present = cast(np.ndarray, support > 0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision[present])),
        "macro_recall": float(np.mean(recall[present])),
        "macro_f1": float(np.mean(f1[present])),
        "class_names": class_names,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(map(int, y_true)),
        "preds": list(map(int, y_pred)),
        "probabilities": probabilities,
    }


def _evaluate(
    model: nn.Module,
    dataset: FeatureDataset,
    class_names: list[str],
    device: torch.device,
) -> dict[str, object]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            logits = model.forward(features.to(device))
            probs = torch.softmax(logits, dim=1)
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
    return _metric_payload(y_true, y_pred, probabilities, class_names)


def _load_numpy(
    frame: pd.DataFrame, class_to_idx: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    features = [_feature_to_vector(path).numpy() for path in frame["feature_path"]]
    labels = _labels_to_indices(frame, class_to_idx)
    return np.vstack(features), labels


def _batch_center_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    loss = torch.tensor(0.0, device=logits.device)
    for class_id in targets.unique():
        class_logits = logits[targets == class_id]
        if len(class_logits) > 1:
            center = class_logits.mean(dim=0, keepdim=True)
            loss = loss + ((class_logits - center) ** 2).mean()
    return loss


def _interpolate_minority(
    features: torch.Tensor, targets: torch.Tensor, train_labels: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = np.bincount(train_labels)
    median_count = np.median(counts[counts > 0])
    minority_classes = {
        idx for idx, count in enumerate(counts) if count <= median_count
    }
    mask = torch.tensor(
        [int(target.item()) in minority_classes for target in targets],
        device=features.device,
    )
    if mask.sum() < 2:
        return features, targets
    minority_features = features[mask]
    minority_targets = targets[mask]
    perm = torch.randperm(len(minority_features), device=features.device)
    mixed = 0.5 * minority_features + 0.5 * minority_features[perm]
    same_class = minority_targets == minority_targets[perm]
    if same_class.any():
        features = torch.cat([features, mixed[same_class]], dim=0)
        targets = torch.cat([targets, minority_targets[same_class]], dim=0)
    return features, targets
