"""Center-Focused Affinity Loss (CFAL) for frozen patch-feature classifiers.

Mahbub et al.; IEEE JBHI 2024. Formulas follow Eq. 5.3--5.7 in the source paper.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def effective_number(count: int, beta: float) -> float:
    """Return the effective number of samples for one class."""
    if count <= 0:
        return 1.0
    if beta >= 1.0:
        return float(count)
    return (1.0 - beta**count) / (1.0 - beta)


def _l2_normalize(vectors: torch.Tensor) -> torch.Tensor:
    """Project vectors onto the unit sphere for scale-stable affinity."""
    return torch.nn.functional.normalize(vectors, dim=-1, eps=1e-8)


def gaussian_affinity(
    features: torch.Tensor, prototypes: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Compute Gaussian affinity on L2-normalized embeddings (Eq. 5.3)."""
    features = _l2_normalize(features)
    prototypes = _l2_normalize(prototypes)
    diff = features.unsqueeze(1) - prototypes.unsqueeze(0)
    sq_dist = diff.square().sum(dim=-1)
    return torch.exp(-sq_dist / sigma)


def affinity_margin_loss(
    affinities: torch.Tensor, targets: torch.Tensor, margin: float
) -> torch.Tensor:
    """Per-sample max-margin affinity term (Eq. 5.5)."""
    batch_idx = torch.arange(len(targets), device=targets.device)
    true_aff = affinities[batch_idx, targets].unsqueeze(1)
    margins = torch.relu(margin + affinities - true_aff)
    class_mask = torch.nn.functional.one_hot(
        targets, num_classes=margins.size(1)
    ).bool()
    return margins.masked_fill(class_mask, 0.0).sum(dim=1)


def diversity_regularizer(prototypes: torch.Tensor) -> torch.Tensor:
    """Prototype diversity penalty R(w) on normalized prototypes (Eq. 5.6)."""
    prototypes = _l2_normalize(prototypes)
    n_classes = prototypes.size(0)
    if n_classes < 2:
        return prototypes.new_tensor(0.0)
    diffs = prototypes.unsqueeze(0) - prototypes.unsqueeze(1)
    sq_dists = diffs.square().sum(dim=-1)
    upper = torch.triu(sq_dists, diagonal=1)
    pairwise = upper[upper > 0]
    if pairwise.numel() == 0:
        return prototypes.new_tensor(0.0)
    mean_dist = pairwise.mean()
    return ((pairwise - mean_dist).square()).mean()


class CfalPrototypeClassifier(nn.Module):
    """Shared MLP trunk with learnable class prototypes in embedding space."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        dropout: float,
        sigma: float,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.prototypes = nn.Parameter(torch.empty(n_classes, hidden_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Map frozen patch features to the CFAL embedding space."""
        return self.encoder(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return class affinities used for prediction and evaluation."""
        return gaussian_affinity(self.encode(features), self.prototypes, self.sigma)


class CenterFocusedAffinityLoss(nn.Module):
    """CFAL objective with class-balanced and center-focused weighting (Eq. 5.7)."""

    def __init__(
        self,
        class_counts: np.ndarray,
        *,
        margin: float,
        sigma: float,
        gamma: float,
        beta: float,
    ) -> None:
        super().__init__()
        effective = np.array(
            [effective_number(int(count), beta) for count in class_counts],
            dtype=np.float64,
        )
        self.register_buffer(
            "inverse_effective",
            torch.tensor(1.0 / np.maximum(effective, 1e-8), dtype=torch.float32),
        )
        self.margin = margin
        self.sigma = sigma
        self.gamma = gamma

    def forward(
        self,
        model: CfalPrototypeClassifier,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the batch CFAL loss."""
        embeddings = model.encode(features)
        affinities = gaussian_affinity(embeddings, model.prototypes, self.sigma)
        margin_loss = affinity_margin_loss(affinities, targets, self.margin)
        batch_idx = torch.arange(len(targets), device=targets.device)
        true_aff = affinities[batch_idx, targets]
        local_weight = (1.0 - true_aff).clamp(min=0.0).pow(self.gamma)
        class_weight = self.inverse_effective[targets]
        weighted = class_weight * local_weight * margin_loss
        return weighted.mean() + diversity_regularizer(model.prototypes)


def build_cfal_model(
    dataset: object,
    settings: dict,
    n_classes: int,
    device: torch.device,
    tuning_params: dict[str, float],
) -> CfalPrototypeClassifier:
    """Build a CFAL prototype classifier for one patch-feature run."""
    sample, _ = dataset[0]  # type: ignore[index]
    sigma = float(tuning_params.get("cfal_sigma", settings["cfal_sigma"]))
    return CfalPrototypeClassifier(
        int(sample.numel()),
        int(settings["hidden_dim"]),
        n_classes,
        float(settings["dropout"]),
        sigma,
    ).to(device)


def build_cfal_loss(
    labels: np.ndarray,
    settings: dict,
    device: torch.device,
    tuning_params: dict[str, float],
) -> CenterFocusedAffinityLoss:
    """Build the CFAL loss for one training run."""
    n_classes = int(labels.max()) + 1 if labels.size else 1
    counts = np.bincount(labels, minlength=n_classes)
    return CenterFocusedAffinityLoss(
        counts,
        margin=float(tuning_params.get("cfal_lambda", settings["cfal_lambda"])),
        sigma=float(tuning_params.get("cfal_sigma", settings["cfal_sigma"])),
        gamma=float(tuning_params.get("cfal_gamma", settings["cfal_gamma"])),
        beta=float(tuning_params.get("cfal_beta", settings["cfal_beta"])),
    ).to(device)


def train_cfal_model(
    train_set: object,
    n_classes: int,
    settings: dict,
    device: torch.device,
    seed: int,
    tuning_params: dict[str, float],
) -> CfalPrototypeClassifier:
    """Train one CFAL patch-feature classifier."""
    labels = train_set.labels.cpu().numpy()  # type: ignore[attr-defined]
    model = build_cfal_model(train_set, settings, n_classes, device, tuning_params)
    loss_fn = build_cfal_loss(labels, settings, device, tuning_params)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    loader = DataLoader(
        train_set,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(1, int(settings["epochs"]) + 1):
        model.train()
        for features, targets in loader:
            batch_features = features.to(device)
            batch_targets = targets.to(device)
            loss = loss_fn(model, batch_features, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model
