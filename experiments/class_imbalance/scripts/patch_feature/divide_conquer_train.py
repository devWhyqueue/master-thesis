"""Training loop for divide-and-conquer patch-feature classifiers."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.patch_feature.divide_conquer_sampling import (
    BinarySubproblemDataset,
    build_sampled_subproblem_datasets,
)
from scripts.patch_feature.training import PatchFeatureDataset


class BinaryExpert(nn.Module):
    """Shared-architecture binary expert trunk and head."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Map frozen patch features to the expert embedding space."""
        return self.encoder(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return binary logits for one subproblem."""
        return self.head(self.encode(features)).squeeze(-1)


class DivideConquerModel(nn.Module):
    """Three binary experts fused into one multiclass classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.expert_hard_tail = BinaryExpert(input_dim, hidden_dim, dropout)
        self.expert_hard_head = BinaryExpert(input_dim, hidden_dim, dropout)
        self.expert_hard_rest = BinaryExpert(input_dim, hidden_dim, dropout)
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, n_classes),
        )

    def expert_embeddings(self, features: torch.Tensor) -> torch.Tensor:
        """Concatenate the three frozen or trainable expert embeddings."""
        return torch.cat(
            [
                self.expert_hard_tail.encode(features),
                self.expert_hard_head.encode(features),
                self.expert_hard_rest.encode(features),
            ],
            dim=-1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return multiclass logits from fused expert embeddings."""
        return self.fusion(self.expert_embeddings(features))

    def freeze_experts(self) -> None:
        """Stop gradient updates for all binary expert trunks."""
        for expert in (
            self.expert_hard_tail,
            self.expert_hard_head,
            self.expert_hard_rest,
        ):
            for param in expert.parameters():
                param.requires_grad_(False)


def build_divide_conquer_model(
    dataset: PatchFeatureDataset,
    settings: dict,
    n_classes: int,
    device: torch.device,
) -> DivideConquerModel:
    """Instantiate the divide-and-conquer model."""
    sample, _ = dataset[0]
    return DivideConquerModel(
        int(sample.numel()),
        int(settings["hidden_dim"]),
        n_classes,
        float(settings["dropout"]),
    ).to(device)


def train_divide_conquer_model(
    train_set: PatchFeatureDataset,
    class_names: list[str],
    n_classes: int,
    settings: dict,
    device: torch.device,
    seed: int,
    tuning_params: dict[str, float],
) -> tuple[DivideConquerModel, dict[str, object]]:
    """Train divide-and-conquer experts then the fusion head."""
    k_clusters = _setting_int(tuning_params, settings, "dnc_k_clusters", default=10)
    n_bins = _setting_int(settings, settings, "dnc_zscore_bins", default=5)
    expert_epochs = _setting_int(settings, settings, "dnc_expert_epochs", default=20)
    sp_ht, sp_hh, sp_hr, diagnostics = build_sampled_subproblem_datasets(
        train_set,
        class_names,
        k_clusters=k_clusters,
        n_bins=n_bins,
        seed=seed,
    )
    model = build_divide_conquer_model(train_set, settings, n_classes, device)
    _train_experts(model, (sp_ht, sp_hh, sp_hr), settings, device, seed, expert_epochs)
    _train_fusion(model, train_set, settings, device, seed, expert_epochs)
    diagnostics.update(
        {
            "dnc_k_clusters": k_clusters,
            "dnc_zscore_bins": n_bins,
            "dnc_expert_epochs": expert_epochs,
        }
    )
    return model, diagnostics


def _train_experts(
    model: DivideConquerModel,
    subproblems: tuple[
        BinarySubproblemDataset, BinarySubproblemDataset, BinarySubproblemDataset
    ],
    settings: dict,
    device: torch.device,
    seed: int,
    expert_epochs: int,
) -> None:
    bce = nn.BCEWithLogitsLoss()
    experts = (
        model.expert_hard_tail,
        model.expert_hard_head,
        model.expert_hard_rest,
    )
    params = [param for expert in experts for param in expert.parameters()]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    loaders = [
        DataLoader(
            dataset,
            batch_size=int(settings["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        for dataset in subproblems
    ]
    for _ in range(1, expert_epochs + 1):
        model.train()
        for expert, loader in zip(experts, loaders, strict=True):
            for features, targets in loader:
                logits = expert(features.to(device))
                loss = bce(logits, targets.to(device, dtype=torch.float32))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


def _train_fusion(
    model: DivideConquerModel,
    train_set: PatchFeatureDataset,
    settings: dict,
    device: torch.device,
    seed: int,
    expert_epochs: int,
) -> None:
    model.freeze_experts()
    optimizer = torch.optim.AdamW(
        model.fusion.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    loader = DataLoader(
        train_set,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    ce = nn.CrossEntropyLoss()
    total_epochs = int(settings["epochs"])
    for _ in range(expert_epochs + 1, total_epochs + 1):
        model.train()
        for features, targets in loader:
            logits = model(features.to(device))
            loss = ce(logits, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def _setting_int(
    primary: dict[str, float] | dict[str, Any],
    fallback: dict[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    if key in primary:
        return int(cast(float, primary[key]))
    value = fallback.get(key, default)
    return int(cast(int | float | str, value))
