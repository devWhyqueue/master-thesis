"""Frozen-feature MLP head."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Sequence[int] | int,
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.model = build_mlp_layers(input_dim, hidden_dim, output_dim, dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def build_mlp_layers(
    input_dim: int,
    hidden_dim: Sequence[int] | int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    if isinstance(hidden_dim, int):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    layers = nn.Sequential()
    previous_dim = input_dim
    for layer_dim in hidden_dim:
        layers.append(nn.Linear(previous_dim, layer_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        previous_dim = layer_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return layers


def build_patch_feature_model(
    input_dim: int, hidden_dim: int, output_dim: int, dropout: float, device: torch.device
) -> nn.Module:
    return build_mlp_layers(input_dim, hidden_dim, output_dim, dropout).to(device)
