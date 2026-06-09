from collections.abc import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Sequence[int] | int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.model = _build_layers(input_dim, hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the multilayer perceptron."""
        return self.model(inputs)


def _build_layers(
    input_dim: int,
    hidden_dim: Sequence[int] | int,
    output_dim: int,
) -> nn.Sequential:
    if isinstance(hidden_dim, int):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, dtype=torch.float64),
        )
    return _build_multi_hidden_layers(input_dim, hidden_dim, output_dim)


def _build_multi_hidden_layers(
    input_dim: int,
    hidden_dim: Sequence[int],
    output_dim: int,
) -> nn.Sequential:
    layers = nn.Sequential()
    previous_dim = input_dim
    for layer_dim in hidden_dim:
        layers.append(nn.Linear(previous_dim, layer_dim, dtype=torch.float64))
        layers.append(nn.ReLU())
        previous_dim = layer_dim
    layers.append(nn.Linear(hidden_dim[-1], output_dim, dtype=torch.float64))
    return layers
