from __future__ import annotations

import torch
from torch import nn


class PatchClassifier(nn.Module):
    """Shared CNN backbone and classifier for patch-level benchmarks."""

    def __init__(self, hidden_dim: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa
        """Return logits and learned embeddings for image batches."""
        embeddings = self.encoder(images)
        return self.classifier(embeddings), embeddings
