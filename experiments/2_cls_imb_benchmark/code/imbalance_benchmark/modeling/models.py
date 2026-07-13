from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "MLP",
    "AttentionMil",
    "DualExpertMil",
    "OkoClassifier",
    "CfalPrototypeClassifier",
]


class MLP(nn.Module):
    """Two-layer MLP classifier for frozen patch features."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> None:
        """Initialize the MLP model."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        return self.net(x)


class AttentionMil(nn.Module):
    """Attention-based WSI-level MIL classifier."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> None:
        """Initialize the AttentionMIL classifier."""
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
        """Forward pass over multiple instance bags."""
        embeddings: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        for bag in bags:
            encoded = self.instance_encoder(bag)
            attn_weights = torch.softmax(self.attention(encoded).squeeze(1), dim=0)
            embeddings.append(torch.sum(encoded * attn_weights.unsqueeze(1), dim=0))
            attentions.append(attn_weights)
        bag_embeddings = torch.stack(embeddings)
        return self.classifier(bag_embeddings), bag_embeddings, attentions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for pre-pooled features."""
        return self.classifier(self.instance_encoder(x))

    def rank_scores(self, bag: torch.Tensor, class_id: int) -> torch.Tensor:
        """Score instances in a bag for a target class using the current model."""
        return torch.softmax(self.classifier(self.instance_encoder(bag)), dim=-1)[
            :, class_id
        ]

    def project_bag_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project and normalize bag embeddings."""
        return F.normalize(self.projector(embeddings), dim=1)


class DualExpertMil(nn.Module):
    """Attention MIL aggregator with dual expert classifiers."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> None:
        """Initialize the DualExpertMIL aggregator."""
        super().__init__()
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.expert_u = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.expert_b = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def aggregate(self, bags: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate instance bags into slide-level representations."""
        embeddings: list[torch.Tensor] = []
        for bag in bags:
            encoded = self.instance_encoder(bag)
            attn_weights = torch.softmax(self.attention(encoded).squeeze(1), dim=0)
            embeddings.append(torch.sum(encoded * attn_weights.unsqueeze(1), dim=0))
        return torch.stack(embeddings)

    def logits_u(self, bag_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute logits for Expert U (unbalanced/natural)."""
        return self.expert_u(bag_embeddings)

    def logits_b(self, bag_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute logits for Expert B (balanced)."""
        return self.expert_b(bag_embeddings)

    def forward_ensemble(self, bags: list[torch.Tensor]) -> torch.Tensor:
        """Ensemble logits of both experts."""
        embeddings = self.aggregate(bags)
        return 0.5 * (self.logits_u(embeddings) + self.logits_b(embeddings))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ensemble forward pass for pre-pooled features."""
        encoded = self.instance_encoder(x)
        return 0.5 * (self.expert_u(encoded) + self.expert_b(encoded))


class OkoClassifier(nn.Module):
    """OKO set learning classifier with main and auxiliary odd heads."""

    def __init__(
        self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float
    ) -> None:
        """Initialize OkoClassifier."""
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.main_head = nn.Linear(hidden_dim, n_classes)
        self.odd_head = nn.Linear(hidden_dim, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs to trunk representation."""
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of OKO classifier (inference uses main head)."""
        return self.main_head(self.encode(x))


class CfalPrototypeClassifier(nn.Module):
    """CFAL classifier with learnable embedding prototypes."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        dropout: float,
        sigma: float,
    ) -> None:
        """Initialize CfalPrototypeClassifier."""
        super().__init__()
        self.sigma = sigma
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.prototypes = nn.Parameter(torch.empty(n_classes, hidden_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features to prototype space."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class affinities for input features."""
        emb = F.normalize(self.encode(x), dim=-1, eps=1e-8)
        proto = F.normalize(self.prototypes, dim=-1, eps=1e-8)
        sq_dist = (emb.unsqueeze(1) - proto.unsqueeze(0)).square().sum(dim=-1)
        return torch.exp(-sq_dist / self.sigma)
