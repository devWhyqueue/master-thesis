"""Stage-one export: pooled embeddings, head state, and the CE run's logits (``stage1.pt``).

Stage two never re-runs the encoder: the source stage-one run's exact embeddings
are cached here, so the frozen-encoder claim in the report holds by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

STAGE1_ARTIFACT_NAME = "stage1.pt"

__all__ = ["STAGE1_ARTIFACT_NAME", "Stage1Artifacts", "save_stage1", "load_stage1"]


@dataclass
class Stage1Artifacts:
    """Cached stage-one outputs a stage-two method fits from."""

    train_embeddings: torch.Tensor
    train_y: np.ndarray
    val_embeddings: torch.Tensor
    val_y: np.ndarray
    test_embeddings: torch.Tensor
    test_y: np.ndarray
    counts: np.ndarray
    head_state: dict[str, Any]
    is_cosine: bool
    cosine_scale: float | None
    sigma: float | None


def save_stage1(path: Path, artifacts: Stage1Artifacts) -> None:
    """Persist one stage-one run's embeddings and head state for stage two."""
    torch.save(
        {
            "train_embeddings": artifacts.train_embeddings,
            "train_y": artifacts.train_y,
            "val_embeddings": artifacts.val_embeddings,
            "val_y": artifacts.val_y,
            "test_embeddings": artifacts.test_embeddings,
            "test_y": artifacts.test_y,
            "counts": artifacts.counts,
            "head_state": artifacts.head_state,
            "is_cosine": artifacts.is_cosine,
            "cosine_scale": artifacts.cosine_scale,
            "sigma": artifacts.sigma,
        },
        path,
    )


def load_stage1(path: Path) -> Stage1Artifacts:
    """Load a stage-one run's cached embeddings and head state."""
    payload = torch.load(path, weights_only=False)
    return Stage1Artifacts(**payload)
