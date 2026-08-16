from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from imbalance_benchmark.datasets.data import ImbalanceDataset
from imbalance_benchmark.modeling.context import MATCHED_BETA_METHOD
from imbalance_benchmark.modeling.losses import effective_number

__all__ = ["mean_one", "signal_criterion"]


def mean_one(raw: np.ndarray) -> torch.Tensor:
    """Rescale a positive per-class weight vector to mean one."""
    w = torch.tensor(raw, dtype=torch.float32)
    return w * (len(w) / w.sum())


def _case_counts(train_ds: ImbalanceDataset) -> np.ndarray:
    """G_c: distinct ``case_id`` count per class, in the dataset's class-index order."""
    per_class = train_ds.df.groupby("cancer_type")["case_id"].nunique()
    return np.array(
        [per_class.get(name, 0) for name in train_ds.classes], dtype=np.float64
    )


def _difficulty_array(ctx: dict[str, Any]) -> np.ndarray:
    """d_c per class, reordered from the frozen pilot evidence into class-index order."""
    difficulty = ctx["difficulty"]
    return np.array(
        [difficulty[name] for name in ctx["train_dataset"].classes], dtype=np.float64
    )


def signal_criterion(
    method: str, param: float | None, device: torch.device, ctx: dict[str, Any]
) -> nn.Module | None:
    """Build the mean-one class-weighted CE criterion for one signal method, or None."""
    if method == "class_balanced_ce":
        beta = float(param if param is not None else 0.99)
        w = mean_one(1.0 / effective_number(ctx["class_counts"], beta))
        return nn.CrossEntropyLoss(weight=w.to(device))
    if method in ("independent_support_ce", MATCHED_BETA_METHOD):
        beta = float(param if param is not None else 0.95)
        counts = _case_counts(ctx["train_dataset"])
        w = mean_one(1.0 / effective_number(counts, beta))
        return nn.CrossEntropyLoss(weight=w.to(device))
    if method == "pilot_difficulty_ce":
        tau = float(param if param is not None else 0.5)
        w = mean_one(np.power(_difficulty_array(ctx) + 1e-3, tau))
        return nn.CrossEntropyLoss(weight=w.to(device))
    return None
