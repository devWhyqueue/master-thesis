from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from imbalance_benchmark.datasets.data import ImbalanceDataset
from imbalance_benchmark.modeling.models import MLP
from imbalance_benchmark.modeling.training.signal_weights import mean_one

__all__ = [
    "EPS_S",
    "SsbPool",
    "init_pool",
    "prepare_ssb_pool",
    "semantic_volumes",
    "ssb_loss",
]

EPS_S = 1e-3


@dataclass
class SsbPool:
    """One rolling matched-draw pool (R=1): equal cases and patches per class.

    ``features`` rolls forward via :func:`_refresh`, one chunk per optimizer
    update; ``volumes`` retains each class's last valid semantic-scale
    estimate across updates where its current draw is unusable.
    """

    class_ids: torch.Tensor
    raw_features: torch.Tensor
    chunk_size: int
    features: torch.Tensor | None = None
    filled: torch.Tensor | None = None
    cursor: int = 0
    volumes: dict[int, float] = field(default_factory=dict)
    invalid_draws: int = 0


def _matched_draw_indices(
    train_ds: ImbalanceDataset, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices and class ids for one draw with equal cases and patches per class."""
    rng = np.random.default_rng(seed)
    by_class = {name: group for name, group in train_ds.df.groupby("cancer_type")}
    groups = [by_class[name] for name in train_ds.classes]
    n_cases = int(min(group["case_id"].nunique() for group in groups))
    per_class_rows = []
    for group in groups:
        unique_cases = np.asarray(group["case_id"].unique())
        cases = rng.choice(unique_cases, size=n_cases, replace=False).tolist()
        per_class_rows.append(group[group["case_id"].isin(cases)].index.to_numpy())
    n_patches = max(2, min(len(rows) for rows in per_class_rows))
    indices, class_ids = [], []
    for class_id, rows in enumerate(per_class_rows):
        replace = len(rows) < n_patches
        indices.append(rng.choice(rows, size=n_patches, replace=replace))
        class_ids.append(np.full(n_patches, class_id))
    return np.concatenate(indices), np.concatenate(class_ids)


def init_pool(train_ds: ImbalanceDataset, seed: int, updates_per_pass: int) -> SsbPool:
    """Build one matched draw and its empty rolling pool, sized for one pass-1 traversal."""
    indices, class_ids = _matched_draw_indices(train_ds, seed)
    raw_features = train_ds.__getitems__(indices.tolist())["features"]
    chunk_size = max(1, math.ceil(len(indices) / updates_per_pass))
    return SsbPool(
        class_ids=torch.from_numpy(class_ids),
        raw_features=raw_features,
        chunk_size=chunk_size,
    )


def prepare_ssb_pool(ctx: dict[str, Any], b_size: int) -> None:
    """Attach a fresh rolling matched-draw pool when the method needs one."""
    if ctx["method"] != "semantic_scale_ce":
        return
    updates_per_pass = math.ceil(len(ctx["train_dataset"]) / b_size)
    ctx["ssb_updates_per_pass"] = updates_per_pass
    ctx["ssb_pool"] = init_pool(ctx["train_dataset"], ctx["seed"], updates_per_pass)


def _refresh(pool: SsbPool, model: MLP, device: torch.device) -> None:
    """Re-encode the next chunk of the matched draw with the current model."""
    n = len(pool.raw_features)
    start = pool.cursor
    stop = min(start + pool.chunk_size, n)
    idx = torch.arange(start, stop)
    with torch.no_grad():
        encoded = model.encode(pool.raw_features[idx].to(device)).cpu()
    if pool.features is None or pool.filled is None:
        pool.features = torch.zeros(n, encoded.shape[-1])
        pool.filled = torch.zeros(n, dtype=torch.bool)
    assert pool.features is not None and pool.filled is not None
    features, filled = pool.features, pool.filled
    features[idx] = encoded
    filled[idx] = True
    pool.cursor = 0 if stop >= n else stop


def _isotropic_scale(features: torch.Tensor) -> float:
    """sqrt(mean per-dimension variance) across the pool's filled features."""
    centered = features - features.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0)
    return float(variance.mean().clamp(min=1e-12).sqrt())


def _log2_volume(centered: torch.Tensor, d: int) -> float:
    """S' = 0.5 log2 det(I + (d/M) Z Z^T), via the smaller of the Gram or covariance."""
    m = centered.shape[0]
    scale = d / m
    if m <= d:
        gram = centered @ centered.T
        mat = torch.eye(m) + scale * gram
    else:
        cov = centered.T @ centered
        mat = torch.eye(d) + scale * cov
    _, logdet = torch.linalg.slogdet(mat.double())
    return 0.5 * float(logdet) / math.log(2.0)


def semantic_volumes(
    features: torch.Tensor, class_ids: torch.Tensor, n_classes: int
) -> dict[int, float]:
    """Measure classwise semantic volume after pooled isotropic scaling."""
    scale = _isotropic_scale(features) if len(features) else 1.0
    d = features.shape[-1]
    volumes = {}
    for class_id in range(n_classes):
        class_features = features[class_ids == class_id]
        if len(class_features) < 2:
            continue
        centered = class_features / scale
        centered -= centered.mean(dim=0, keepdim=True)
        volume = _log2_volume(centered, d)
        if volume != 0.0:
            volumes[class_id] = volume
    return volumes


def _update_volumes(pool: SsbPool, n_classes: int) -> None:
    """Recompute each class's semantic scale from its currently filled draw."""
    if pool.features is None or pool.filled is None:
        return
    filled = pool.features[pool.filled]
    class_ids = pool.class_ids[pool.filled]
    for class_id in range(n_classes):
        if int((class_ids == class_id).sum()) < 2:
            pool.invalid_draws += 1
    pool.volumes.update(semantic_volumes(filled, class_ids, n_classes))


def _ssb_weights(pool: SsbPool, n_classes: int, tau: float) -> torch.Tensor:
    raw = np.array(
        [max(pool.volumes.get(c, EPS_S), EPS_S) ** (-tau) for c in range(n_classes)]
    )
    return mean_one(raw)


def ssb_loss(
    model: MLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    ctx: dict[str, Any],
    step: int,
) -> torch.Tensor:
    """Semantic-scale weighted CE: unit weights through pass 5, SSB weights from pass 6."""
    pool: SsbPool = ctx["ssb_pool"]
    device, n_classes = ctx["device"], ctx["n_classes"]
    _refresh(pool, model, device)
    logits = model(inputs)
    reweight_step = 5 * ctx["ssb_updates_per_pass"]
    if step <= reweight_step:
        if step == reweight_step:
            _update_volumes(pool, n_classes)
            ctx.setdefault("method_diagnostics", {})["ssb_invalid_draws"] = (
                pool.invalid_draws
            )
        return F.cross_entropy(logits, targets)
    _update_volumes(pool, n_classes)
    ctx.setdefault("method_diagnostics", {})["ssb_invalid_draws"] = pool.invalid_draws
    tau = float(ctx["param"] if ctx["param"] is not None else 0.5)
    weight = _ssb_weights(pool, n_classes, tau).to(device)
    return F.cross_entropy(logits, targets, weight=weight)
