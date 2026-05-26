"""Cluster sampling and subproblem datasets for divide-and-conquer training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import Dataset

from scripts.common import EXPERIMENT_ROOT
from scripts.patch_feature.training import PatchFeatureDataset
from scripts.training.support_tiers import load_class_tier_labels


@dataclass(frozen=True)
class SubproblemSpec:
    name: str
    positive_classes: frozenset[str]
    negative_classes: frozenset[str]


class BinarySubproblemDataset(Dataset):
    def __init__(
        self,
        base: PatchFeatureDataset,
        indices: np.ndarray,
        positive_mask: np.ndarray,
    ) -> None:
        self.base = base
        self.indices = indices
        self.labels = torch.tensor(positive_mask.astype(np.int64), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        base_idx = int(self.indices[idx])
        features, _ = self.base[base_idx]
        return features, int(self.labels[idx].item())


def dnc_class_partitions(class_names: list[str]) -> dict[str, frozenset[str]]:
    """Return fixed, disjoint D&C groups derived only from dataset support."""
    slide_counts_path = (
        EXPERIMENT_ROOT / "outputs" / "tables" / "class_distribution.csv"
    )
    tier_labels = load_class_tier_labels(class_names, slide_counts_path)
    if tier_labels is None:
        raise FileNotFoundError(
            f"Missing class distribution table: {slide_counts_path}"
        )
    partitions = {
        tier: frozenset(name for name, label in tier_labels.items() if label == tier)
        for tier in ("tail", "body", "head")
    }
    if any(not group for group in partitions.values()):
        raise ValueError("D&C requires non-empty tail, body, and head support tiers")
    if set().union(*partitions.values()) != set(class_names):
        raise ValueError("D&C support tiers must cover every patch class exactly once")
    return partitions


def cluster_sample_binary_indices(
    dataset: PatchFeatureDataset,
    positive_idx: np.ndarray,
    negative_idx: np.ndarray,
    *,
    k_clusters: int,
    n_bins: int,
    seed: int,
    fit_cap: int = 20000,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Downsample the majority side with cluster and z-score stratification."""
    n_pos = len(positive_idx)
    n_neg = len(negative_idx)
    stats = {"positive": n_pos, "negative_before": n_neg, "negative_after": n_neg}
    if n_pos == 0 or n_neg == 0:
        return positive_idx, negative_idx, stats

    minority = n_pos if n_pos <= n_neg else n_neg
    majority_idx = negative_idx if n_pos <= n_neg else positive_idx
    sampled = _sample_majority_indices(
        dataset, majority_idx, minority, k_clusters, n_bins, seed, fit_cap
    )
    if n_pos <= n_neg:
        stats["negative_after"] = int(len(sampled))
        return positive_idx, sampled, stats
    stats["positive"] = int(len(sampled))
    return sampled, negative_idx, stats


def build_sampled_subproblem_datasets(
    train_set: PatchFeatureDataset,
    class_names: list[str],
    *,
    k_clusters: int,
    n_bins: int,
    seed: int,
) -> tuple[
    BinarySubproblemDataset,
    BinarySubproblemDataset,
    BinarySubproblemDataset,
    dict[str, object],
]:
    """Build cluster-sampled binary datasets for all three subproblems."""
    specs, diagnostics = _subproblem_specs(class_names)
    subproblem_stats = cast(dict[str, object], diagnostics["subproblems"])
    datasets = [
        _sampled_binary_dataset(
            train_set, spec, k_clusters, n_bins, seed, subproblem_stats
        )
        for spec in specs
    ]
    return datasets[0], datasets[1], datasets[2], diagnostics


def _subproblem_specs(
    class_names: list[str],
) -> tuple[tuple[SubproblemSpec, SubproblemSpec, SubproblemSpec], dict[str, object]]:
    partitions = dnc_class_partitions(class_names)
    tail = partitions["tail"]
    body = partitions["body"]
    head = partitions["head"]
    non_tail = body | head
    specs = (
        SubproblemSpec("tail_vs_body", tail, body),
        SubproblemSpec("tail_vs_head", tail, head),
        SubproblemSpec("tail_vs_rest", tail, non_tail),
    )
    diagnostics: dict[str, object] = {
        "partition_basis": "fixed_dataset_slide_support_tiers",
        "head_classes": sorted(head),
        "body_classes": sorted(body),
        "tail_classes": sorted(tail),
        "subproblems": {},
    }
    return specs, diagnostics


def _sampled_binary_dataset(
    train_set: PatchFeatureDataset,
    spec: SubproblemSpec,
    k_clusters: int,
    n_bins: int,
    seed: int,
    subproblem_stats: dict[str, object],
) -> BinarySubproblemDataset:
    class_column = train_set.rows["cancer_type"].astype(str)
    pos_idx = np.flatnonzero(class_column.isin(list(spec.positive_classes)))
    neg_idx = np.flatnonzero(class_column.isin(list(spec.negative_classes)))
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError(f"D&C subproblem {spec.name} requires examples on both sides")
    pos_idx, neg_idx, stats = cluster_sample_binary_indices(
        train_set,
        pos_idx,
        neg_idx,
        k_clusters=k_clusters,
        n_bins=n_bins,
        seed=seed,
    )
    indices = np.concatenate([pos_idx, neg_idx])
    positive_mask = np.concatenate(
        [np.ones(len(pos_idx), dtype=bool), np.zeros(len(neg_idx), dtype=bool)]
    )
    order = np.random.default_rng(seed).permutation(len(indices))
    subproblem_stats[spec.name] = stats
    return BinarySubproblemDataset(train_set, indices[order], positive_mask[order])


def _sample_majority_indices(
    dataset: PatchFeatureDataset,
    majority_idx: np.ndarray,
    minority: int,
    k_clusters: int,
    n_bins: int,
    seed: int,
    fit_cap: int,
) -> np.ndarray:
    majority_features = _features_at_indices(dataset, majority_idx)
    if len(majority_features) <= fit_cap:
        fit_features = majority_features
    else:
        selected = np.random.default_rng(seed).choice(
            len(majority_features), fit_cap, False
        )
        fit_features = majority_features[selected]
    k_actual = max(1, min(k_clusters, len(fit_features)))
    kmeans = KMeans(n_clusters=k_actual, random_state=seed, n_init="auto")
    kmeans.fit(_l2_normalize(fit_features))

    normalized = _l2_normalize(majority_features)
    labels = kmeans.predict(normalized)
    centroids = _l2_normalize(kmeans.cluster_centers_)
    dists = np.linalg.norm(normalized - centroids[labels], axis=1)
    z_scores = (dists - dists.mean()) / (dists.std() + 1e-8)
    bins = np.digitize(
        z_scores,
        np.linspace(z_scores.min(), z_scores.max(), max(n_bins, 1) + 1)[1:-1],
    )
    return _stratified_pick(majority_idx, bins, minority, n_bins, seed)


def _stratified_pick(
    majority_idx: np.ndarray,
    bins: np.ndarray,
    minority: int,
    n_bins: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    per_bin = max(1, minority // max(n_bins, 1))
    for bin_id in range(max(bins.max() + 1, n_bins)):
        pool = majority_idx[bins == bin_id]
        if len(pool) == 0:
            continue
        take = min(per_bin, len(pool), minority - len(selected))
        if take > 0:
            selected.extend(rng.choice(pool, take, replace=False).tolist())
    if len(selected) < minority:
        pool = np.setdiff1d(majority_idx, np.array(selected, dtype=np.int64))
        extra = min(minority - len(selected), len(pool))
        if extra > 0:
            selected.extend(rng.choice(pool, extra, replace=False).tolist())
    return np.array(selected[:minority], dtype=np.int64)


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-8)


def _features_at_indices(
    dataset: PatchFeatureDataset, indices: np.ndarray
) -> np.ndarray:
    return np.stack([dataset[int(idx)][0].numpy() for idx in indices], axis=0)
