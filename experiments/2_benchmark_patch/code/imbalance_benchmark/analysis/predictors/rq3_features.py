from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from imbalance_benchmark.datasets.data import (
    BagFeatureDataset,
    ImbalanceDataset,
    load_training_dataset,
)
from imbalance_benchmark.datasets.features.cache import bank_index
from imbalance_benchmark.modeling.training.semantic_scale import (
    EPS_S,
    init_pool,
    semantic_volumes,
)

logger = logging.getLogger(__name__)


def feature_frame(
    manifest: Path,
    split: str | None,
    is_mil: bool,
    class_names: list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load fixed embeddings and integer targets from one frozen manifest partition."""
    dataset = load_training_dataset(manifest, is_mil, split, class_names=class_names)
    if is_mil:
        bags = cast(BagFeatureDataset, dataset)
        features = [np.r_[bag.mean(0).cpu(), bag.std(0).cpu()] for bag, _ in bags]
    else:
        patches = cast(ImbalanceDataset, dataset)
        features = bank_index(patches.rows).cpu().numpy()
    return np.asarray(features), dataset.get_int_targets()


def _deprived_classes(
    balanced: dict[str, Any], imbalanced: dict[str, Any]
) -> list[str]:
    """Classes receiving fewer nominal examples than in balanced reference."""
    balanced_counts = balanced["allocated_counts"]
    return [
        name
        for name, count in imbalanced["allocated_counts"].items()
        if count < balanced_counts[name]
    ]


def _independent_shortage(
    balanced: dict[str, Any], imbalanced: dict[str, Any], is_mil: bool
) -> float:
    """Mean log loss of independent support across nominally deprived classes."""
    names = _deprived_classes(balanced, imbalanced)
    if not names:
        return 0.0
    key = "n_slides" if is_mil else "n_patients"
    return float(
        np.mean(
            [
                np.log(
                    balanced["contribution_stats"][name][key]
                    / imbalanced["contribution_stats"][name][key]
                )
                for name in names
            ]
        )
    )


def _diversity_shortage(
    balanced: dict[str, Any],
    imbalanced: dict[str, Any],
    balanced_volumes: dict[int, float],
    imbalanced_volumes: dict[int, float],
    class_names: list[str],
) -> float:
    """Mean log semantic-volume loss across nominally deprived classes."""
    names = _deprived_classes(balanced, imbalanced)
    if not names:
        return 0.0
    ratios = [
        np.log(
            max(balanced_volumes.get(class_names.index(name), EPS_S), EPS_S)
            / max(imbalanced_volumes.get(class_names.index(name), EPS_S), EPS_S)
        )
        for name in names
    ]
    return float(np.mean(ratios))


def _fixed_diversity(
    manifest: Path,
    is_mil: bool,
    class_names: list[str] | None,
    seed: int,
) -> dict[int, float]:
    """Semantic volumes from one condition's matched frozen-feature draw."""
    dataset = load_training_dataset(manifest, is_mil, None, class_names=class_names)
    if not is_mil:
        pool = init_pool(cast(ImbalanceDataset, dataset), seed, updates_per_pass=1)
        return semantic_volumes(pool.raw_features, pool.class_ids, len(dataset.classes))
    bags = cast(BagFeatureDataset, dataset)
    features = torch.stack([torch.cat((bag.mean(0), bag.std(0))) for bag, _ in bags])
    targets = torch.from_numpy(dataset.get_int_targets())
    return semantic_volumes(features, targets, len(np.unique(targets)))


def _reference_block(
    paths: dict[str, Path],
    is_mil: bool,
    class_names: list[str] | None,
    balanced_condition: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Cache balanced support and diversity before condition comparisons."""
    logger.info("rq3: seeding resident feature bank")
    feature_frame(paths["data"] / "manifest.csv", "validation", is_mil, class_names)
    balanced_path = Path(balanced_condition["path"])
    return {
        "condition": balanced_condition,
        "diversity": _fixed_diversity(balanced_path, is_mil, class_names, seed),
        "seed": seed,
    }


def _covariates(
    paths: dict[str, Path],
    is_mil: bool,
    condition: dict[str, Any],
    reference: dict[str, Any],
    freeze: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Signal shortages measured before mitigation fitting or test evaluation."""
    del paths
    class_names = list((freeze or {}).get("class_names", [])) or None
    cond_path = Path(condition["path"])
    if not cond_path.exists():
        raise RuntimeError(f"Missing frozen controlled manifest for RQ3: {cond_path}")
    names = class_names or list(condition["allocated_counts"])
    condition_diversity = _fixed_diversity(
        cond_path, is_mil, names, int(reference["seed"])
    )
    balanced = reference["condition"]
    return {
        "independent_shortage": _independent_shortage(balanced, condition, is_mil),
        "diversity_shortage": _diversity_shortage(
            balanced,
            condition,
            reference["diversity"],
            condition_diversity,
            names,
        ),
    }
