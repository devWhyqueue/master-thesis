from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier

from imbalance_benchmark.common import compute_data_hash
from imbalance_benchmark.datasets.data import (
    BagFeatureDataset,
    ImbalanceDataset,
    load_training_dataset,
)
from imbalance_benchmark.datasets.features.cache import bank_index


def _recalls(
    predictions: np.ndarray, labels: np.ndarray, class_names: list[str]
) -> dict[str, float]:
    """Return classwise recall for one complete OOF prediction vector."""
    return {
        name: float((predictions[labels == index] == index).mean())
        for index, name in enumerate(class_names)
    }


def _fold_predictions(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Generate grouped five-fold OOF predictions for both fixed probes."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    linear, knn = np.full(labels.size, -1, int), np.full(labels.size, -1, int)
    folds = []
    for fold, (train, test) in enumerate(splitter.split(features, labels, groups)):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Grouped difficulty fold leaked a case")
        linear[test] = (
            LogisticRegression(class_weight="balanced", max_iter=1000)
            .fit(features[train], labels[train])
            .predict(features[test])
        )
        knn[test] = (
            KNeighborsClassifier(n_neighbors=min(5, len(train)))
            .fit(features[train], labels[train])
            .predict(features[test])
        )
        folds.append(
            {
                "fold": fold,
                "train_groups": sorted(map(str, np.unique(groups[train]))),
                "test_groups": sorted(map(str, np.unique(groups[test]))),
            }
        )
    return linear, knn, folds


def grouped_difficulty(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Return five-fold grouped OOF recall evidence for one balanced pilot."""
    n_classes = len(class_names)
    if min(np.unique(groups[labels == index]).size for index in range(n_classes)) < 5:
        raise ValueError("Difficulty pilot requires five groups in every class")
    linear, knn, folds = _fold_predictions(features, labels, groups)
    return {
        "folds": folds,
        "linear_probe_recall": _recalls(linear, labels, class_names),
        "knn_recall": _recalls(knn, labels, class_names),
    }


def pilot_difficulty(
    manifest_path: Path, manifest: pd.DataFrame, is_mil: bool, class_names: list[str]
) -> dict[str, Any]:
    """Load one largest pilot manifest and attach its grouped difficulty evidence."""
    dataset = load_training_dataset(manifest_path, is_mil, class_names=class_names)
    if is_mil:
        bags = cast(BagFeatureDataset, dataset)
        rows = [(bag.mean(0).cpu().numpy(), int(target)) for bag, target in bags]
        features, labels = zip(*rows, strict=True)
        features, labels = np.asarray(features), np.asarray(labels)
        identities = dataset.df.drop_duplicates("slide_id")
    else:
        patches = cast(ImbalanceDataset, dataset)
        features = bank_index(patches.rows).cpu().numpy()
        labels = np.asarray(patches.get_int_targets())
        identities = patches.df
    evidence = grouped_difficulty(
        features, labels, identities["case_id"].astype(str).to_numpy(), class_names
    )
    evidence["pilot_manifest_sha256"] = compute_data_hash(manifest.to_dict("records"))
    return evidence


def aggregate_difficulty(
    evidence_by_seed: dict[int, dict[str, Any]], native_order: list[str]
) -> dict[str, Any]:
    """Average signed pilot probes and rank classes with native-order ties."""

    def _average(metric: str) -> dict[str, float]:
        return {
            name: float(
                np.mean([item[metric][name] for item in evidence_by_seed.values()])
            )
            for name in native_order
        }

    linear, knn = _average("linear_probe_recall"), _average("knn_recall")
    return {
        "by_construction": {
            str(seed): value for seed, value in evidence_by_seed.items()
        },
        "linear_probe_recall": linear,
        "knn_recall": knn,
        "difficulty": {name: 1.0 - score for name, score in linear.items()},
        "ranking_easiest_to_hardest": sorted(
            native_order,
            key=lambda name: (1.0 - linear[name], native_order.index(name)),
        ),
    }
