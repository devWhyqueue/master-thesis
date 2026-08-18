from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "_as_members",
    "_ba_observed",
    "_tail_nll_observed",
]


def _as_members(arm: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
    """Normalize one permutation arm to a list of confirmatory members.

    A bare array is one member; a list is several confirmatory members whose
    per-row contribution is their mean (protocol app:testing's matched-vs-
    unmatched contrast, where either side can average more than one method).
    """
    return arm if isinstance(arm, list) else [arm]


def _case_macro_mean(
    row_values: np.ndarray, case_ids: np.ndarray, mask: np.ndarray
) -> float:
    """Mean of one class's per-case value, each case counted once regardless of row count."""
    per_case = pd.Series(row_values[mask]).groupby(case_ids[mask], sort=False).mean()
    return float(per_case.mean())


def _member_ba_observed(
    labels: np.ndarray, predictions: np.ndarray, case_ids: np.ndarray, n_classes: int
) -> float:
    values = []
    for seed_predictions in predictions:
        value = 0.0
        for class_index in range(n_classes):
            mask = labels == class_index
            if mask.any():
                correct = (seed_predictions == class_index).astype(np.float64)
                value += _case_macro_mean(correct, case_ids, mask)
        values.append(value / n_classes)
    return float(np.mean(values))


def _ba_observed(
    labels: np.ndarray,
    predictions: np.ndarray | list[np.ndarray],
    case_ids: np.ndarray,
    n_classes: int,
) -> float:
    """Mean observed case-macro BA across confirmatory members and confirmation seeds."""
    return float(
        np.mean(
            [
                _member_ba_observed(labels, member, case_ids, n_classes)
                for member in _as_members(predictions)
            ]
        )
    )


def _member_tail_nll_observed(
    labels: np.ndarray,
    probabilities: np.ndarray,
    case_ids: np.ndarray,
    tail_classes: list[int],
) -> float:
    values = []
    for seed_probabilities in probabilities:
        value = 0.0
        counted = 0
        for class_index in tail_classes:
            mask = labels == class_index
            if mask.any():
                nll = -np.log(np.clip(seed_probabilities[:, class_index], 1e-12, 1.0))
                value += _case_macro_mean(nll, case_ids, mask)
                counted += 1
        values.append(value / max(counted, 1))
    return float(np.mean(values))


def _tail_nll_observed(
    labels: np.ndarray,
    probabilities: np.ndarray | list[np.ndarray],
    case_ids: np.ndarray,
    tail_classes: list[int],
) -> float:
    """Mean observed case-macro tail-NLL across confirmatory members and confirmation seeds."""
    return float(
        np.mean(
            [
                _member_tail_nll_observed(labels, member, case_ids, tail_classes)
                for member in _as_members(probabilities)
            ]
        )
    )
