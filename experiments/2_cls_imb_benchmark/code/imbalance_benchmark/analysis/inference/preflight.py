from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.inference.bootstrap import (
    _class_preflight,
    _preflight_row_weights,
)


def _diagnostics_by_class(
    identity: pd.DataFrame, weights: np.ndarray, n_replicates: int
) -> dict[str, Any]:
    """Calculate aggregate diagnostics for every observed class."""
    labels = identity["cancer_type"].to_numpy()
    return {
        str(label): _class_preflight(
            rows["case_id"].to_numpy(), weights[labels == label, :], n_replicates
        )
        for label, rows in identity.groupby("cancer_type")
    }


def _diagnostics_by_split(
    identity: pd.DataFrame, weights: np.ndarray, n_replicates: int
) -> dict[str, dict[str, Any]]:
    """Calculate representation and support diagnostics in every split/class cell."""
    split_col = "patient_split" if "patient_split" in identity else None
    splits = identity[split_col].astype(str).unique() if split_col else ["0"]
    labels = identity["cancer_type"].to_numpy()
    result: dict[str, dict[str, Any]] = {}
    for split in sorted(splits):
        split_mask = _split_mask(identity, split_col, split)
        result[str(split)] = {
            label: _split_class_diagnostic(
                identity, weights, labels, split_mask, label, n_replicates
            )
            for label in sorted(
                identity.loc[split_mask, "cancer_type"].astype(str).unique()
            )
        }
    return result


def _split_mask(identity: pd.DataFrame, column: str | None, split: str) -> np.ndarray:
    """Return the rows belonging to one split, or all rows for a single split frame."""
    return (
        identity[column].astype(str).to_numpy() == split
        if column
        else np.ones(len(identity), dtype=bool)
    )


def _split_class_diagnostic(
    identity: pd.DataFrame,
    weights: np.ndarray,
    labels: np.ndarray,
    split_mask: np.ndarray,
    label: str,
    n_replicates: int,
) -> dict[str, Any]:
    """Record one split/class bootstrap diagnostic with representation checks."""
    mask = split_mask & (labels == label)
    diagnostic = _class_preflight(
        identity.loc[mask, "case_id"].to_numpy(), weights[mask, :], n_replicates
    )
    represented = bool((weights[mask, :].sum(axis=0) > 0).all())
    diagnostic.update(
        all_replicates_represented=represented, metric_computable=represented
    )
    return diagnostic


def _multiplicities_match(identity: pd.DataFrame, weights: np.ndarray) -> bool:
    """Check that a patient's resampling multiplicity is shared across appearances."""
    cases = identity["case_id"].astype(str).to_numpy()
    return bool(
        all(
            np.all(
                weights[cases == case, :]
                == weights[np.flatnonzero(cases == case)[0], :]
            )
            for case in np.unique(cases)
        )
    )


def _preflight_result(
    n_replicates: int,
    seed: int,
    by_class: dict[str, Any],
    by_split_class: dict[str, dict[str, Any]],
    multiplicities_match: bool,
) -> dict[str, Any]:
    """Assemble the immutable preflight report from its component diagnostics."""
    diagnostics = [
        value for split in by_split_class.values() for value in split.values()
    ]
    return {
        "n_replicates": n_replicates,
        "seed": seed,
        "by_class": by_class,
        "by_split_class": by_split_class,
        "all_split_level_metrics_computable": all(
            value["metric_computable"] for value in diagnostics
        ),
        "identical_multiplicities_across_split_appearances": multiplicities_match,
        "is_descriptive_only": any(
            value["is_descriptive_only"] for value in diagnostics
        ),
    }


def run_preflight(
    identity: pd.DataFrame,
    n_replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Per split-frame, by-class preflight: unique resampled patients, Kish, max weight.

    Kish and max-weight-fraction are averaged across replicates and flagged
    descriptive-only when the mean Kish count is below five, or when one
    patient supplies more than 50% of a class's weight in more than 5% of
    replicates, per report §"Imbalance deficit, recovery, and inference".
    """
    row_weights = _preflight_row_weights(identity, n_replicates, seed)
    by_class = _diagnostics_by_class(identity, row_weights, n_replicates)
    by_split_class = _diagnostics_by_split(identity, row_weights, n_replicates)
    return _preflight_result(
        n_replicates,
        seed,
        by_class,
        by_split_class,
        _multiplicities_match(identity, row_weights),
    )
