from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.inference.bootstrap import (
    build_strata,
    expand_to_rows,
    kish_effective_count,
    resample_patient_weights,
)


def _class_preflight(
    case_of_row: np.ndarray, class_row_weights: np.ndarray, n_replicates: int
) -> dict[str, Any]:
    """Aggregate one class's row weights to per-patient weights and summarize them."""
    unique_cases = np.unique(case_of_row)
    pos = {c: i for i, c in enumerate(unique_cases)}
    idx = np.asarray([pos[c] for c in case_of_row])
    patient_w = np.zeros((len(unique_cases), n_replicates), dtype=np.float64)
    np.add.at(patient_w, idx, class_row_weights)
    kish = kish_effective_count(patient_w)
    sum_w, max_w = patient_w.sum(axis=0), patient_w.max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        max_frac = np.where(sum_w > 0, max_w / np.maximum(sum_w, 1e-12), 0.0)
    frac_dominant = float(np.mean(max_frac > 0.5))
    mean_kish = float(np.mean(kish))
    # Percentile, not minimum: the minimum keeps falling as `n_replicates` grows,
    # so a floor on it would measure the replicate budget, not the cell.
    p2_5_kish = float(np.percentile(kish, 2.5))
    # Under the Bayesian bootstrap every patient's weight is a.s. positive in
    # every replicate, so this is really a structural patient-count check
    # (distinct from Kish's weight-concentration check), not a resampling
    # artifact the way it was under the retired multinomial draw.
    unique_resampled = float(np.mean((patient_w > 0).sum(axis=0)))
    return {
        "unique_resampled_patients": unique_resampled,
        "kish_effective_count": mean_kish,
        "p2_5_kish_effective_count": p2_5_kish,
        "max_patient_weight_fraction": float(np.mean(max_frac)),
        "frac_replicates_dominant": frac_dominant,
        "is_descriptive_only": bool(
            p2_5_kish < 5.0 or frac_dominant > 0.05 or unique_resampled < 5.0
        ),
    }


def _preflight_row_weights(
    identity: pd.DataFrame, n_replicates: int, seed: int
) -> np.ndarray:
    """Resample the identity frame's patients and broadcast weights back to its rows."""
    strata = build_strata(identity)
    rng = np.random.default_rng(seed)
    case_ids, patient_weights = resample_patient_weights(strata, n_replicates, rng)
    return expand_to_rows(case_ids, patient_weights, identity["case_id"].to_numpy())


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
    # Defensive: under the Bayesian bootstrap every weight is a.s. positive, so
    # this is always True; kept as a guard against a future resampling change.
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


def _weights_vary(
    identity: pd.DataFrame, weights: np.ndarray, n_replicates: int
) -> bool:
    """True iff every patient's resampled weight actually varies across replicates.

    A Bayesian-bootstrap sanity check: a constant weight would mean the Dirichlet
    draw is not doing its job for that patient (e.g. a regression reintroducing a
    degenerate weight), so this holds for every patient given several replicates.
    """
    if n_replicates <= 1:
        return True
    cases = identity["case_id"].to_numpy()
    _, first_row_of_case = np.unique(cases, return_index=True)
    patient_w = weights[first_row_of_case, :]
    return bool(np.all(np.var(patient_w, axis=1) > 0))


def _preflight_result(
    n_replicates: int,
    seed: int,
    by_class: dict[str, Any],
    by_split_class: dict[str, dict[str, Any]],
    multiplicities_match: bool,
    weights_vary: bool,
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
        "patient_weights_vary": weights_vary,
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

    Mean Kish and max-weight fractions are reported across replicates. Per report
    §"Imbalance deficit, recovery, and inference", inference is descriptive-only
    when the 2.5th percentile of the Kish count is below five, one patient
    supplies over 50% of a class's weight in over 5% of replicates, or a
    split/class cell has fewer than five contributing patients.
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
        _weights_vary(identity, row_weights, n_replicates),
    )


_VALIDITY_CHECKS = {
    "all_split_level_metrics_computable": (
        "a split/class cell is unrepresented in some replicate"
    ),
    "identical_multiplicities_across_split_appearances": (
        "a split unit received different multiplicities across its split appearances"
    ),
    "patient_weights_vary": (
        "a patient's resampled weight does not vary across replicates"
    ),
}


def require_valid_preflight(preflight: dict[str, Any]) -> None:
    """Raise when the label-only resampling scheme is itself invalid.

    Report §"Uncertainty from split-unit resampling": failure of a preflight
    check stops the analysis rather than silently discarding replicates. Weight
    concentration is different in kind: it only designates the dataset--regime
    descriptive, which ``is_descriptive_only`` carries forward instead.
    """
    failures = [
        reason
        for key, reason in _VALIDITY_CHECKS.items()
        if not preflight.get(key, False)
    ]
    if failures:
        raise RuntimeError(
            "Label-only bootstrap preflight failed: "
            + "; ".join(failures)
            + ". Fix the resampling scheme before freezing; replicates must not "
            "be discarded to make it pass."
        )


def bootstrap_preflight(
    identity: pd.DataFrame, n_replicates: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    """Run the label-only bootstrap feasibility diagnostic."""
    return run_preflight(identity, n_replicates, seed)
