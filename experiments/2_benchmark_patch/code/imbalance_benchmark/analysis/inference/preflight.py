from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.inference.bootstrap import (
    build_strata,
    kish_effective_count,
    resample_patient_weights,
)


def _case_positions(case_of_row: np.ndarray, case_pos: dict[Any, int]) -> np.ndarray:
    """Positions into the shared patient-weight matrix for one class's contributing cases."""
    cases = np.unique(case_of_row)
    return np.asarray([case_pos[c] for c in cases], dtype=int)


def _class_preflight(
    positions: np.ndarray,
    patient_weights: np.ndarray,
    n_replicates: int,
) -> dict[str, Any]:
    """Aggregate one class's per-patient weights and summarize them.

    A case's cell weight is its Dirichlet weight alone: the shared crossed
    bootstrap (:class:`BootstrapContext`) draws one weight per case, not one
    per case-row, so a case contributing many rows (patches/slides) never
    outweighs a case contributing one.
    """
    patient_w = patient_weights[positions, :]
    kish = kish_effective_count(patient_w)
    sum_w, max_w = patient_w.sum(axis=0), patient_w.max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        max_frac = np.where(sum_w > 0, max_w / np.maximum(sum_w, 1e-12), 0.0)
    frac_dominant = float(np.mean(max_frac > 0.5))
    mean_kish = float(np.mean(kish))
    # Percentile, not minimum - the minimum keeps falling as replicates grow.
    p2_5_kish = float(np.percentile(kish, 2.5))
    # Bayesian-bootstrap weights are a.s. positive, so this is a structural
    # patient-count check, distinct from Kish's weight-concentration check.
    unique_resampled = float(np.mean((patient_w > 0).sum(axis=0)))
    return {
        "unique_resampled_patients": unique_resampled,
        "kish_effective_count": mean_kish,
        "p2_5_kish_effective_count": p2_5_kish,
        "max_patient_weight_fraction": float(np.mean(max_frac)),
        "frac_replicates_dominant": frac_dominant,
        # Defensive: a.s. positive under the Bayesian bootstrap; guards a
        # future resampling regression, not a real failure today.
        "all_replicates_represented": bool((sum_w > 0).all()),
        "is_descriptive_only": bool(
            p2_5_kish < 5.0 or frac_dominant > 0.05 or unique_resampled < 5.0
        ),
    }


def _diagnostics_by_class(
    identity: pd.DataFrame,
    patient_weights: np.ndarray,
    case_pos: dict[Any, int],
    n_replicates: int,
) -> dict[str, Any]:
    """Calculate aggregate diagnostics for every observed class."""
    return {
        str(label): _class_preflight(
            _case_positions(rows["case_id"].to_numpy(), case_pos),
            patient_weights,
            n_replicates,
        )
        for label, rows in identity.groupby("cancer_type")
    }


def _diagnostics_by_split(
    identity: pd.DataFrame,
    patient_weights: np.ndarray,
    case_pos: dict[Any, int],
    n_replicates: int,
) -> dict[str, dict[str, Any]]:
    """Calculate representation and support diagnostics in every split/class cell."""
    split_col = "patient_split" if "patient_split" in identity else None
    splits = identity[split_col].astype(str).unique() if split_col else ["0"]
    labels = identity["cancer_type"].to_numpy()
    result: dict[str, dict[str, Any]] = {}
    for split in sorted(splits):
        split_mask = _split_mask(identity, split_col, split)
        cell = {
            label: _class_preflight(
                _case_positions(
                    identity.loc[split_mask & (labels == label), "case_id"].to_numpy(),
                    case_pos,
                ),
                patient_weights,
                n_replicates,
            )
            for label in sorted(
                identity.loc[split_mask, "cancer_type"].astype(str).unique()
            )
        }
        for diagnostic in cell.values():
            diagnostic["metric_computable"] = diagnostic["all_replicates_represented"]
        result[str(split)] = cell
    return result


def _split_mask(identity: pd.DataFrame, column: str | None, split: str) -> np.ndarray:
    """Return the rows belonging to one split, or all rows for a single split frame."""
    return (
        identity[column].astype(str).to_numpy() == split
        if column
        else np.ones(len(identity), dtype=bool)
    )


def _multiplicities_match(
    identity: pd.DataFrame, case_ids: np.ndarray, case_pos: dict[Any, int]
) -> bool:
    """Structural guard: every row now looks up its weight by ``case_id`` through
    ``case_pos``, so multiplicity is shared by construction, not by comparison.
    Only the precondition is checked - ``case_ids`` unique and covering ``identity``.
    """
    return len(case_ids) == len(set(case_ids)) and set(identity["case_id"]).issubset(
        set(case_pos)
    )


def _weights_vary(patient_weights: np.ndarray, n_replicates: int) -> bool:
    """True iff every patient's resampled weight actually varies across replicates.

    A Bayesian-bootstrap sanity check: a constant weight would mean the Dirichlet
    draw is not doing its job for that patient (e.g. a regression reintroducing a
    degenerate weight), so this holds for every patient given several replicates.
    """
    if n_replicates <= 1:
        return True
    return bool(np.all(np.var(patient_weights, axis=1) > 0))


def _preflight_result(
    n_replicates: int,
    seed: int,
    by_class: dict[str, Any],
    by_split_class: dict[str, dict[str, Any]],
    multiplicities_match: bool,
    weights_vary: bool,
) -> dict[str, Any]:
    """Assemble the immutable preflight report from its component diagnostics.

    ``is_descriptive_only`` is designated from ``by_class`` - the pooled cell
    over every split appearance a case makes - because that is the cell the
    shared crossed bootstrap (:class:`BootstrapContext`) actually draws
    against. ``by_split_class`` stays in the report and still gates
    ``all_split_level_metrics_computable``, a distinct representation check.
    """
    split_diagnostics = [
        value for split in by_split_class.values() for value in split.values()
    ]
    return {
        "n_replicates": n_replicates,
        "seed": seed,
        "by_class": by_class,
        "by_split_class": by_split_class,
        "all_split_level_metrics_computable": all(
            value["metric_computable"] for value in split_diagnostics
        ),
        "identical_multiplicities_across_split_appearances": multiplicities_match,
        "patient_weights_vary": weights_vary,
        "is_descriptive_only": any(
            value["is_descriptive_only"] for value in by_class.values()
        ),
    }


def run_preflight(
    identity: pd.DataFrame,
    n_replicates: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """By-class preflight over the pooled cell: unique resampled patients, Kish, max weight.

    ``by_class`` pools every case over the union of its split appearances,
    matching the one shared weight :class:`BootstrapContext` draws per case -
    not a per-split cell, which would certify an estimand nobody reports.
    Per report §"Imbalance deficit, recovery, and inference", inference is
    descriptive-only when the pooled cell's 2.5th-percentile Kish is below
    five, one patient supplies over 50% of pooled weight in over 5% of
    replicates, or fewer than five patients contribute."""
    strata = build_strata(identity)
    rng = np.random.default_rng(seed)
    case_ids, patient_weights = resample_patient_weights(strata, n_replicates, rng)
    case_pos = {c: i for i, c in enumerate(case_ids)}
    by_class = _diagnostics_by_class(identity, patient_weights, case_pos, n_replicates)
    by_split_class = _diagnostics_by_split(
        identity, patient_weights, case_pos, n_replicates
    )
    return _preflight_result(
        n_replicates,
        seed,
        by_class,
        by_split_class,
        _multiplicities_match(identity, case_ids, case_pos),
        _weights_vary(patient_weights, n_replicates),
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
