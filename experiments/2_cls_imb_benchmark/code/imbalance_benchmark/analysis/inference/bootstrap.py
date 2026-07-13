from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

__all__ = [
    "build_strata",
    "resample_patient_weights",
    "expand_to_rows",
    "resample_seed_indices",
    "kish_effective_count",
    "bootstrap_preflight",
    "weighted_balanced_accuracy",
    "weighted_macro_nll",
    "gather_seed_resampled",
]


def build_strata(identity: pd.DataFrame) -> pd.Series:
    """Map each patient to its stratum: the sorted tuple of classes it contributes to.

    Patients sharing a stratum contribute the identical set of classes, so
    resampling *within* a stratum can only reshuffle which patient supplies a
    class, never remove a class the stratum contributes — the report's
    "preserving every observed contribution stratum guarantees class
    representation" invariant.
    """
    grouped = identity.groupby("case_id")["cancer_type"].apply(
        lambda s: tuple(sorted(set(s)))
    )
    return cast(pd.Series, grouped)


def resample_patient_weights(
    strata: pd.Series, n_replicates: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Resample patients with replacement within each stratum, vectorized over replicates.

    Returns ``(case_ids, weights)`` where ``weights`` has shape
    ``(n_patients, n_replicates)``; each stratum's column sum is exactly the
    stratum's patient count in every replicate (only the within-stratum
    allocation varies), which is what preserves class representation exactly.
    """
    case_ids = strata.index.to_numpy()
    weights = np.zeros((len(case_ids), n_replicates), dtype=np.int64)
    for _, members in strata.groupby(strata):
        idx = np.flatnonzero(np.isin(case_ids, members.index.to_numpy()))
        m = len(idx)
        draws = rng.multinomial(m, np.full(m, 1.0 / m), size=n_replicates)
        weights[idx, :] = draws.T
    return case_ids, weights


def expand_to_rows(
    case_ids: np.ndarray, patient_weights: np.ndarray, row_case_ids: np.ndarray
) -> np.ndarray:
    """Broadcast each patient's per-replicate weight to all of its rows (slides/patches)."""
    position = {c: i for i, c in enumerate(case_ids)}
    row_idx = np.asarray([position[c] for c in row_case_ids])
    return patient_weights[row_idx, :]


def resample_seed_indices(
    n_seeds: int, n_replicates: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw, per replicate, a size-``n_seeds`` resample (with replacement) of seed indices.

    Report: "the five matched confirmation initialization indices are
    resampled as paired algorithmic-noise blocks and averaged before split
    aggregation" — the same draw is shared by balanced CE, imbalanced CE, and
    the method being compared, since callers pass this same matrix to each.
    """
    return rng.integers(0, n_seeds, size=(n_replicates, n_seeds))


def gather_seed_resampled(
    per_seed_metric: np.ndarray, seed_idx: np.ndarray
) -> np.ndarray:
    """Average a (n_seeds, n_replicates) metric matrix over each replicate's seed resample.

    ``seed_idx`` has shape ``(n_replicates, n_seeds)`` from
    :func:`resample_seed_indices`. Returns shape ``(n_replicates,)``.
    """
    n_replicates = per_seed_metric.shape[1]
    col = np.arange(n_replicates)[:, None]
    gathered = per_seed_metric[seed_idx, col]
    return gathered.mean(axis=1)


def kish_effective_count(weights: np.ndarray) -> np.ndarray:
    """Kish effective count ``(sum w)^2 / sum w^2`` per replicate column."""
    sum_w = weights.sum(axis=0)
    sum_w2 = (weights.astype(np.float64) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sum_w2 > 0, (sum_w**2) / np.maximum(sum_w2, 1e-12), 0.0)
    return out


def weighted_balanced_accuracy(
    labels: np.ndarray, preds: np.ndarray, row_weights: np.ndarray, n_classes: int
) -> np.ndarray:
    """Weighted macro recall (balanced accuracy) per replicate column."""
    n_replicates = row_weights.shape[1]
    out = np.zeros(n_replicates, dtype=np.float64)
    for c in range(n_classes):
        mask = labels == c
        if not mask.any():
            continue
        correct = mask & (preds == c)
        class_weight = row_weights[mask, :].sum(axis=0)
        correct_weight = row_weights[correct, :].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            recall = np.where(
                class_weight > 0, correct_weight / np.maximum(class_weight, 1e-12), 0.0
            )
        out += recall
    return out / n_classes


def weighted_macro_nll(
    labels: np.ndarray,
    probabilities: np.ndarray,
    row_weights: np.ndarray,
    class_subset: list[int],
) -> np.ndarray:
    """Weighted mean NLL for a subset of classes, averaged unweighted across that subset.

    Used both for natural macro NLL (``class_subset`` = all classes) and the
    tail-group macro NLL used by the calibration deficit gate.
    """
    n_replicates = row_weights.shape[1]
    out = np.zeros(n_replicates, dtype=np.float64)
    per_sample_nll = -np.log(
        np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    )
    counted = 0
    for c in class_subset:
        mask = labels == c
        if not mask.any():
            continue
        w = row_weights[mask, :]
        class_weight = w.sum(axis=0)
        weighted_nll = (w * per_sample_nll[mask, None]).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out += np.where(
                class_weight > 0, weighted_nll / np.maximum(class_weight, 1e-12), 0.0
            )
        counted += 1
    return out / max(counted, 1)


def _class_preflight(
    case_of_row: np.ndarray, class_row_weights: np.ndarray, n_replicates: int
) -> dict[str, Any]:
    """Aggregate one class's row weights to per-patient weights and summarize them."""
    unique_cases = np.unique(case_of_row)
    pos = {c: i for i, c in enumerate(unique_cases)}
    idx = np.asarray([pos[c] for c in case_of_row])
    patient_w = np.zeros((len(unique_cases), n_replicates), dtype=np.int64)
    np.add.at(patient_w, idx, class_row_weights)
    kish = kish_effective_count(patient_w)
    sum_w, max_w = patient_w.sum(axis=0), patient_w.max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        max_frac = np.where(sum_w > 0, max_w / np.maximum(sum_w, 1e-12), 0.0)
    frac_dominant = float(np.mean(max_frac > 0.5))
    mean_kish = float(np.mean(kish))
    return {
        "unique_resampled_patients": float(np.mean((patient_w > 0).sum(axis=0))),
        "kish_effective_count": mean_kish,
        "max_patient_weight_fraction": float(np.mean(max_frac)),
        "frac_replicates_dominant": frac_dominant,
        "is_descriptive_only": bool(mean_kish < 5.0 or frac_dominant > 0.05),
    }


def _preflight_row_weights(
    identity: pd.DataFrame, n_replicates: int, seed: int
) -> np.ndarray:
    """Resample the identity frame's patients and broadcast weights back to its rows."""
    strata = build_strata(identity)
    rng = np.random.default_rng(seed)
    case_ids, patient_weights = resample_patient_weights(strata, n_replicates, rng)
    return expand_to_rows(case_ids, patient_weights, identity["case_id"].to_numpy())


def bootstrap_preflight(
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
    class_col = identity["cancer_type"].to_numpy()
    by_class = {
        str(cls): _class_preflight(
            rows["case_id"].to_numpy(), row_weights[class_col == cls, :], n_replicates
        )
        for cls, rows in identity.groupby("cancer_type")
    }
    return {
        "n_replicates": n_replicates,
        "seed": seed,
        "by_class": by_class,
        "is_descriptive_only": any(v["is_descriptive_only"] for v in by_class.values()),
    }
