from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from imbalance_benchmark.analysis.metrics import confidence_bin_index

__all__ = [
    "build_strata",
    "resample_patient_weights",
    "expand_to_rows",
    "resample_seed_indices",
    "kish_effective_count",
    "case_class_divisor",
    "PatientWeights",
    "weighted_balanced_accuracy",
    "weighted_macro_nll",
    "weighted_ece",
    "gather_seed_resampled",
]


@dataclass(frozen=True)
class PatientWeights:
    """Bootstrap weights held per patient; rows of a patient always share a weight.

    ``sum_rows w(row)*v(row)`` equals ``sum_patients w(patient)*(sum-of-patient
    v(row))``, so bincounting to patients once then matmul-ing the weight
    matrix reproduces the row-expanded result at a fraction of the cost.
    """

    row_patient: np.ndarray  # (n_rows,) index into `patient`
    patient: np.ndarray  # (n_patients, n_replicates) float64

    @property
    def n_replicates(self) -> int:
        """Number of bootstrap replicate columns (including the observed one)."""
        return self.patient.shape[1]

    def sums(
        self, values: np.ndarray | float, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Sum over rows of ``weight(row) * values(row)`` per replicate.
        ``values`` may be scalar (all-ones) or per-row; ``mask`` restricts the sum.
        """
        row_patient = self.row_patient if mask is None else self.row_patient[mask]
        if np.isscalar(values):
            weights = None if values == 1.0 else np.full(len(row_patient), values)
        else:
            weights = np.asarray(values) if mask is None else np.asarray(values)[mask]
        per_patient = np.bincount(
            row_patient, weights=weights, minlength=len(self.patient)
        )
        return per_patient @ self.patient

    def class_sums(
        self, values: np.ndarray | float, codes: np.ndarray, n_codes: int
    ) -> np.ndarray:
        """Row sums split by ``codes``: returns ``(n_codes, n_replicates)`` via one
        bincount over a flattened ``(code, patient)`` index plus a single GEMM --
        reads the weight matrix once, not ``n_codes`` times.
        """
        n_patients = len(self.patient)
        if np.isscalar(values):
            weights = None if values == 1.0 else np.full(len(codes), values)
        else:
            weights = np.asarray(values)
        flat_index = codes.astype(np.int64) * n_patients + self.row_patient
        per_cell = np.bincount(
            flat_index, weights=weights, minlength=n_codes * n_patients
        )
        return per_cell.reshape(n_codes, n_patients) @ self.patient


def _contribution_vector(
    rows: pd.DataFrame, split_col: str | None
) -> tuple[tuple[str, str, int], ...]:
    """Encode one patient's observed split-by-class row counts deterministically."""
    splits = (
        rows[split_col].astype(str) if split_col else pd.Series("0", index=rows.index)
    )
    counts = pd.DataFrame(
        {"split": splits, "class": rows["cancer_type"].astype(str)}
    ).value_counts()
    records = []
    for key, count in counts.sort_index().items():
        split, cls = cast(tuple[Any, Any], key)
        records.append((str(split), str(cls), int(count)))
    return tuple(records)


def build_strata(identity: pd.DataFrame) -> pd.Series:
    """Map each patient to its complete split-by-class contribution stratum.
    ``patient_split``, when present, keeps one resampling multiplicity per
    patient across all three fixed split repetitions.
    """
    split_col = "patient_split" if "patient_split" in identity else None

    columns = ["cancer_type"] + ([split_col] if split_col else [])
    grouped = identity.groupby("case_id", sort=True)[columns].apply(
        _contribution_vector, split_col=split_col
    )
    return cast(pd.Series, grouped)


def resample_patient_weights(
    strata: pd.Series, n_replicates: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Bayesian bootstrap (Rubin 1981): independent Dirichlet weight per patient.
    Returns ``(case_ids, weights)``, shape ``(n_patients, n_replicates)``. Each
    patient's almost-surely-positive draw keeps variance nonzero for a unique
    contribution pattern. ``strata`` supplies only the patient index (grouping
    now feeds :mod:`preflight`, not the draw itself).
    """
    case_ids = strata.index.to_numpy()
    n = len(case_ids)
    weights = n * rng.dirichlet(np.ones(n), size=n_replicates).T
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
    Shared by balanced CE, imbalanced CE, and the compared method, so the five
    confirmation seeds resample as one paired block, not independently per arm.
    """
    return rng.integers(0, n_seeds, size=(n_replicates, n_seeds))


def gather_seed_resampled(
    per_seed_metric: np.ndarray, seed_idx: np.ndarray
) -> np.ndarray:
    """Average a (n_seeds, n_replicates) metric matrix over each replicate's seed resample.
    ``seed_idx`` is ``(n_replicates, n_seeds)`` from :func:`resample_seed_indices`.
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


def case_class_divisor(case_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-row ``1 / (rows of this case within this class)`` - the case-macro weight.
    ponytail: assumes rows exchangeable within one class; grouped by (case, class).
    """
    key = pd.DataFrame({"case": case_ids, "label": labels})
    counts = key.groupby(["case", "label"])["case"].transform("size").to_numpy()
    return 1.0 / counts.astype(np.float64)


def weighted_balanced_accuracy(
    labels: np.ndarray,
    preds: np.ndarray,
    weights: PatientWeights,
    n_classes: int,
    case_divisor: np.ndarray,
) -> np.ndarray:
    """Case-macro balanced accuracy: a case's rows count once per class."""
    out = np.zeros(weights.n_replicates, dtype=np.float64)
    for c in range(n_classes):
        mask = labels == c
        if not mask.any():
            continue
        correct = mask & (preds == c)
        class_weight = weights.sums(case_divisor, mask)
        correct_weight = weights.sums(case_divisor, correct)
        with np.errstate(divide="ignore", invalid="ignore"):
            recall = np.where(
                class_weight > 0, correct_weight / np.maximum(class_weight, 1e-12), 0.0
            )
        out += recall
    return out / n_classes


def weighted_macro_nll(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: PatientWeights,
    class_subset: list[int],
    case_divisor: np.ndarray,
) -> np.ndarray:
    """Case-macro mean NLL for a class subset (natural macro NLL, or tail-group calibration)."""
    out = np.zeros(weights.n_replicates, dtype=np.float64)
    per_sample_nll = -np.log(
        np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    )
    counted = 0
    for c in class_subset:
        mask = labels == c
        if not mask.any():
            continue
        class_weight = weights.sums(case_divisor, mask)
        weighted_nll = weights.sums(case_divisor * per_sample_nll, mask)
        with np.errstate(divide="ignore", invalid="ignore"):
            out += np.where(
                class_weight > 0, weighted_nll / np.maximum(class_weight, 1e-12), 0.0
            )
        counted += 1
    return out / max(counted, 1)


def weighted_ece(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: PatientWeights,
    n_bins: int = 10,
) -> np.ndarray:
    """Fixed-binning ECE per replicate under the shared crossed patient weights."""
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == labels).astype(np.float64)
    bin_of_row = confidence_bin_index(confidence, n_bins)
    total = weights.sums(1.0)
    out = np.zeros(weights.n_replicates, dtype=np.float64)
    for b in range(n_bins):
        mask = bin_of_row == b
        if not mask.any():
            continue
        bin_weight = weights.sums(1.0, mask)
        acc = weights.sums(correct, mask)
        conf = weights.sums(confidence, mask)
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.where(
                bin_weight > 0, np.abs(acc - conf) / np.maximum(bin_weight, 1e-12), 0.0
            )
            out += np.where(total > 0, gap * bin_weight / np.maximum(total, 1e-12), 0.0)
    return out
