"""Patient-mean Virchow2 embeddings and the cosine coverage distance (Eq. distance)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from imbalance_benchmark.datasets.feature_provenance import load_stored_feature_tensor

__all__ = [
    "PatientClassKey",
    "patient_class_means",
    "assert_consistent_patch_counts",
    "training_mu",
    "embed",
    "cosine_distances",
]

PatientClassKey = tuple[str, str]


def _accumulate_group(
    tensor: np.ndarray,
    group: pd.DataFrame,
    has_index: bool,
    sums: dict[PatientClassKey, np.ndarray],
    counts: dict[PatientClassKey, int],
) -> None:
    """Add one slide tensor's rows to the running per-(patient, class) sums."""
    indices = group["feature_index"].to_numpy() if has_index else None
    cases = group["case_id"].astype(str).to_numpy()
    classes = group["cancer_type"].astype(str).to_numpy()
    for row in range(len(group)):
        idx = int(indices[row]) if indices is not None and pd.notna(indices[row]) else 0
        key = (cases[row], classes[row])
        if key not in sums:
            sums[key] = tensor[idx].copy()
            counts[key] = 1
        else:
            sums[key] += tensor[idx]
            counts[key] += 1


def patient_class_means(
    manifest: pd.DataFrame,
) -> tuple[dict[PatientClassKey, np.ndarray], dict[PatientClassKey, int]]:
    """Mean feature vector and patch count of every (patient, class) in a manifest.

    Groups rows by ``feature_path`` so each slide tensor is loaded once, and
    accumulates float64 sums to keep the mean numerically stable.
    """
    has_index = "feature_index" in manifest.columns
    sums: dict[PatientClassKey, np.ndarray] = {}
    counts: dict[PatientClassKey, int] = {}
    for feature_path, group in manifest.groupby("feature_path", sort=False):
        tensor = load_stored_feature_tensor(str(feature_path)).double().numpy()
        _accumulate_group(tensor, group, has_index, sums, counts)
    return {key: value / counts[key] for key, value in sums.items()}, counts


def assert_consistent_patch_counts(
    reference: dict[PatientClassKey, int], manifest: pd.DataFrame
) -> None:
    """Verify a split's (patient, class) patch counts match the reference manifest.

    The three splits are patient-disjoint relabellings of the same underlying
    patch pool, so patient-class means computed from one split's manifest are
    valid for every split -- but only if this invariant actually holds.
    """
    counts = Counter(
        zip(
            manifest["case_id"].astype(str),
            manifest["cancer_type"].astype(str),
        )
    )
    for key, expected in reference.items():
        actual = counts[key]
        if actual != expected:
            raise RuntimeError(
                f"Patch count for patient {key[0]}, class {key[1]} differs across "
                f"splits ({actual} != {expected}); patient-class means are not "
                "split-invariant."
            )


def training_mu(
    means: dict[PatientClassKey, np.ndarray], train_df: pd.DataFrame
) -> np.ndarray:
    """Mean of the training (patient, class) means for one split (mu_r)."""
    keys = {
        (str(case), str(cls))
        for case, cls in zip(train_df["case_id"], train_df["cancer_type"])
    }
    vectors = [means[key] for key in keys if key in means]
    if not vectors:
        raise ValueError("No training patient-class means found to average for mu")
    return np.mean(vectors, axis=0)


def embed(
    means: dict[PatientClassKey, np.ndarray], mu: np.ndarray
) -> dict[PatientClassKey, np.ndarray]:
    """Centre on mu and L2-normalise every patient-class mean (Eq. distance)."""
    out: dict[PatientClassKey, np.ndarray] = {}
    for key, vec in means.items():
        centred = vec - mu
        norm = float(np.linalg.norm(centred))
        if norm == 0.0:
            raise ValueError(f"Zero-norm embedding for {key}; cannot normalise")
        out[key] = centred / norm
    return out


def cosine_distances(e_a: np.ndarray, e_b: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance between two embedding matrices (Eq. distance)."""
    return 1.0 - e_a @ e_b.T
