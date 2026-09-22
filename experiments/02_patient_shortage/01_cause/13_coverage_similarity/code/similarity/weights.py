"""Class-stratified multinomial patient weights shared across all three splits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imbalance_benchmark.analysis.inference.bootstrap import PatientWeights
from imbalance_benchmark.analysis.inference.context import _crossed_test_identity

from breadth import exp2_split_paths

from sites.recall import contexts

__all__ = ["weighted_contexts"]


def _class_stratified_weights(
    identity: pd.DataFrame, n_replicates: int, seed: int
) -> tuple[dict[str, int], np.ndarray]:
    """Class-stratified multinomial patient weights, one weight shared across splits."""
    per_patient_classes = identity.groupby("case_id")["cancer_type"].nunique()
    counts = per_patient_classes.to_numpy()
    if (counts > 1).any():
        bad = per_patient_classes.index.to_numpy()[counts > 1].tolist()
        raise ValueError(f"Patient(s) with more than one class in test identity: {bad}")

    case_ids = sorted(identity["case_id"].astype(str).unique())
    position = {c: i for i, c in enumerate(case_ids)}
    patient_class = (
        identity.drop_duplicates("case_id")
        .set_index("case_id")["cancer_type"]
        .astype(str)
    )
    class_of_patient = patient_class.to_numpy()
    patients = patient_class.index.to_numpy()

    weights = np.ones((len(case_ids), n_replicates), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for c_name in sorted(set(class_of_patient.tolist())):
        members = patients[class_of_patient == c_name]
        idxs = np.array([position[c] for c in members])
        n_c = len(idxs)
        draws = rng.multinomial(n_c, np.full(n_c, 1.0 / n_c), size=n_replicates - 1)
        weights[idxs, 1:] = draws.T
    return position, weights


def _shared_weights(
    config: dict[str, Any], n_replicates: int, seed: int
) -> tuple[dict[str, int], np.ndarray]:
    identity = _crossed_test_identity(exp2_split_paths(config, 0), is_mil=False)
    return _class_stratified_weights(identity, n_replicates, seed)


def weighted_contexts(
    config: dict[str, Any], n_replicates: int, seed: int
) -> dict[int, Any]:
    """Every split's BootstrapContext with its weights replaced by the shared patient weights."""
    ctxs = contexts(config)
    position, weights = _shared_weights(config, n_replicates, seed)
    for ctx in ctxs.values():
        row_patient = np.array([position[c] for c in ctx.case_ids])
        ctx.weights = PatientWeights(row_patient, weights)
        ctx.n_replicates = weights.shape[1]
    return ctxs
