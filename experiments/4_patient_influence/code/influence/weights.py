"""Patient-average loss weights and the contribution-inequality audit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["patient_average_weights", "contribution_audit"]


def _group_stats(
    labels: np.ndarray, patients: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Factorize (class, patient) pairs into per-group patch counts.

    Returns ``pair_codes`` (one group code per patch), ``m_ic`` (patch count per
    group), ``class_of_code`` (class of each group), ``n_c`` (patch count per
    class), and ``G_c`` (contributing-patient count per class).
    """
    pairs = np.asarray(
        [f"{lbl}|{p}" for lbl, p in zip(labels.tolist(), patients)], dtype=object
    )
    pair_codes, pair_uniques = pd.factorize(pairs, sort=False)
    m_ic = np.bincount(pair_codes)
    class_of_code = np.asarray(
        [int(key.split("|", 1)[0]) for key in pair_uniques], dtype=np.int64
    )
    n_classes = int(labels.max()) + 1
    n_c = np.bincount(labels, minlength=n_classes)
    g_c = np.bincount(class_of_code, minlength=n_classes)
    return pair_codes, m_ic, class_of_code, n_c, g_c


def _check_invariants(weights: np.ndarray, labels: np.ndarray, n_c: np.ndarray) -> None:
    """Raise if weights are non-positive or do not preserve class/total weight mass."""
    if np.any(weights <= 0.0):
        raise RuntimeError("Non-positive patient-average weight encountered")
    for c, count in enumerate(n_c):
        if count == 0:
            continue
        class_sum = float(np.sum(weights[labels == c]))
        if not np.isclose(class_sum, count, rtol=1e-8, atol=1e-6):
            raise RuntimeError(f"Class {c} weight sum {class_sum} != n_c {count}")
    total = float(np.sum(weights))
    if not np.isclose(total, len(labels), rtol=1e-8, atol=1e-6):
        raise RuntimeError(f"Total weight {total} != N {len(labels)}")


def patient_average_weights(labels: np.ndarray, patients: Sequence[str]) -> np.ndarray:
    """Compute ``w_j = n_c / (G_c * m_ic)`` for each patch (report eq. 1)."""
    labels = np.asarray(labels, dtype=np.int64)
    pair_codes, m_ic, _, n_c, g_c = _group_stats(labels, patients)
    weights = n_c[labels] / (g_c[labels] * m_ic[pair_codes])
    _check_invariants(weights, labels, n_c)
    return weights


def _empty_audit_row(name: str, n_c: int) -> dict[str, Any]:
    """Build a zeroed audit row for a class with no contributing patients."""
    return {
        "class": name,
        "n_c": n_c,
        "G_c": 0,
        "min_m_ic": 0,
        "max_m_ic": 0,
        "max_share": 0.0,
        "D_c": 0.0,
    }


def _audit_row(name: str, n_c: int, g_c: int, m_vals: np.ndarray) -> dict[str, Any]:
    """Build an audit row from one class's per-patient patch counts."""
    shares = m_vals / n_c
    d_c = 0.5 * float(np.sum(np.abs(shares - 1.0 / g_c)))
    return {
        "class": name,
        "n_c": n_c,
        "G_c": g_c,
        "min_m_ic": int(m_vals.min()),
        "max_m_ic": int(m_vals.max()),
        "max_share": float(shares.max()),
        "D_c": d_c,
    }


def contribution_audit(
    labels: np.ndarray, patients: Sequence[str], class_names: Sequence[str]
) -> list[dict[str, Any]]:
    """Report per-class contribution counts and the equal-influence departure D_c."""
    labels = np.asarray(labels, dtype=np.int64)
    _, m_ic, class_of_code, n_c, g_c = _group_stats(labels, patients)
    rows: list[dict[str, Any]] = []
    for c, name in enumerate(class_names):
        if g_c[c] == 0:
            rows.append(_empty_audit_row(name, int(n_c[c])))
            continue
        rows.append(
            _audit_row(name, int(n_c[c]), int(g_c[c]), m_ic[class_of_code == c])
        )
    return rows
