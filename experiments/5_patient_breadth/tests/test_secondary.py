"""Unit tests for the secondary endpoint distributions and their tables."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from imbalance_benchmark.analysis.inference.bootstrap import (
    PatientWeights,
    case_class_divisor,
    weighted_balanced_accuracy,
)

from breadth.analyze.tables import build_class_recall_table, build_secondary_table
from breadth.analyze.secondary import (
    _patch_micro_balanced_accuracy,
    _patient_macro_recalls,
)


def _context(case_ids: np.ndarray, labels: np.ndarray, n_replicates: int = 4):
    """Build a minimal stand-in carrying the fields the endpoint helpers read."""
    rng = np.random.default_rng(0)
    unique_cases = list(dict.fromkeys(case_ids.tolist()))
    row_patient = np.array([unique_cases.index(c) for c in case_ids])
    observed = np.ones((len(unique_cases), 1))
    resampled = rng.integers(0, 3, size=(len(unique_cases), n_replicates - 1)).astype(
        float
    )
    weights = PatientWeights(row_patient, np.concatenate([observed, resampled], axis=1))
    return SimpleNamespace(
        weights=weights,
        case_class_divisor=case_class_divisor(case_ids, labels.astype(str)),
        n_replicates=weights.n_replicates,
    )


def test_class_recalls_average_to_balanced_accuracy():
    """Per-class patient-macro recalls decompose the primary endpoint exactly."""
    rng = np.random.default_rng(7)
    n_rows, n_classes = 60, 3
    labels = rng.integers(0, n_classes, size=n_rows)
    preds = np.where(rng.random(n_rows) < 0.6, labels, (labels + 1) % n_classes)
    case_ids = np.array([f"case_{i // 4}" for i in range(n_rows)])
    ctx = _context(case_ids, labels)

    recalls = _patient_macro_recalls(ctx, labels, preds, n_classes)
    balanced = weighted_balanced_accuracy(
        labels, preds, ctx.weights, n_classes, ctx.case_class_divisor
    )

    assert recalls.shape == (n_classes, ctx.n_replicates)
    assert np.allclose(recalls.mean(axis=0), balanced)


def test_patch_micro_balanced_accuracy_matches_unweighted_macro_recall():
    """With unit weights the patch-micro endpoint is plain per-class patch recall."""
    labels = np.array([0, 0, 0, 1, 1, 2])
    preds = np.array([0, 0, 1, 1, 2, 2])
    case_ids = np.array(["a", "a", "b", "b", "c", "c"])
    ctx = _context(case_ids, labels, n_replicates=1)

    value = _patch_micro_balanced_accuracy(ctx, labels, preds, 3)

    assert np.isclose(value[0], np.mean([2 / 3, 1 / 2, 1.0]))


def test_secondary_tables_render():
    """Both secondary tables emit complete LaTeX tabulars."""
    est = {"point": 1.0, "ci_2_5": 0.5, "ci_97_5": 1.5}
    scalars = {
        "macro_nll": est,
        "expected_calibration_error": est,
        "patch_micro_balanced_accuracy": est,
    }
    secondary = {
        "cells": {f"G{g}_m{m}": scalars for g in (5, 10, 20) for m in (8, 16, 32)},
        "equal_budget_X": scalars,
        "class_recalls": {
            "tumor": {"G5_m32": est, "G20_m8": est, "equal_budget_X": est}
        },
    }

    endpoint_tex = build_secondary_table(secondary)
    recall_tex = build_class_recall_table(secondary)

    assert endpoint_tex.count(r"\\") == 11
    assert r"Macro NLL (nats)" in endpoint_tex
    assert r"\end{tabular}" in recall_tex
    assert "tumor" in recall_tex
