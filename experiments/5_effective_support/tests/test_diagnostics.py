"""Checks for temperature rescaling and the site-coverage decomposition."""

from __future__ import annotations

import numpy as np
from imbalance_benchmark.analysis.calibration import softmax

from breadth.analyze.diagnostics import (
    equal_budget_decomposition,
    patient_class_pairs,
    tissue_source_site,
)
from breadth.calibrate import scale_stored_probabilities


def test_stored_probabilities_rescale_like_logits():
    """Scaling stored softmax outputs equals scaling the logits, even far tails."""
    logits = np.random.default_rng(0).normal(scale=40.0, size=(50, 6))
    scaled = scale_stored_probabilities(softmax(logits), 2.5)
    assert np.allclose(scaled, softmax(logits / 2.5), atol=1e-12)


def test_site_transitions_sum_to_patient_macro_difference():
    """The four transition contributions add up to the balanced-accuracy gap."""
    rng = np.random.default_rng(1)
    cases = np.array([f"TCGA-{s}-{i:04d}" for i in range(40) for s in ["AA"]])[
        rng.integers(0, 40, 400)
    ]
    labels = np.array([int(c[-1]) % 3 for c in cases])
    broad = patient_class_pairs(cases, labels, np.where(rng.random(400) < 0.7, labels, -1))
    deep = patient_class_pairs(cases, labels, np.where(rng.random(400) < 0.5, labels, -1))

    parts = equal_budget_decomposition(
        broad, rng.random(len(broad)) < 0.5, deep, rng.random(len(deep)) < 0.5
    )

    gap = 100.0 * (broad["weight"] * (broad["recall"] - deep["recall"])).sum()
    assert np.isclose(sum(p["contribution"] for p in parts.values()), gap)
    assert np.isclose(sum(p["share"] for p in parts.values()), 1.0)
    assert np.isclose(broad["weight"].sum(), 1.0)
    assert tissue_source_site("TCGA-OR-A5J1") == "OR"
    assert tissue_source_site("170") is None
