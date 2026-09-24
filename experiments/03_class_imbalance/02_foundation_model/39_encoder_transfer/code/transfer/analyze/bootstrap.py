"""Shared evaluation-patient bootstrap context for exp-39's analyze stage.

Built from exp-2's own test-patient identity (encoder-independent: both manifests
share the same rows/order, phase 03), with exp-39's own frozen seed and replicate
count (``transfer.BOOTSTRAP_SEED``/``N_REPLICATES``, protocol phase 01), so the same
patient-cluster resample applies identically across both encoders' arms.

``BootstrapContext`` draws continuous (Bayesian-bootstrap) Dirichlet weights per
patient, so every observed row's class keeps strictly positive replicate weight
whenever it has at least one observed test row (``PatientWeights.sums`` sums weights
over rows that exist); an undefined-recall replicate would need a class with zero
observed test rows, which the frozen 3-split design excludes for every scheduled
class. The "redraw and record the rejection count" rule (protocol phase 01) is
therefore satisfied structurally, not by a runtime redraw loop; ``rejected_replicates``
below stays 0 and is recorded for that reason rather than left unaudited.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.common import N_PATIENT_SPLITS

from breadth import exp2_split_paths

from decomposition.model import draw_weights

from transfer import BOOTSTRAP_SEED, N_REPLICATES

__all__ = ["contexts", "class_support_rejections", "encoder_draw_weights"]


def contexts(config: dict[str, Any]) -> dict[int, BootstrapContext]:
    """One shared BootstrapContext per split, seeded for exp-39's own protocol."""
    return {
        s: BootstrapContext(
            exp2_split_paths(config, s),
            is_mil=False,
            n_replicates=N_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
        for s in range(N_PATIENT_SPLITS)
    }


def class_support_rejections(labels: np.ndarray, n_classes: int) -> int:
    """Count of classes with zero observed test rows in one fit's stored labels.

    Every class-weight sum in :class:`BootstrapContext` is a sum of strictly
    positive Dirichlet weights over the rows observed for that class, so it can
    only vanish (undefined recall) when a class has no observed test row at
    all -- a data-completeness fact independent of any bootstrap replicate, not
    something a redraw could fix. The module docstring records why the
    protocol's redraw rule reduces to this structural check.
    """
    return int(sum(1 for c in range(n_classes) if not (labels == c).any()))


def encoder_draw_weights(n_draws: int, n_replicates: int) -> np.ndarray:
    """Frequency weights (F, R) resampling training draws within each split.

    Shared across both encoders and every arm by construction: this draws once
    from ``transfer.BOOTSTRAP_SEED`` and the same weight matrix multiplies every
    encoder's/arm's per-fit accuracy in ``transfer.analyze``.
    """
    fit_split = np.repeat(np.arange(N_PATIENT_SPLITS), n_draws)
    return draw_weights(
        fit_split, n_draws, n_replicates, np.random.default_rng(BOOTSTRAP_SEED)
    )
