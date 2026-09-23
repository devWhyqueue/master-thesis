"""Cross-fitted class covariates: headroom (r1 recall) and margin, each held out per split.

Split s's covariate value for a class only uses r1 fits of the *other two* splits, so noise in
the r1 baseline used by the outcome (same fits, same split) cannot mechanically correlate with
the covariate. Both covariates are patient-level bootstrap distributions (2000 replicates, the
shared ``breadth.BOOTSTRAP_SEED``), not point estimates, so the WLS in ``classprops.model`` sees
their sampling noise on every replicate column, not just column 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imbalance_benchmark.analysis.inference.context import BootstrapContext

from centre import N_DRAWS, N_SPLITS
from centre.analyze import pooled

from decomposition.model import draw_weights

from sites.recall import contexts

from spectrum import baseline_config

from classprops.pool import FIT_SPLIT, read_test_margins

__all__ = ["cross_fitted_headroom", "cross_fitted_margin", "shared_draw_weight"]

SplitMargins = tuple[np.ndarray, np.ndarray, np.ndarray]


def _exclude_split(w: np.ndarray, split: int) -> np.ndarray:
    """Zero out one split's fit rows so pooling only sees the other two splits."""
    out = w.copy()
    out[FIT_SPLIT == split] = 0.0
    return out


def cross_fitted_headroom(
    r1_stack: np.ndarray, w_draw: np.ndarray, names: list[str]
) -> np.ndarray:
    """(S, C, R): r1 patient-macro recall for class c, pooled over splits != s."""
    n_splits, n_classes = N_SPLITS, len(names)
    out = np.empty((n_splits, n_classes, r1_stack.shape[-1]), dtype=np.float64)
    for s in range(n_splits):
        w_excl = _exclude_split(w_draw, s)
        for ci in range(n_classes):
            out[s, ci] = pooled(r1_stack[:, ci, :], w_excl)
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted median of (N,) values, per (N, R) replicate-weight column."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w, axis=0)
    half = cum[-1] * 0.5
    idx = np.argmax(cum >= half, axis=0)
    return v[idx]


def _class_margin(
    per_split: dict[int, SplitMargins],
    ctxs: dict[int, BootstrapContext],
    others: list[int],
    class_idx: int,
) -> np.ndarray:
    """Weighted-median margin for one class, pooled over the given (other) splits."""
    vals, wts = [], []
    for o in others:
        margins_o, canon_o, row_patient_o = per_split[o]
        mask = canon_o == class_idx
        vals.append(margins_o[mask])
        wts.append(ctxs[o].weights.patient[row_patient_o[mask], :])
    return _weighted_median(np.concatenate(vals), np.concatenate(wts, axis=0))


def cross_fitted_margin(
    config: dict[str, Any], names: list[str], n_replicates: int
) -> np.ndarray:
    """(S, C, R): weighted-median r1 test-patch margin for class c, splits != s pooled.

    Margin is log p_true - max_{d != true} log p_d, patient-bootstrap weighted via the same
    per-split ``BootstrapContext`` used everywhere else in this codebase.
    """
    r1_config = baseline_config(config, "prevalence_outputs")
    ctxs = contexts(r1_config)
    n_classes = len(names)
    per_split = {
        s: read_test_margins(r1_config, names, "r1", s, ctxs[s])
        for s in range(N_SPLITS)
    }
    out = np.empty((N_SPLITS, n_classes, n_replicates), dtype=np.float64)
    for s in range(N_SPLITS):
        others = [o for o in range(N_SPLITS) if o != s]
        for ci in range(n_classes):
            out[s, ci] = _class_margin(per_split, ctxs, others, ci)
    return out


def shared_draw_weight(
    fit_split: np.ndarray, n_replicates: int, seed: int
) -> np.ndarray:
    """(F, R) draw-resampling weight shared across every pool/arm of one dataset."""
    return draw_weights(fit_split, N_DRAWS, n_replicates, np.random.default_rng(seed))
