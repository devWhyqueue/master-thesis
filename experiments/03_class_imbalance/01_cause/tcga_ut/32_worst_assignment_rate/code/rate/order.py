"""Derive the worst / flip / mild rank orders (argmax S_rate / direction control / argmin S_rate)
and their per-split permutations.

Confusion weights are the absolute rate ``w_cd = P(pred d | true c)`` (off-diagonal counts divided
by the class's full row total), not exp-31's row-normalized off-diagonal share -- this decouples w
from h, so a class with almost no errors no longer carries the same total weight as a heavily
confused one. The score formula, search, and ``order_perm`` are exp-31's ``worst`` logic, reused
unchanged (``worst.score``, ``worst.order.order_perm``).

The fit stage and the analysis stage both call ``derive_orders``, so the class-to-rank assignment
used by the ``worst_r100``/``flip_r100``/``mild_r100`` fits is fixed before any of them run and is
reproduced exactly (not re-derived some other way) during analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from assignment.analyze import _class_recall_stack, _pool
from assignment.properties import _confusion_counts, class_properties

from permutation.model import ranked_counts, z_of_counts

from worst import N_RESTARTS, RATIO_SEARCH, SEARCH_SEED
from worst.score import search

__all__ = ["confusion_rates", "derive_orders", "score_inputs"]


def confusion_rates(confusion: np.ndarray) -> np.ndarray:
    """Absolute rate w[c, d] = P(pred d | true c): off-diagonal counts / c's full row total."""
    off = confusion.astype(np.float64).copy()
    np.fill_diagonal(off, 0.0)
    row_sum = confusion.astype(np.float64).sum(axis=1, keepdims=True)
    return np.divide(off, row_sum, out=np.zeros_like(off), where=row_sum > 0)


def score_inputs(
    exp25_config: dict[str, Any], names: list[str], g: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(w, h, z) -- confusion rates, r1-recall headroom, and rank-100 z-profile ``S`` is built from."""
    class_acc = _class_recall_stack(exp25_config, names, ("r1",))
    _, _, _, _, r1_own = _pool(class_acc, names)
    properties = class_properties(exp25_config, names, r1_own)
    w = confusion_rates(_confusion_counts(exp25_config, names))
    h = np.array([properties[c]["r1_recall"] for c in names])
    z = z_of_counts(ranked_counts(RATIO_SEARCH, g, len(names)), g)
    return w, h, z


def derive_orders(
    exp25_config: dict[str, Any], names: list[str], g: int
) -> dict[str, list[str]]:
    """argmax S_rate (worst) and its reversal (flip), argmin S_rate (mild); rank order, index 0 = head."""
    w, h, z = score_inputs(exp25_config, names, g)
    worst_perm = search(w, h, z, maximize=True, seed=SEARCH_SEED, n_restarts=N_RESTARTS)
    mild_perm = search(w, h, z, maximize=False, seed=SEARCH_SEED, n_restarts=N_RESTARTS)
    worst = [names[i] for i in worst_perm]
    mild = [names[i] for i in mild_perm]
    return {"worst": worst, "flip": list(reversed(worst)), "mild": mild}
