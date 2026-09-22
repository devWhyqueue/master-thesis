"""Derive the sep / flip / tog rank orders (worst / direction-control / mildest) and their
per-split permutations.

The fit stage and the analysis stage both call ``derive_orders``, so the class-to-rank assignment
used by the ``sep_r100``/``flip_r100``/``tog_r100``/``sep_r10`` fits is fixed before any of them
run and is reproduced exactly (not re-derived some other way) during analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from assignment.analyze import _class_recall_stack, _pool
from assignment.properties import _confusion_counts, class_properties

from permutation.model import ranked_counts, z_of_counts

from worst import N_RESTARTS, RATIO_SEARCH, SEARCH_SEED
from worst.score import confusion_weights, search

__all__ = ["derive_orders", "order_perm", "score_inputs"]


def score_inputs(
    exp25_config: dict[str, Any], names: list[str], g: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(w, h, z) -- confusion weights, r1-recall headroom, and rank-100 z-profile ``S`` is built from."""
    class_acc = _class_recall_stack(exp25_config, names, ("r1",))
    _, _, _, _, r1_own = _pool(class_acc, names)
    properties = class_properties(exp25_config, names, r1_own)
    w = confusion_weights(_confusion_counts(exp25_config, names))
    h = np.array([properties[c]["r1_recall"] for c in names])
    z = z_of_counts(ranked_counts(RATIO_SEARCH, g, len(names)), g)
    return w, h, z


def derive_orders(
    exp25_config: dict[str, Any], names: list[str], g: int
) -> dict[str, list[str]]:
    """argmax S (sep) and its reversal (flip), argmin S (tog); rank order, index 0 = head."""
    w, h, z = score_inputs(exp25_config, names, g)
    sep_perm = search(w, h, z, maximize=True, seed=SEARCH_SEED, n_restarts=N_RESTARTS)
    tog_perm = search(w, h, z, maximize=False, seed=SEARCH_SEED, n_restarts=N_RESTARTS)
    sep = [names[i] for i in sep_perm]
    tog = [names[i] for i in tog_perm]
    return {"sep": sep, "flip": list(reversed(sep)), "tog": tog}


def order_perm(rank_order: list[str], split_names: list[str]) -> np.ndarray:
    """One split's class-to-rank permutation (``perm[rank] = local class index``, rank 0 = head)."""
    return np.array([split_names.index(name) for name in rank_order])
