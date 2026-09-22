"""Canonical tail order (exp-25's pooled r1 recall) and its easy/hard-tail permutations.

The fit stage and the analysis stage both call these two functions, so the class-to-rank
assignment used by the ``easy_r100``/``hard_r100`` fits is fixed before either B fit runs and
is reproduced exactly (not re-derived some other way) during analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from assignment.analyze import _class_recall_stack, _pool
from assignment.properties import class_properties

__all__ = ["tail_order", "order_perm"]


def tail_order(exp25_config: dict[str, Any], names: list[str]) -> list[str]:
    """Canonical class names sorted by pooled exp-25 r1 test recall, ascending."""
    class_acc = _class_recall_stack(exp25_config, names, ("r1",))
    _, _, _, _, r1_own = _pool(class_acc, names)
    properties = class_properties(exp25_config, names, r1_own)
    return sorted(names, key=lambda c: properties[c]["r1_recall"])


def order_perm(order: list[str], split_names: list[str], easy: bool) -> np.ndarray:
    """One split's class-to-rank permutation (``perm[rank] = local class index``, rank 0 = head).

    Easy-tail places the highest-recall class (``order[-1]``) at the last, smallest-count rank;
    hard-tail reverses the order, placing the lowest-recall class there instead.
    """
    ranked = order if easy else list(reversed(order))
    return np.array([split_names.index(name) for name in ranked])
