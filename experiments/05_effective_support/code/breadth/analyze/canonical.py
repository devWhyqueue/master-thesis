"""Canonical, split-independent class order and per-split label permutations.

Each split's own ``manifest_freeze.json`` lists its classes in an
independently-generated order, so an integer label stored in one split's fit
run record (encoded via that split's own class order at fit time) does not
mean the same class as the same integer in another split's run record.
Anything that pools or subgroup-selects per-split recall arrays by class name
must first reorder each split's raw class-indexed array into one fixed,
split-independent order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from decodability.evidence import load_freeze_meta

from breadth import exp2_split_paths

__all__ = ["canonical_class_names", "canonical_permutation"]


def canonical_class_names(config: dict[str, Any]) -> list[str]:
    """A fixed, alphabetically-sorted class order, independent of any split."""
    names = load_freeze_meta(exp2_split_paths(config, 0))["class_names"]
    return sorted(names)


def canonical_permutation(
    config: dict[str, Any], split_idx: int, canonical_names: list[str]
) -> np.ndarray:
    """Row-gather index into one split's own class-ordered array.

    For a local (that split's own order) class-indexed array ``arr``,
    ``arr[canonical_permutation(...)][c]`` is the row for canonical class
    ``c`` -- i.e. this reorders ``arr`` from split ``split_idx``'s own class
    order into ``canonical_names``'s order.
    """
    local_names = list(
        load_freeze_meta(exp2_split_paths(config, split_idx))["class_names"]
    )
    return np.array([local_names.index(name) for name in canonical_names])
