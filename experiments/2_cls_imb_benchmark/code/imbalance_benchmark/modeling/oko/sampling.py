from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def build_class_index(labels: np.ndarray) -> dict[int, list[int]]:
    """Map each class label to the dataset indices that carry it."""
    index: dict[int, list[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        index.setdefault(int(label), []).append(idx)
    return index


def _independent_units(dataset: Any) -> np.ndarray | None:
    """Independent-unit id (patient, or slide) per patch, aligned to dataset order."""
    frame = getattr(dataset, "df", None)
    if frame is None or "case_id" not in getattr(frame, "columns", []):
        return None
    return frame["case_id"].to_numpy()


def _sample_distinct_odd_classes(
    pair_classes: np.ndarray, n_classes: int, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample k distinct odd classes per set (report requirement) from [C] \\ {pair}."""
    out = np.empty((len(pair_classes), k), dtype=np.int64)
    for row, pair in enumerate(pair_classes.tolist()):
        others = np.array([c for c in range(n_classes) if c != pair])
        out[row] = rng.choice(others, size=k, replace=False)
    return out


def _fill_column(
    class_arrays: dict[int, np.ndarray],
    classes: np.ndarray,
    set_indices: np.ndarray,
    columns: list[int],
    rng: np.random.Generator,
) -> None:
    """Fill the given columns of set_indices with samples drawn per assigned class."""
    for c, pool_arr in class_arrays.items():
        mask = classes == c
        count = int(mask.sum())
        if count == 0:
            continue
        for column in columns:
            drawn = rng.integers(len(pool_arr), size=count)
            set_indices[mask, column] = pool_arr[drawn]


def _class_unit_pools(
    class_index: dict[int, list[int]], units: np.ndarray
) -> dict[int, dict[Any, list[int]]]:
    """Group each class's example indices by their independent unit (patient/slide)."""
    pools: dict[int, dict[Any, list[int]]] = {}
    for class_id, indices in class_index.items():
        by_unit: dict[Any, list[int]] = {}
        for idx in indices:
            by_unit.setdefault(units[idx], []).append(idx)
        pools[class_id] = by_unit
    return pools


@dataclass
class OkoPools:
    """Step-invariant sampling structures derived once from class_index/units per fit."""

    class_arrays: dict[int, np.ndarray]
    unit_pools: dict[int, dict[Any, list[int]]] | None


def _build_oko_pools(
    class_index: dict[int, list[int]], units: np.ndarray | None
) -> OkoPools:
    """Precompute the per-class ndarray pools and unit groupings once per fit."""
    return OkoPools(
        class_arrays={c: np.array(pool) for c, pool in class_index.items()},
        unit_pools=_class_unit_pools(class_index, units) if units is not None else None,
    )


def _draw_distinct_unit_pair(
    by_unit: dict[Any, list[int]], rng: np.random.Generator
) -> tuple[int, int]:
    """Draw two same-class examples from two distinct independent units."""
    unit_keys = list(by_unit)
    first_unit, second_unit = rng.choice(len(unit_keys), size=2, replace=False)
    first = rng.choice(by_unit[unit_keys[first_unit]])
    second = rng.choice(by_unit[unit_keys[second_unit]])
    return int(first), int(second)


def _fill_distinct_pair_indices(
    class_index: dict[int, list[int]],
    pair_classes: np.ndarray,
    set_indices: np.ndarray,
    rng: np.random.Generator,
    unit_pools: dict[int, dict[Any, list[int]]] | None,
) -> None:
    """Fill the two same-class positions from two distinct independent units.

    The report requires class-aware batches to hold two distinct same-class
    *independent units* (patients/slides), not merely two distinct example
    indices which could be two patches from the same patient/slide. When no unit
    pool map is supplied the two examples are only guaranteed distinct.
    """
    for class_id, pool in class_index.items():
        rows = np.flatnonzero(pair_classes == class_id)
        if not len(rows):
            continue
        if unit_pools is not None:
            by_unit = unit_pools[class_id]
            if len(by_unit) < 2:
                raise ValueError(
                    "OKO requires two distinct same-class independent units"
                )
            for row in rows:
                set_indices[row, :2] = _draw_distinct_unit_pair(by_unit, rng)
            continue
        if len(pool) < 2:
            raise ValueError("OKO requires two distinct examples in every pair class")
        for row in rows:
            set_indices[row, :2] = rng.choice(pool, size=2, replace=False)


def sample_oko_sets(
    class_index: dict[int, list[int]],
    n_classes: int,
    n_sets: int,
    k: int,
    rng: np.random.Generator,
    units: np.ndarray | None = None,
    pools: OkoPools | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample n_sets odd-k-out sets (Algorithm 1, Muttenthaler et al. 2024).

    Returns (pair_classes, set_indices of shape (n_sets, k+2), first_odd_classes);
    the auxiliary loss uses only the first odd slot. With ``units``, each set's
    two same-class examples come from distinct independent units. ``pools`` lets
    a caller that already built :func:`_build_oko_pools` skip rebuilding the
    step-invariant per-class arrays and unit groupings; when omitted they are
    built from ``class_index``/``units`` as before.
    """
    if pools is None:
        pools = _build_oko_pools(class_index, units)
    pair_classes = rng.integers(n_classes, size=n_sets)
    set_indices = np.empty((n_sets, k + 2), dtype=np.int64)
    _fill_distinct_pair_indices(
        class_index, pair_classes, set_indices, rng, pools.unit_pools
    )
    odd_classes = _sample_distinct_odd_classes(pair_classes, n_classes, k, rng)
    for slot in range(k):
        _fill_column(
            pools.class_arrays, odd_classes[:, slot], set_indices, [2 + slot], rng
        )
    return pair_classes, set_indices, odd_classes[:, 0]
