"""Stage 1 (build): narrow/random/wide manifest construction at fixed support.

Pins every count exp-2's frozen ``balanced``/``severe`` allocations fix at
``(cancer_type, case_id, slide_id)`` granularity and re-selects only which
patches occupy the fixed slots (plan Stage 1). ``random`` is a content copy
of the exp-2 source condition; ``narrow`` and ``wide`` are deterministic
nearest-to-mean / farthest-point selections over frozen Virchow2 features.

No file under ``experiments/2_benchmark_patch`` is edited; ``imbalance_benchmark``
is imported as a library (see ``__main__.py``'s ``sys.path`` prepend).

This is a facade over the package's submodules (kept small per this repo's
file-length rule): ``constants``, ``selection``, ``pool``, ``allocation``,
``freeze``. Every name below is importable as ``diversity.manifests.<name>``.
"""

from __future__ import annotations

from diversity.manifests.allocation import assert_invariants, build_allocation_levels
from diversity.manifests.constants import (
    ALLOCATIONS,
    ANCHOR_ASSIGNMENT,
    LEVELS,
    SOURCE_MANIFEST,
)
from diversity.manifests.freeze import (
    build_derived_freeze,
    build_split,
    exp2_base_paths,
    exp2_split_paths,
    verify_derived_freeze,
)
from diversity.manifests.pool import (
    SLOT_KEYS,
    eligible_pool,
    headroom_table,
    pool_features,
    slot_table,
)
from diversity.manifests.selection import (
    _select_narrow,
    _select_wide,
    select_narrow,
    select_wide,
)

__all__ = [
    "LEVELS",
    "ALLOCATIONS",
    "SOURCE_MANIFEST",
    "ANCHOR_ASSIGNMENT",
    "SLOT_KEYS",
    "exp2_base_paths",
    "exp2_split_paths",
    "build_split",
    "slot_table",
    "eligible_pool",
    "pool_features",
    "headroom_table",
    "select_narrow",
    "select_wide",
    "build_allocation_levels",
    "build_derived_freeze",
    "verify_derived_freeze",
    "assert_invariants",
]
