"""Stage 3 (fit, GPU): confirm-style fits for the narrow/wide cells, plus anchor import.

Not run in this task (no cluster access, no real feature bank locally) --
implemented and unit-testable, per plan Stage 3.

Work item = (split, level in {narrow, wide}, allocation, method, seed) with
``METHODS = (ce, weighted_ce, semantic_scale_ce)``: 3 splits x 2 levels x 2
allocations x 3 methods x 5 seeds = 180 per dataset. The ``random`` level is
never fitted here; it is imported from exp-2's own confirmed runs by
:func:`import_anchor`, gated on manifest sha256 and ``tuning_params``
equality so a stale anchor is never silently reused.

Facade over ``units`` (work-item enumeration), ``context`` (RunContext
inputs and tuning selections), ``run`` (per-shard execution), and ``anchor``
(random-cell import) -- kept small per this repo's file-length rule.
"""

from __future__ import annotations

from diversity.fit.anchor import import_anchor
from diversity.fit.run import run_fit_shard
from diversity.fit.units import (
    FIT_LEVELS,
    METHODS,
    N_SEEDS,
    FitUnit,
    fit_units,
    resolve_fit_bundle,
)

__all__ = [
    "METHODS",
    "FIT_LEVELS",
    "N_SEEDS",
    "FitUnit",
    "fit_units",
    "resolve_fit_bundle",
    "run_fit_shard",
    "import_anchor",
]
