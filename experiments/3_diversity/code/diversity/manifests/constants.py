"""Shared vocabulary for exp-3's manifest construction: levels, allocations, and the
exp-2 source-file/anchor-assignment lookups they map onto."""

from __future__ import annotations

# The three diversity levels (exp-3's Hydra ``assignment``) and the two
# allocations exp-3 reuses from exp-2 (exp-3's Hydra ``condition``).
LEVELS = ("narrow", "random", "wide")
ALLOCATIONS = ("balanced", "severe")

# Exp-2 source manifest per allocation (plan Stage 1, point 1): 'balanced' is
# assignment-independent (written once, no assignment prefix); 'severe' is
# written per tail assignment and exp-3 anchors to the native order only.
SOURCE_MANIFEST = {
    "balanced": "manifest_balanced.csv",
    "severe": "manifest_native_severe.csv",
}

# The exp-2 Hydra ``assignment`` value whose results/tuning-selection
# directory anchors exp-3's own ``random`` level for each allocation (see
# modeling/workflows/condition_scope.py::scoped_assignments: 'balanced' is
# confirmed under the 'unassigned' placeholder, 'severe' under 'native').
ANCHOR_ASSIGNMENT = {"balanced": "unassigned", "severe": "native"}
