"""Prepend earlier experiment code dirs to sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parents[3]
_CODE_DIRS = (
    "02_benchmark_patch",
    "03_classifier_limitation",
    "05_effective_support",
    "06_multidirectional_redundancy",
    "07_site_coverage",
    "08_patient_coverage",
    "14_shortage_decomposition",
    "16_centre_error",
    "17_patient_directions",
    "19_centre_shrinkage",
    "22_spectrum_regularization",
)

for _name in _CODE_DIRS:
    _code_dir = str(next(_EXPERIMENTS.glob(f"*/{_name}/code")))
    if _code_dir not in sys.path:
        sys.path.insert(0, _code_dir)
