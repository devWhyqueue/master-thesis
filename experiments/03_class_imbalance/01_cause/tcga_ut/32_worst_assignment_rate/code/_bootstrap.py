"""Prepend earlier experiment code dirs to sys.path."""

from __future__ import annotations

import sys
from itertools import chain
from pathlib import Path

_EXPERIMENTS = next(
    p for p in Path(__file__).resolve().parents if p.name == "experiments"
)
_CODE_DIRS = (
    "02_benchmark_patch",
    "03_classifier_limitation",
    "05_effective_support",
    "06_multidirectional_redundancy",  # redundancy.exp5_config, used by neighbours.accuracy
    "07_site_coverage",
    "08_patient_coverage",  # neighbours.accuracy.recall_stack, used by centre.analyze.arm_accuracy
    "14_shortage_decomposition",
    "16_centre_error",
    "17_patient_directions",
    "22_spectrum_regularization",  # spectrum.baseline_config, reused for exp-32 output addressing
    "25_damage_tcga_ut",  # prevalence.* : arms, allocation, fit/weight helpers reused as-is
    "29_tail_assignment",  # assignment.* : class recall pooling, properties, io reused as-is
    "30_tail_assignment",  # permutation.design/model/order reused as-is
    "31_worst_assignment",  # worst.score/order.order_perm/figures/report reused as-is
)

for _name in _CODE_DIRS:
    _code_dir = str(
        next(
            chain(
                _EXPERIMENTS.glob(f"*/{_name}/code"),
                _EXPERIMENTS.glob(f"*/*/{_name}/code"),
                _EXPERIMENTS.glob(f"*/*/*/{_name}/code"),
            )
        )
    )
    if _code_dir not in sys.path:
        sys.path.insert(0, _code_dir)
