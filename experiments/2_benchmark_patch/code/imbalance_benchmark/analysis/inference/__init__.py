from __future__ import annotations

from imbalance_benchmark.analysis.inference.bootstrap import (
    build_strata,
    expand_to_rows,
    gather_seed_resampled,
    kish_effective_count,
    resample_patient_weights,
    resample_seed_indices,
    weighted_balanced_accuracy,
    weighted_macro_nll,
)
from imbalance_benchmark.analysis.inference.preflight import bootstrap_preflight
from imbalance_benchmark.analysis.inference.gates import (
    calibration_gate,
    ci_excludes_zero,
    deficit,
    discrimination_gate,
    recovery,
)
from imbalance_benchmark.analysis.inference.confirmatory.holm import (
    PRIMARY_METHODS,
    apply_holm,
    confirmatory_family,
    holm_adjust_pvalues,
)
from imbalance_benchmark.analysis.inference.permutation import (
    paired_block_permutation_ba,
    paired_block_permutation_tail_nll,
)
from imbalance_benchmark.analysis.inference.recovery import gates_and_recovery

__all__ = [
    "gates_and_recovery",
    "bootstrap_preflight",
    "build_strata",
    "expand_to_rows",
    "gather_seed_resampled",
    "kish_effective_count",
    "resample_patient_weights",
    "resample_seed_indices",
    "weighted_balanced_accuracy",
    "weighted_macro_nll",
    "calibration_gate",
    "ci_excludes_zero",
    "deficit",
    "discrimination_gate",
    "recovery",
    "PRIMARY_METHODS",
    "apply_holm",
    "confirmatory_family",
    "holm_adjust_pvalues",
    "paired_block_permutation_ba",
    "paired_block_permutation_tail_nll",
]
