"""Constants and arm plans for the class-properties cross-dataset experiment (exp-35).

Zero new fits: every arm read here is already stored by an earlier experiment. The pool for one
(dataset, pool_kind, rho) combination is a list of ``(arm, slurm_output_key, count_field)``
triples -- ``count_field`` is ``"class_counts"`` for every arm except the prior-only ``P{rho}``
arms (exp-27/28), whose physical patch counts stay balanced and whose injected prior instead lives
in the stored ``prior_counts`` field (``cause.fit._write_adjusted``'s convention).
"""

from __future__ import annotations

__all__ = [
    "RATIOS",
    "POOL_KINDS",
    "GATE_MIN_FRACTION",
    "CLASS_COMPOSITION_SHRINK_MIN",
    "CLASS_COMPOSITION_G1_FACTOR",
    "DATASET_SPECIFIC_SHRINK_MAX",
    "DATASET_SPECIFIC_G1_FACTOR",
    "ArmSpec",
    "total_arm_plan",
    "prior_arm_plan",
    "arm_plan",
    "sensitivity_c_arm_plan",
]

RATIOS: tuple[int, ...] = (10, 100)
POOL_KINDS: tuple[str, ...] = ("total", "prior")

# Gate: >= 5 of 7 BRACS classes inside the TCGA-UT (h, m) convex hull, averaged over splits.
GATE_MIN_FRACTION: float = 5.0 / 7.0

# Answer thresholds (primary = total pool, rho = 100).
CLASS_COMPOSITION_SHRINK_MIN: float = 0.7
CLASS_COMPOSITION_G1_FACTOR: float = 0.5
DATASET_SPECIFIC_SHRINK_MAX: float = 0.3
DATASET_SPECIFIC_G1_FACTOR: float = 0.5

ArmSpec = tuple[str, str, str]  # (arm, slurm_output_key, count_field)


def total_arm_plan(dataset: str, rho: int) -> list[ArmSpec]:
    """Total-arm pool: r{rho} plus every reused rank-shuffle variant at that rho."""
    plan: list[ArmSpec] = [(f"r{rho}", "prevalence_outputs", "class_counts")]
    if dataset == "bracs":
        plan += [
            (f"a{k}_r{rho}", "assignment_outputs", "class_counts") for k in range(1, 7)
        ]
    elif dataset == "tcga_ut" and rho == 100:
        plan += [
            ("easy_r100", "tail_outputs", "class_counts"),
            ("hard_r100", "tail_outputs", "class_counts"),
            ("worst_r100", "rate_outputs", "class_counts"),
            ("flip_r100", "rate_outputs", "class_counts"),
            ("mild_r100", "rate_outputs", "class_counts"),
        ]
    else:
        raise ValueError(f"Unknown dataset {dataset!r}")
    return plan


def prior_arm_plan(rho: int) -> list[ArmSpec]:
    """Prior-only-arm pool: P{rho}, allocation read from its stored injected prior."""
    return [(f"P{rho}", "p_outputs", "prior_counts")]


def arm_plan(dataset: str, pool: str, rho: int) -> list[ArmSpec]:
    """Dispatch to the total or prior arm plan for one (dataset, pool, rho)."""
    if pool == "total":
        return total_arm_plan(dataset, rho)
    if pool == "prior":
        return prior_arm_plan(rho)
    raise ValueError(f"Unknown pool {pool!r}")


def sensitivity_c_arm_plan(dataset: str, rho: int) -> list[ArmSpec]:
    """Sensitivity (c): drop exp-30/32 fixed orders, random r{rho} draws only."""
    return [(f"r{rho}", "prevalence_outputs", "class_counts")]
