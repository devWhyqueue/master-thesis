"""Constants for the directional-shrinkage experiment (exp-20)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imbalance_benchmark.common import ensure_dirs, read_run_record, split_paths

from sites import allocation_dir

from centre import PATIENT_COUNTS

__all__ = [
    "ARMS",
    "ALPHA_FACTORS",
    "NONZERO_ALPHA_FACTORS",
    "REUSED_ARM_FAMILIES",
    "baseline_config",
    "baseline_arm_dir",
    "baseline_arm_score",
]

ARMS: tuple[str, ...] = tuple(
    f"{family}{g}" for family in ("A", "At") for g in PATIENT_COUNTS
)
ALPHA_FACTORS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
# alpha = 0 is never fit: its score and full run record are always read from the R arm of
# slurm.baseline_outputs, since directional shrinkage at alpha = 0 leaves the cohort's centres,
# and therefore its fit, identical to R's.
NONZERO_ALPHA_FACTORS: tuple[float, ...] = tuple(a for a in ALPHA_FACTORS if a)

# Reused arm families keyed by the config path pointing at their source experiment's outputs:
# R/C from exp-16 (TCGA-UT) or exp-18 (BRACS); S/St from exp-19; RWc from exp-17 (TCGA-UT) or
# exp-18 (BRACS, where it already sits alongside R/C).
REUSED_ARM_FAMILIES: dict[str, tuple[str, ...]] = {
    "baseline_outputs": ("R", "C"),
    "exp19_outputs": ("S", "St"),
    "whitening_outputs": ("RWc",),
}


def baseline_config(config: dict[str, Any], key: str) -> dict[str, Any]:
    """View this config's stored source-experiment outputs (``key``) as a standalone config."""
    out = config.get("slurm", {}).get(key)
    if not out:
        raise ValueError(f"Config missing slurm.{key}")
    return {**config, "paths": {"outputs": str(out)}}


def baseline_arm_dir(
    config: dict[str, Any], split_idx: int, draw_idx: int, arm: str
) -> Path:
    """Output directory of one arm's fit in ``slurm.baseline_outputs`` (R and C's source)."""
    paths = split_paths(
        ensure_dirs(baseline_config(config, "baseline_outputs")), split_idx
    )
    return allocation_dir(paths, arm, draw_idx)


def baseline_arm_score(
    config: dict[str, Any], split_idx: int, draw_idx: int, arm: str
) -> float:
    """Baseline arm's validation score for this shard, reused in place of a fresh fit."""
    record = read_run_record(
        baseline_arm_dir(config, split_idx, draw_idx, arm), array_fields=()
    )
    if record is None:
        raise RuntimeError(f"Baseline {arm} arm missing; fit exp-16/18 before exp-20")
    return float(
        record["splits"]["validation"]["endpoints"]["patient_macro_balanced_accuracy"]
    )
