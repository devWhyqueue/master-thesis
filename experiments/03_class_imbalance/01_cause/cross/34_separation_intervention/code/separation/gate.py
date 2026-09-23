"""Pilot gates (PLAN.md "Execution and stopping gates"): reproduction, manipulation, mechanism
promise, precision, and numerics. All five read the pilot's own run records directly -- every
threshold here is a deterministic tolerance, a pooled validation point contrast, or a projected
power -- so main submission never depends on a heavier (bootstrap-CI) analysis pass than this.
Gates 2-4 hold out the test split entirely; only gate 1's reproduction check touches it, and only
to compare against an already-frozen prior experiment's own stored numbers.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from imbalance_benchmark.common import (
    ensure_dirs,
    output_root,
    read_run_record,
    split_paths,
    write_json,
)

from breadth import LAMBDAS

from sites import allocation_dir

from spectrum import baseline_config

from separation import (
    ARMS,
    GATE_MANIPULATION_PP,
    GATE_MECHANISM_PP,
    GATE_POWER_TARGET,
    GATE_REPRODUCTION_TOL_PP,
    GATE_TRIVIAL_BOUNDS,
    N_SPLITS,
    SEVERITY,
    pilot_settings,
)
from separation.precision import projected_power

__all__ = ["run_gate"]

logger = logging.getLogger(__name__)

_NATIVE_ARM_MAP = {
    "B": "r1",
    "R": f"r{SEVERITY}",
    "P": f"P{SEVERITY}",
    "S": f"S{SEVERITY}",
}


def _record(config: dict[str, Any], allocation: str, split_idx: int) -> dict[str, Any]:
    paths = split_paths(ensure_dirs(config), split_idx)
    rec = read_run_record(
        allocation_dir(paths, allocation, 0),
        splits=("validation", "test"),
        array_fields=(),
    )
    if rec is None:
        raise RuntimeError(f"Missing pilot run record: {allocation} split={split_idx}")
    return rec


def _val_ba(config: dict[str, Any], allocation: str, split_idx: int) -> float:
    rec = _record(config, allocation, split_idx)
    return (
        float(
            rec["splits"]["validation"]["endpoints"]["patient_macro_balanced_accuracy"]
        )
        * 100.0
    )


def _test_ba(config: dict[str, Any], allocation: str, split_idx: int) -> float:
    rec = _record(config, allocation, split_idx)
    return (
        float(rec["splits"]["test"]["endpoints"]["patient_macro_balanced_accuracy"])
        * 100.0
    )


def _val_damage(
    config: dict[str, Any], setting: str, split_idx: int
) -> dict[str, float]:
    b, p, s, r = (_val_ba(config, f"{setting}_{arm}", split_idx) for arm in ARMS)
    return {"D_P": b - p, "D_S": b - s, "D_R": b - r}


def gate1_reproduction(config: dict[str, Any]) -> dict[str, Any]:
    """Native BRACS replay reproduces exp-28's stored native BA within 0.01 pp (BRACS only)."""
    if config["dataset"]["name"] != "bracs":
        return {
            "pass": True,
            "skipped": True,
            "reason": "no replay setting for this dataset",
        }
    prevalence_cfg = baseline_config(config, "native_prevalence_outputs")
    cause_cfg = baseline_config(config, "native_cause_outputs")
    diffs = []
    for split_idx in range(N_SPLITS):
        for arm in ARMS:
            mine = _test_ba(config, f"native_bracs_replay_{arm}", split_idx)
            ref_cfg = prevalence_cfg if arm in ("B", "R") else cause_cfg
            theirs = _test_ba(ref_cfg, _NATIVE_ARM_MAP[arm], split_idx)
            diffs.append(abs(mine - theirs))
    max_diff = max(diffs)
    return {"max_abs_diff_pp": max_diff, "pass": max_diff <= GATE_REPRODUCTION_TOL_PP}


def _treated_and_native(dataset: str) -> tuple[str, str, bool]:
    """(native setting, treated setting, whether the manipulation should raise balanced BA)."""
    if dataset == "bracs":
        return "native_bracs_replay", "expanded_bracs", True
    return "tcga_native_10", "tcga_contracted_10", False


def gate2_manipulation(config: dict[str, Any]) -> dict[str, Any]:
    """Validation balanced-arm (B) BA moves >= 3 pp pooled, consistent direction, non-trivial."""
    native, treated, expect_increase = _treated_and_native(config["dataset"]["name"])
    treated_ba = [_val_ba(config, f"{treated}_B", s) for s in range(N_SPLITS)]
    native_ba = [_val_ba(config, f"{native}_B", s) for s in range(N_SPLITS)]
    per_split = [t - n for t, n in zip(treated_ba, native_ba)]
    pooled_diff = float(np.mean(per_split))
    signed = pooled_diff if expect_increase else -pooled_diff
    consistent = sum((d > 0) == expect_increase for d in per_split) >= 2
    lo, hi = GATE_TRIVIAL_BOUNDS
    non_trivial = all(lo <= v <= hi for v in treated_ba + native_ba)
    passed = bool(signed >= GATE_MANIPULATION_PP and consistent and non_trivial)
    return {
        "pooled_diff_pp": pooled_diff,
        "per_split_diff_pp": per_split,
        "consistent_direction": consistent,
        "non_trivial": non_trivial,
        "pass": passed,
    }


def gate3_mechanism(config: dict[str, Any]) -> dict[str, Any]:
    """Total damage (D_R) moves >= 2 pp in the predicted direction; D_P/D_S follow when pooled."""
    native, treated, expect_increase = _treated_and_native(config["dataset"]["name"])
    native_d = [_val_damage(config, native, s) for s in range(N_SPLITS)]
    treated_d = [_val_damage(config, treated, s) for s in range(N_SPLITS)]
    pooled = {
        key: float(
            np.mean([t[key] for t in treated_d]) - np.mean([n[key] for n in native_d])
        )
        for key in ("D_P", "D_S", "D_R")
    }
    predicted_sign = 1.0 if expect_increase else -1.0
    move = predicted_sign * pooled["D_R"] >= GATE_MECHANISM_PP
    prior_ok = np.sign(pooled["D_P"]) == predicted_sign
    support_ok = np.sign(pooled["D_S"]) == predicted_sign
    passed = bool(move and prior_ok and support_ok)
    return {"pooled_change_pp": pooled, "pass": passed}


def gate4_precision(config: dict[str, Any]) -> dict[str, Any]:
    """Patient-cluster simulation from centred validation D_R contrasts projects >= 80% power."""
    native, treated, expect_increase = _treated_and_native(config["dataset"]["name"])
    sign = 1.0 if expect_increase else -1.0
    per_split_effect = np.array(
        [
            sign
            * (
                _val_damage(config, treated, s)["D_R"]
                - _val_damage(config, native, s)["D_R"]
            )
            for s in range(N_SPLITS)
        ]
    )
    power = projected_power(per_split_effect)
    return {
        "per_split_effect_pp": per_split_effect.tolist(),
        "projected_power": power,
        "pass": power >= GATE_POWER_TARGET,
    }


def gate5_numerics(config: dict[str, Any]) -> dict[str, Any]:
    """Every pilot fit converged; boundary lambda selections are flagged, not gated."""
    dataset = config["dataset"]["name"]
    not_converged: list[str] = []
    boundary: list[dict[str, Any]] = []
    for setting in pilot_settings(dataset):
        for arm in ARMS:
            for split_idx in range(N_SPLITS):
                rec = _record(config, f"{setting}_{arm}", split_idx)
                label = f"{setting}_{arm} split={split_idx}"
                solver = rec.get("solver", {})
                if not solver.get("converged", False):
                    not_converged.append(label)
                lam = solver.get("lambda")
                if lam is not None and lam in (min(LAMBDAS), max(LAMBDAS)):
                    boundary.append({"fit": label, "lambda": lam})
    return {
        "not_converged": not_converged,
        "boundary_hits": boundary,
        "pass": not not_converged,
    }


def run_gate(config: dict[str, Any]) -> dict[str, Any]:
    """Run all five pilot gates, write diagnostics.json, and stop main submission on any failure."""
    results = {
        "reproduction": gate1_reproduction(config),
        "manipulation": gate2_manipulation(config),
        "mechanism": gate3_mechanism(config),
        "precision": gate4_precision(config),
        "numerics": gate5_numerics(config),
    }
    overall = all(g["pass"] for g in results.values())
    payload = {"gates": results, "overall_pass": overall}
    out_p = output_root(config) / "data" / "diagnostics.json"
    write_json(out_p, payload)
    if not overall:
        failed = [name for name, g in results.items() if not g["pass"]]
        raise RuntimeError(f"Pilot gate(s) failed: {failed}; see {out_p}")
    logger.info("All pilot gates passed for %s.", config["dataset"]["name"])
    return payload
