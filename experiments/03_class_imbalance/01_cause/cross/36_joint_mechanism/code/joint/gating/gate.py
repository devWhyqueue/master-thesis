"""Pilot gates (PLAN.md "Execution and genuine precheck"): integrity, prior mechanism, support
mechanism, joint rescue, robustness, and precision. Gates 2-4 hold out the test split; only gate 1
touches it, to compare against an already-frozen sibling experiment's own stored numbers.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from imbalance_benchmark.common import output_root, write_json

from spectrum import baseline_config

from joint import (
    ARMS,
    GATE_INTEGRITY_TOL_PP,
    GATE_PRECISION_HALFWIDTH_PP,
    GATE_PRIOR_PP,
    GATE_RESCUE_ACC_PP,
    GATE_RESCUE_BALANCED_DROP_MAX_PP,
    GATE_RESCUE_DAMAGE_PP,
    GATE_SUPPORT_PP,
    N_SPLITS,
    PILOT_DRAWS,
)
from joint.gating.data import (
    NATIVE_REF_DRAW,
    NATIVE_REF_SETTING,
    rescue_contrast,
    test_ba,
    val_ba,
    val_damage,
)
from joint.gating.precision import projected_halfwidth
from joint.gating.robustness import gate5_robustness

__all__ = ["run_gate"]

logger = logging.getLogger(__name__)


def gate1_integrity(config: dict[str, Any]) -> dict[str, Any]:
    """This dataset's native setting reproduces exp-34's own native-replay draw 0 within 0.01 pp."""
    dataset = config["dataset"]["name"]
    sep_config = baseline_config(config, "separation_outputs")
    ref_setting = NATIVE_REF_SETTING[dataset]
    diffs = []
    for split_idx in range(N_SPLITS):
        for arm in ARMS:
            mine = test_ba(config, f"native_{arm}", split_idx, NATIVE_REF_DRAW)
            theirs = test_ba(
                sep_config, f"{ref_setting}_{arm}", split_idx, NATIVE_REF_DRAW
            )
            diffs.append(abs(mine - theirs))
    max_diff = max(diffs)
    return {"max_abs_diff_pp": max_diff, "pass": max_diff <= GATE_INTEGRITY_TOL_PP}


def gate2_prior_mechanism(config: dict[str, Any]) -> dict[str, Any]:
    """Separation moves pooled D_P >= 1 pp in the predicted direction, consistent in >=2 splits."""
    dataset = config["dataset"]["name"]
    expect_increase = dataset == "tcga_ut"
    per_split = []
    for split_idx in range(N_SPLITS):
        native_p = np.mean(
            [val_damage(config, "native", split_idx, d)["D_P"] for d in PILOT_DRAWS]
        )
        sep_p = np.mean(
            [
                val_damage(config, "separation_only", split_idx, d)["D_P"]
                for d in PILOT_DRAWS
            ]
        )
        per_split.append(float(sep_p - native_p))
    pooled_diff = float(np.mean(per_split))
    signed = pooled_diff if expect_increase else -pooled_diff
    consistent = sum((d > 0) == expect_increase for d in per_split) >= 2
    passed = bool(signed >= GATE_PRIOR_PP and consistent)
    return {
        "pooled_diff_pp": pooled_diff,
        "per_split_diff_pp": per_split,
        "pass": passed,
    }


def gate3_support_mechanism(config: dict[str, Any]) -> dict[str, Any]:
    """BRACS only: correction reduces expanded-BRACS D_S by >=1 pp and beats wrong-direction by >=1 pp."""
    if config["dataset"]["name"] != "bracs":
        return {"pass": True, "skipped": True, "reason": "gate 3 is BRACS-only"}
    sep_d_s, joint_d_s, wrong_d_s = [], [], []
    for split_idx in range(N_SPLITS):
        for draw_idx in PILOT_DRAWS:
            sep_d_s.append(
                val_damage(config, "separation_only", split_idx, draw_idx)["D_S"]
            )
            joint_d_s.append(val_damage(config, "joint", split_idx, draw_idx)["D_S"])
            b = val_ba(config, "joint_B", split_idx, draw_idx)
            wrong_s = val_ba(config, "joint_S_wrong", split_idx, draw_idx)
            wrong_d_s.append(b - wrong_s)
    correction_gain = float(np.mean(sep_d_s) - np.mean(joint_d_s))
    wrong_direction_gain = float(np.mean(wrong_d_s) - np.mean(joint_d_s))
    passed = bool(
        correction_gain >= GATE_SUPPORT_PP and wrong_direction_gain >= GATE_SUPPORT_PP
    )
    return {
        "correction_gain_pp": correction_gain,
        "wrong_direction_gain_pp": wrong_direction_gain,
        "pass": passed,
    }


def _pooled_rescue(
    config: dict[str, Any],
) -> tuple[dict[int, dict[str, float]], dict[int, list[float]]]:
    per_draw: dict[int, dict[str, float]] = {}
    per_split: dict[int, list[float]] = {s: [] for s in range(N_SPLITS)}
    keys = ("damage_fall_pp", "r_rise_pp", "b_drop_pp")
    for draw_idx in PILOT_DRAWS:
        contrasts = [rescue_contrast(config, s, draw_idx) for s in range(N_SPLITS)]
        for s, c in enumerate(contrasts):
            per_split[s].append(c["damage_fall_pp"])
        per_draw[draw_idx] = {
            k: float(np.mean([c[k] for c in contrasts])) for k in keys
        }
    return per_draw, per_split


def gate4_joint_rescue(config: dict[str, Any]) -> dict[str, Any]:
    """BRACS only: total damage falls, R-arm accuracy rises, B-arm accuracy holds, both pilot draws."""
    if config["dataset"]["name"] != "bracs":
        return {"pass": True, "skipped": True, "reason": "gate 4 is BRACS-only"}
    per_draw, per_split = _pooled_rescue(config)
    both_draws_positive = all(v["damage_fall_pp"] > 0 for v in per_draw.values())
    splits_positive = bool(sum(np.mean(v) > 0 for v in per_split.values()) >= 2)
    pooled = {
        k: float(np.mean([v[k] for v in per_draw.values()]))
        for k in ("damage_fall_pp", "r_rise_pp", "b_drop_pp")
    }
    passed = bool(
        pooled["damage_fall_pp"] >= GATE_RESCUE_DAMAGE_PP
        and pooled["r_rise_pp"] >= GATE_RESCUE_ACC_PP
        and pooled["b_drop_pp"] <= GATE_RESCUE_BALANCED_DROP_MAX_PP
        and both_draws_positive
        and splits_positive
    )
    return {
        "per_draw": per_draw,
        "pooled": pooled,
        "both_draws_positive": both_draws_positive,
        "splits_positive": splits_positive,
        "pass": passed,
    }


def gate6_precision(config: dict[str, Any]) -> dict[str, Any]:
    """BRACS only: projected main-design CI half-width for the joint rescue is <= 1 pp."""
    if config["dataset"]["name"] != "bracs":
        return {"pass": True, "skipped": True, "reason": "gate 6 is BRACS-only"}
    contrast = np.array(
        [
            rescue_contrast(config, split_idx, draw_idx)["damage_fall_pp"]
            for draw_idx in PILOT_DRAWS
            for split_idx in range(N_SPLITS)
        ]
    )
    halfwidth = projected_halfwidth(contrast)
    return {
        "projected_halfwidth_pp": halfwidth,
        "pass": halfwidth <= GATE_PRECISION_HALFWIDTH_PP,
    }


def run_gate(config: dict[str, Any]) -> dict[str, Any]:
    """Run all six pilot gates, write diagnostics.json, and stop main submission on any failure."""
    results = {
        "integrity": gate1_integrity(config),
        "prior_mechanism": gate2_prior_mechanism(config),
        "support_mechanism": gate3_support_mechanism(config),
        "joint_rescue": gate4_joint_rescue(config),
        "robustness": gate5_robustness(config),
        "precision": gate6_precision(config),
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
