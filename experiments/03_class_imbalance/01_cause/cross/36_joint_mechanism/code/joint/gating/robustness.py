"""Gate 5 (robustness): convergence, boundary-lambda endpoint stability, and (BRACS only) a
positive fixed-lambda rescue -- a boundary selection is stable only when its own validation score
is close to the next grid point's, not accepted as an automatic numerical pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from joint import ARMS, GATE_BOUNDARY_STABILITY_PP, N_SPLITS, PILOT_DRAWS
from joint.grid import LAMBDAS, read_grid
from joint.gating.data import allocation_out_dir, record, test_ba

__all__ = ["gate5_robustness"]


def _boundary_gap(
    config: dict[str, Any],
    setting: str,
    arm: str,
    split_idx: int,
    draw_idx: int,
    lam: float,
) -> float:
    candidates = read_grid(
        allocation_out_dir(config, f"{setting}_{arm}", split_idx, draw_idx)
    )
    by_lambda = {c.lambda_val: c.val_score for c in candidates}
    neighbour_lam = LAMBDAS[1] if lam == min(LAMBDAS) else LAMBDAS[-2]
    return abs(by_lambda[lam] - by_lambda[neighbour_lam]) * 100.0


def _boundary_check(config: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    not_converged: list[str] = []
    unstable_boundary: list[dict[str, Any]] = []
    for setting in ("native", "separation_only", "centre_only", "joint"):
        for arm in ARMS:
            for draw_idx in PILOT_DRAWS:
                for split_idx in range(N_SPLITS):
                    rec = record(config, f"{setting}_{arm}", split_idx, draw_idx)
                    label = f"{setting}_{arm} split={split_idx} draw={draw_idx}"
                    if not rec.get("solver", {}).get("converged", False):
                        not_converged.append(label)
                    lam = rec.get("solver", {}).get("lambda")
                    if lam is None or lam not in (min(LAMBDAS), max(LAMBDAS)):
                        continue
                    gap = _boundary_gap(config, setting, arm, split_idx, draw_idx, lam)
                    if gap > GATE_BOUNDARY_STABILITY_PP:
                        unstable_boundary.append(
                            {"fit": label, "lambda": lam, "neighbour_gap_pp": gap}
                        )
    return not_converged, unstable_boundary


def _fixed_lambda_rescue(config: dict[str, Any]) -> float:
    gains = []
    for draw_idx in PILOT_DRAWS:
        for split_idx in range(N_SPLITS):
            native_r = test_ba(config, "native_R_fixedlambda", split_idx, draw_idx)
            native_b = test_ba(config, "native_B_fixedlambda", split_idx, draw_idx)
            joint_r = test_ba(config, "joint_R_fixedlambda", split_idx, draw_idx)
            joint_b = test_ba(config, "joint_B_fixedlambda", split_idx, draw_idx)
            gains.append((native_b - native_r) - (joint_b - joint_r))
    return float(np.mean(gains))


def gate5_robustness(config: dict[str, Any]) -> dict[str, Any]:
    """Convergence, boundary-lambda stability, and (BRACS only) a positive fixed-lambda rescue."""
    not_converged, unstable_boundary = _boundary_check(config)
    fixed_gain = (
        _fixed_lambda_rescue(config) if config["dataset"]["name"] == "bracs" else None
    )
    fixed_ok = fixed_gain is None or fixed_gain > 0
    passed = bool(not not_converged and not unstable_boundary and fixed_ok)
    return {
        "not_converged": not_converged,
        "boundary_hits": unstable_boundary,
        "fixed_lambda_bracs_rescue_pp": fixed_gain,
        "pass": passed,
    }
