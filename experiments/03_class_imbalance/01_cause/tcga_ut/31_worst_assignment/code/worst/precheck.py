"""Part A gate (0 new fits): sanity-check ``S`` against exp-30's already-observed easy/hard/random
damage before any new fit is submitted.

- G1: S(easy order) > S(hard order) -- must match the observed direction (D_easy > D_hard).
- G2: mean S over exp-25's 30 random orders > S(easy) > S(hard) -- must match the observed
  D_random > D_easy > D_hard direction.
- G3 (descriptive, not binding): Spearman(S_f, D_f) > 0 over the 30 stored random orders.

Pass = G1 and G2. No score tuning loop -- the score is fixed by the exp-31 plan before this runs.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from imbalance_benchmark.common import output_root, write_json

from breadth.analyze.canonical import canonical_class_names

from spectrum import baseline_config

from prevalence import patients_per_class

from assignment.analyze import _class_recall_stack
from assignment.properties import _spearman

from permutation.design import stored_counts
from permutation.order import tail_order

from worst.figures import gate_figure
from worst.order import order_perm, score_inputs
from worst.score import score

__all__ = ["gate_markers", "run_precheck"]


def gate_markers(
    config: dict[str, Any], result: dict[str, Any]
) -> dict[str, tuple[float, float, float, float]]:
    """(S, D, lower, upper) per sorted order, reading exp-30's stored damage estimates."""
    exp30_config = baseline_config(config, "tail_outputs")
    path = output_root(exp30_config) / "data" / "analysis.json"
    estimates = json.loads(path.read_text(encoding="utf-8"))["estimates"]
    return {
        name: (
            result[f"s_{name}"],
            estimates[f"D_{name}"]["point"],
            estimates[f"D_{name}"]["ci_2_5"],
            estimates[f"D_{name}"]["ci_97_5"],
        )
        for name in ("easy", "hard")
    }


def _random_s_and_d(
    exp25_config: dict[str, Any],
    names: list[str],
    g: int,
    w: np.ndarray,
    h: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(S_f, D_f) for each of exp-25's 30 stored random-order r100 fits."""
    counts = stored_counts(exp25_config, names, "r100")  # (F, C), canonical order
    perms = np.argsort(-counts, axis=1)  # rank-0 (largest count / head) first
    s_random = np.array([score(perms[f], w, h, z) for f in range(counts.shape[0])])
    class_acc = _class_recall_stack(exp25_config, names, ("r1", "r100"))
    d_random = class_acc["r1"][:, :, 0].mean(axis=1) - class_acc["r100"][:, :, 0].mean(
        axis=1
    )
    return s_random, d_random


def _easy_hard_scores(
    exp25_config: dict[str, Any],
    names: list[str],
    w: np.ndarray,
    h: np.ndarray,
    z: np.ndarray,
) -> tuple[float, float]:
    """S(easy order), S(hard order) -- exp-30's pooled-r1-recall tail orders."""
    order = tail_order(exp25_config, names)
    easy_perm = order_perm(order, names)
    hard_perm = order_perm(list(reversed(order)), names)
    return score(easy_perm, w, h, z), score(hard_perm, w, h, z)


def _gate_result(
    s_easy: float, s_hard: float, s_random: np.ndarray, d_random: np.ndarray
) -> dict[str, Any]:
    """G1/G2/G3 and their inputs, ready to write to ``precheck.json``."""
    s_random_mean = float(np.mean(s_random))
    g1_pass = s_easy > s_hard
    g2_pass = s_random_mean > s_easy > s_hard
    return {
        "s_easy": s_easy,
        "s_hard": s_hard,
        "s_random_mean": s_random_mean,
        "s_random": s_random.tolist(),
        "d_random": d_random.tolist(),
        "g1_pass": g1_pass,
        "g2_pass": g2_pass,
        "g3_spearman": _spearman(list(s_random), list(d_random)),
        "pass": g1_pass and g2_pass,
    }


def run_precheck(config: dict[str, Any]) -> dict[str, Any]:
    """Compute G1/G2/G3 and write ``precheck.json``."""
    names = canonical_class_names(config)
    exp25_config = baseline_config(config, "prevalence_outputs")
    g = patients_per_class(exp25_config)
    w, h, z = score_inputs(exp25_config, names, g)

    s_easy, s_hard = _easy_hard_scores(exp25_config, names, w, h, z)
    s_random, d_random = _random_s_and_d(exp25_config, names, g, w, h, z)
    result = _gate_result(s_easy, s_hard, s_random, d_random)

    write_json(output_root(config) / "data" / "precheck.json", result)
    gate_figure(
        s_random,
        d_random,
        gate_markers(config, result),
        output_root(config) / "figures" / "gate_score_vs_damage.pdf",
    )
    return result
