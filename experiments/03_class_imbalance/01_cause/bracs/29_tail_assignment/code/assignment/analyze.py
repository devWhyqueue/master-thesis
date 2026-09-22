"""Analyze stage: pool reused (exp-26) and new (shifted-tail) arm accuracy, write analysis,
diagnostics, and figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import output_root, write_json

from breadth import BOOTSTRAP_SEED
from breadth.analyze.canonical import canonical_class_names, canonical_permutation

from sites import allocation_dir
from sites.recall import contexts

from neighbours.accuracy import recall_stack

from decomposition.model import draw_weights

from centre import N_DRAWS, N_SPLITS
from centre.analyze import pooled

from spectrum import baseline_config

from directions.analyze import _write_analysis

from assignment import ATYPICAL, NEW_FIT_ARMS, RATIOS, REUSED_ARMS
from assignment._io import paths_by_split, require_record
from assignment.figures import damage_figure, rank_response_figure
from assignment.properties import class_properties, correlations_by_rho
from assignment.rank import Pooled, TailDamage, tail_damage_for_rho

__all__ = ["run_analyze"]


def _class_recall_stack(
    config: dict[str, Any], names: list[str], arms: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Per arm, (F, C, R) per-class recall in percent (as centre.analyze.arm_accuracy, no mean)."""
    ctxs = contexts(config)
    keys = [(s, d) for s in range(N_SPLITS) for d in range(N_DRAWS)]
    paths = paths_by_split(config)
    perms = {s: canonical_permutation(config, s, names) for s in range(N_SPLITS)}
    return {
        arm: recall_stack(
            [allocation_dir(paths[s], arm, d) for s, d in keys],
            [ctxs[s] for s, _ in keys],
            [perms[s] for s, _ in keys],
            len(names),
        )
        * 100.0
        for arm in arms
    }


def _mean_realized_rho(paths: dict[int, dict[str, Path]], arm: str) -> float:
    return float(
        np.mean(
            [
                require_record(allocation_dir(paths[s], arm, d))["realized_rho"]
                for s in range(N_SPLITS)
                for d in range(N_DRAWS)
            ]
        )
    )


def _realized_rho(
    config: dict[str, Any], exp26_config: dict[str, Any]
) -> dict[str, float]:
    """Mean realized rho per arm over every stored fit."""
    paths_new, paths_reused = paths_by_split(config), paths_by_split(exp26_config)
    out = {arm: _mean_realized_rho(paths_reused, arm) for arm in REUSED_ARMS}
    out.update({arm: _mean_realized_rho(paths_new, arm) for arm in NEW_FIT_ARMS})
    return out


def _pool(
    class_acc: dict[str, np.ndarray], names: list[str]
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Class-mean (F, R) arm accuracy, the draw-resampling weights, pooled BA, and r1's own recall."""
    acc = {arm: a.mean(axis=1) for arm, a in class_acc.items()}
    fit_split = np.repeat(np.arange(N_SPLITS), N_DRAWS)
    w = draw_weights(
        fit_split, N_DRAWS, acc["r1"].shape[-1], np.random.default_rng(BOOTSTRAP_SEED)
    )
    ba = {arm: pooled(a, w) for arm, a in acc.items()}
    r1_own = {c: pooled(class_acc["r1"][:, ci, :], w) for ci, c in enumerate(names)}
    return acc, fit_split, w, ba, r1_own


def _all_tail_damage(
    config: dict[str, Any],
    exp26_config: dict[str, Any],
    names: list[str],
    pooled_acc: Pooled,
) -> tuple[dict[int, TailDamage], dict[str, np.ndarray]]:
    """Every ratio's tail-damage bundle, and the flat distributions dict for analysis.json."""
    results = {
        rho: tail_damage_for_rho(config, exp26_config, names, pooled_acc, rho, ATYPICAL)
        for rho in RATIOS
    }
    dists = {f"arm_{arm}": d for arm, d in pooled_acc.ba.items()}
    for result in results.values():
        dists.update(result.dists)
    return results, dists


def _properties_and_correlations(
    exp26_config: dict[str, Any],
    names: list[str],
    r1_own: dict[str, np.ndarray],
    results: dict[int, TailDamage],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    """r1 recall / confusability per class, and their correlation with tail damage per ratio."""
    properties = class_properties(exp26_config, names, r1_own)
    correlations = correlations_by_rho(
        names,
        properties,
        {rho: r.d_by_class for rho, r in results.items()},
        {rho: r.own_loss_by_class for rho, r in results.items()},
    )
    return properties, correlations


def _write_diagnostics(
    path: Path,
    config: dict[str, Any],
    exp26_config: dict[str, Any],
    names: list[str],
    results: dict[int, TailDamage],
    properties: dict[str, dict[str, float]],
    correlations: dict[str, dict[str, dict[str, float]]],
) -> None:
    write_json(
        path.with_name("diagnostics.json"),
        {
            "realized_rho": _realized_rho(config, exp26_config),
            "rank_response_matrix": {
                str(rho): r.matrix.tolist() for rho, r in results.items()
            },
            "own_rank_only_r2": {str(rho): r.r2 for rho, r in results.items()},
            "tail_damage_sd_across_classes": {
                str(rho): float(np.std([r.d_by_class[c][0] for c in names]))
                for rho, r in results.items()
            },
            "class_properties": properties,
            "correlations": correlations,
        },
    )


def _write_figures(
    config: dict[str, Any],
    dists: dict[str, np.ndarray],
    names: list[str],
    properties: dict[str, dict[str, float]],
    results: dict[int, TailDamage],
) -> None:
    order = sorted(names, key=lambda c: properties[c]["r1_recall"])
    figures = output_root(config) / "figures"
    damage_figure(dists, order, figures / "tail_damage.pdf")
    rank_response_figure(
        results[100].matrix, names, 100, figures / "rank_response_r100.pdf"
    )


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool reused (exp-26) and new (shifted-tail) arm accuracy, write analysis, diagnostics, figures."""
    names = canonical_class_names(config)
    exp26_config = baseline_config(config, "prevalence_outputs")
    class_acc = {
        **_class_recall_stack(exp26_config, names, REUSED_ARMS),
        **_class_recall_stack(config, names, NEW_FIT_ARMS),
    }
    acc, fit_split, w, ba, r1_own = _pool(class_acc, names)
    results, dists = _all_tail_damage(
        config, exp26_config, names, Pooled(class_acc, ba, r1_own, w)
    )
    properties, correlations = _properties_and_correlations(
        exp26_config, names, r1_own, results
    )

    path = _write_analysis(config, acc, fit_split, dists)
    _write_diagnostics(
        path, config, exp26_config, names, results, properties, correlations
    )
    _write_figures(config, dists, names, properties, results)
    return path
