"""Analyze stage: Part A (exp-25 reanalysis) + Part B (easy/hard r100 confirmatory contrast).

Pools exp-25's stored r1/r10/r100/N fits with the two new tail arms, fits the per-class
piecewise own-recall model (``permutation.model``) on the pooled fits, and validates/extends it
with leave-one-draw-out prediction, a 10,000-sample random-permutation spread, and the
headroom/confusability correlations. Writes analysis.json, diagnostics.json, and two figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from imbalance_benchmark.common import output_root, write_json

from breadth.analyze.canonical import canonical_class_names
from breadth.analyze.secondary import pack_estimate
from spectrum import baseline_config
from prevalence import patients_per_class
from directions.analyze import _write_analysis

from assignment.analyze import _class_recall_stack, _pool
from assignment.properties import class_properties

from permutation import (
    MODEL_RATIOS,
    N_SAMPLED,
    NEW_FIT_ARMS,
    RATIO_B,
    REUSED_ARMS,
    SAMPLE_SEED,
)
from permutation.design import lodo, model_design, native_draws, stored_counts
from permutation.figures import spread_figure, tail_loss_figure
from permutation.model import (
    Coefficients,
    fit_piecewise,
    predict_d,
    predict_delta,
    ranked_counts,
    sampled_permutation_d,
    spearman_distribution,
    tail_z,
    z_of_counts,
)
from permutation.order import tail_order

__all__ = ["run_analyze"]


class _Fit(NamedTuple):
    """Everything the analysis is built from: pooled accuracy, model coefficients, properties."""

    names: list[str]
    g: int
    k: int
    acc: dict[str, np.ndarray]
    fit_split: np.ndarray
    ba: dict[str, np.ndarray]
    properties: dict[str, dict[str, float]]
    order: list[str]
    z_c: np.ndarray
    delta_c: np.ndarray
    coefs: Coefficients
    coefs_point: Coefficients


def _prepare(config: dict[str, Any]) -> _Fit:
    """Pool exp-25's reused arms with the new tail arms and fit the piecewise model."""
    names = canonical_class_names(config)
    exp25_config = baseline_config(config, "prevalence_outputs")
    g = patients_per_class(exp25_config)
    class_acc = {
        **_class_recall_stack(exp25_config, names, REUSED_ARMS),
        **_class_recall_stack(config, names, NEW_FIT_ARMS),
    }
    acc, fit_split, w, ba, r1_own = _pool(class_acc, names)
    properties = class_properties(exp25_config, names, r1_own)
    order = tail_order(exp25_config, names)
    z_c, delta_c, w_c = model_design(exp25_config, names, class_acc, w, g)
    coefs = fit_piecewise(z_c, delta_c, w_c)
    return _Fit(
        names,
        g,
        len(names),
        acc,
        fit_split,
        ba,
        properties,
        order,
        z_c,
        delta_c,
        coefs,
        coefs.point(),
    )


def _out_of_sample(
    config: dict[str, Any],
    names: list[str],
    coefs: Coefficients,
    g: int,
    ba: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Predicted vs. observed D for the new (out-of-sample) easy/hard-tail orders."""
    out: dict[str, Any] = {}
    for arm in NEW_FIT_ARMS:
        z_arm = z_of_counts(stored_counts(config, names, arm), g).mean(axis=0)  # (C,)
        out[arm] = {
            "predicted": pack_estimate(predict_d(coefs, z_arm)),
            "observed": pack_estimate(ba["r1"] - ba[arm]),
        }
    return out


def _spread(coefs_point: Coefficients, g: int, k: int) -> dict[int, np.ndarray]:
    """Predicted-D distribution of ``N_SAMPLED`` random permutations, per model ratio."""
    rng = np.random.default_rng(SAMPLE_SEED)
    return {
        rho: sampled_permutation_d(
            coefs_point, z_of_counts(ranked_counts(rho, g, k), g), rng, N_SAMPLED
        )
        for rho in MODEL_RATIOS
    }


def _tail_own_loss(coefs: Coefficients, g: int, k: int) -> dict[int, np.ndarray]:
    """Per-class predicted own-recall loss (C, R) if that class alone sat at the tail rank."""
    return {
        rho: -predict_delta(coefs, np.full(k, tail_z(rho, g, k))) / k
        for rho in MODEL_RATIOS
    }


def _correlations(
    properties: dict[str, dict[str, float]],
    names: list[str],
    own_loss_by_rho: dict[int, np.ndarray],
) -> dict[str, dict[str, dict[str, float]]]:
    """Headroom (r1 recall) / confusability vs. predicted tail own loss, per ratio."""
    r1_recall = [properties[c]["r1_recall"] for c in names]
    confusability = [properties[c]["confusability"] for c in names]
    return {
        str(rho): {
            "r1_recall_vs_predicted_tail_own_loss": pack_estimate(
                spearman_distribution(r1_recall, own_loss)
            ),
            "confusability_vs_predicted_tail_own_loss": pack_estimate(
                spearman_distribution(confusability, own_loss)
            ),
        }
        for rho, own_loss in own_loss_by_rho.items()
    }


def _endpoint_dists(ba: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The pre-specified BA-damage contrasts, from pooled arm accuracy alone."""
    dists = {f"arm_{a}": d for a, d in ba.items()}
    dists["D_easy"] = ba["r1"] - ba[f"easy_r{RATIO_B}"]
    dists["D_hard"] = ba["r1"] - ba[f"hard_r{RATIO_B}"]
    dists["D_easy_minus_hard"] = dists["D_easy"] - dists["D_hard"]
    dists["D_N"] = ba["r1"] - ba["N"]
    for rho in MODEL_RATIOS:
        dists[f"D_{rho}_native_mean"] = ba["r1"] - ba[f"r{rho}"]
    return dists


def _write_diagnostics(
    path: Path,
    fitted: _Fit,
    lodo_result: dict[str, Any],
    out_of_sample: dict[str, Any],
    spread: dict[int, np.ndarray],
    draws: dict[int, np.ndarray],
    correlations: dict[str, dict[str, dict[str, float]]],
) -> None:
    write_json(
        path.with_name("diagnostics.json"),
        {
            "tail_order": fitted.order,
            "model_coefficients": {
                c: {
                    "a": float(fitted.coefs_point.a[ci]),
                    "b_pos": float(fitted.coefs_point.b_pos[ci]),
                    "b_neg": float(fitted.coefs_point.b_neg[ci]),
                }
                for ci, c in enumerate(fitted.names)
            },
            "leave_one_draw_out": lodo_result,
            "out_of_sample": out_of_sample,
            "sampled_permutation_spread": {
                str(rho): {
                    "mean": float(np.mean(s)),
                    "sd": float(np.std(s)),
                    "ci_2_5": float(np.percentile(s, 2.5)),
                    "ci_97_5": float(np.percentile(s, 97.5)),
                    "min": float(np.min(s)),
                    "max": float(np.max(s)),
                }
                for rho, s in spread.items()
            },
            "native_draws_D": {str(rho): d.tolist() for rho, d in draws.items()},
            "class_properties": fitted.properties,
            "correlations": correlations,
        },
    )


def _write_figures(
    config: dict[str, Any],
    fitted: _Fit,
    dists: dict[str, np.ndarray],
    spread: dict[int, np.ndarray],
    draws: dict[int, np.ndarray],
    tail_own_loss: dict[int, np.ndarray],
) -> None:
    figures = output_root(config) / "figures"
    spread_figure(
        spread[RATIO_B],
        draws[RATIO_B],
        {"N": dists["D_N"], "easy_r100": dists["D_easy"], "hard_r100": dists["D_hard"]},
        figures / "assignment_spread_r100.pdf",
    )
    tail_loss_figure(
        tail_own_loss,
        [fitted.properties[c]["r1_recall"] for c in fitted.names],
        figures / "tail_loss_vs_recall.pdf",
    )


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool exp-25's reused arms with the new easy/hard-tail arms; fit, validate, and report."""
    fitted = _prepare(config)
    lodo_result = lodo(fitted.z_c, fitted.delta_c)
    out_of_sample = _out_of_sample(
        config, fitted.names, fitted.coefs, fitted.g, fitted.ba
    )
    spread = _spread(fitted.coefs_point, fitted.g, fitted.k)
    draws = {rho: native_draws(fitted.delta_c, rho) for rho in MODEL_RATIOS}
    tail_own_loss = _tail_own_loss(fitted.coefs, fitted.g, fitted.k)
    correlations = _correlations(fitted.properties, fitted.names, tail_own_loss)

    dists = _endpoint_dists(fitted.ba)
    path = _write_analysis(config, fitted.acc, fitted.fit_split, dists)
    _write_diagnostics(
        path, fitted, lodo_result, out_of_sample, spread, draws, correlations
    )
    _write_figures(config, fitted, dists, spread, draws, tail_own_loss)
    return path
