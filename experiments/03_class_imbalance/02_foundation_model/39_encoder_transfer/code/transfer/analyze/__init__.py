"""Analyze stage: paired encoder contrasts, prior/support decomposition, probability
quality, and B-selected-lambda sensitivity, from exp-39's stored fit evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.common import N_PATIENT_SPLITS, output_root, write_json

from breadth.analyze.canonical import canonical_class_names
from breadth.analyze.secondary import pack_estimate

from transfer import ENCODERS
from transfer.analyze.accuracy import decomposition, pack, pooled_arms
from transfer.analyze.bootstrap import (
    class_support_rejections,
    contexts,
    encoder_draw_weights,
)
from transfer.analyze.common import N_DRAWS, fit_dirs, paths_by_split, require_record
from transfer.analyze.sensitivity import (
    b_lambda_sensitivity,
    lambda_temperature,
    thirds,
)
from transfer.schedule import load_train_identity

__all__ = ["run_analyze"]

_ARMS = ("B", "R10", "R100", "P10", "P100", "S10", "S100")
_PROB_ARMS = ("B", "R10", "R100")


def _rejections(paths: dict[int, dict[str, Path]], n_classes: int) -> int:
    """Total classes with zero observed test rows, over every encoder's B-arm fit."""
    return sum(
        class_support_rejections(
            np.asarray(
                require_record(d, splits=("test",), array_fields=("labels",))["splits"][
                    "test"
                ]["labels"]
            ),
            n_classes,
        )
        for m in ENCODERS
        for _, _, d in fit_dirs(paths, m, "B")
    )


def _estimates(
    pooled_ba: dict[str, dict[str, np.ndarray]], dists: dict[str, np.ndarray]
) -> dict[str, Any]:
    estimates: dict[str, Any] = {
        f"arm_{m}_{a}": pack_estimate(d)
        for m, arms in pooled_ba.items()
        for a, d in arms.items()
    }
    for key, dist in dists.items():
        estimates[key] = pack(dist, primary=key == "delta_DR_100")
    return estimates


def _prob_estimates(
    pooled_prob: dict[str, dict[str, dict[str, np.ndarray]]],
) -> dict[str, Any]:
    return {
        f"{m}_{a}_{k}": pack_estimate(d)
        for m, arms in pooled_prob.items()
        for a, cols in arms.items()
        for k, d in cols.items()
    }


def _diagnostics(
    config: dict[str, Any],
    paths: dict[int, dict[str, Path]],
    ctxs: dict[int, BootstrapContext],
    n_classes: int,
) -> dict[str, Any]:
    split_names = {
        s: load_train_identity(config, s)[1] for s in range(N_PATIENT_SPLITS)
    }
    return {
        "class_support_rejections": _rejections(paths, n_classes),
        "rank_recall": thirds(paths, ctxs, split_names),
        "lambda_temperature": lambda_temperature(paths),
        "b_lambda_sensitivity": b_lambda_sensitivity(config, paths, ctxs, n_classes),
    }


def _write_outputs(
    config: dict[str, Any],
    estimates: dict[str, Any],
    prob_estimates: dict[str, Any],
    diagnostics: dict[str, Any],
    dists: dict[str, np.ndarray],
) -> Path:
    out_p = output_root(config) / "data" / "analysis.json"
    write_json(
        out_p,
        {
            "draws_per_split": N_DRAWS,
            "estimates": estimates,
            "probability_quality": prob_estimates,
        },
    )
    write_json(out_p.with_name("diagnostics.json"), diagnostics)
    np.savez(
        output_root(config) / "data" / "distributions.npz",
        **cast(dict[str, Any], dists),
    )
    return out_p


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool BA/NLL/ECE per encoder/arm, decompose the prior/support/encoder contrasts,
    and the B-lambda sensitivity, then write analysis.json/diagnostics.json."""
    names = canonical_class_names(config)
    n_classes = len(names)
    paths = paths_by_split(config)
    ctxs: dict[int, BootstrapContext] = contexts(config)

    w = encoder_draw_weights(N_DRAWS, ctxs[0].n_replicates)
    pooled_ba, pooled_prob = pooled_arms(paths, ctxs, n_classes, _ARMS, _PROB_ARMS, w)
    dists = decomposition(pooled_ba)
    diagnostics = _diagnostics(config, paths, ctxs, n_classes)
    return _write_outputs(
        config,
        _estimates(pooled_ba, dists),
        _prob_estimates(pooled_prob),
        diagnostics,
        dists,
    )
