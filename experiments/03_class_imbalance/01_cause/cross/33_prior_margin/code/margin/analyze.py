"""Analyze stage: orchestrates the gate/H1/H2 statistics (``margin.gate``) plus H3 and output
writing.

Run once per dataset (``inject`` then ``analyze``). H1's cross-dataset gap and H3's BRACS overlay
need the *other* dataset's own analysis: this reads it through ``slurm.peer_outputs`` (mirroring
``spectrum.baseline_config``'s addressing) when that run has already completed, and otherwise
leaves them out, so the two per-dataset submissions can run in either order but only the one that
runs second fills them in.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from imbalance_benchmark.common import (
    ensure_dirs,
    output_root,
    read_run_record,
    split_paths,
    write_json,
)

from breadth.analyze.canonical import canonical_class_names

from sites import allocation_dir

from centre import N_DRAWS, N_SPLITS

from spectrum import baseline_config

from directions.analyze import _write_analysis

from margin import MARGIN_RHOS, N_SUBSETS, SUBSET_K, SUBSET_SEED
from margin.figures import margin_cdf_figure, onset_figure, subset_scatter_figure
from margin.gate import (
    Gate,
    H2Result,
    Peer,
    build_dists,
    gate_check,
    h1_gap,
    h2_curve,
    label,
    peer_data,
    pool,
)
from margin.subset_eval import load_r1_cache, subset_point
from margin.subsets import greedy_confused_subset, point_contexts, random_subsets

__all__ = ["run_analyze"]

logger = logging.getLogger(__name__)

_PROBABILITY_FLOOR = np.finfo(np.float64).tiny


class AnalysisState(NamedTuple):
    """Everything one dataset's ``run_analyze`` computes, before it is written out."""

    acc: dict[str, np.ndarray]
    fit_split: np.ndarray
    dists: dict[str, np.ndarray]
    ba: dict[str, np.ndarray]
    gate: Gate
    h2_result: H2Result
    h1: dict[str, Any] | None
    outcome: str
    margins: np.ndarray
    own_median_margin: float
    h3: list[dict[str, Any]] | None
    peer: Peer


def _full_margins(ds_paths: dict[int, dict[str, Path]]) -> np.ndarray:
    """Every r1 test-patch margin log p_true - max_{d!=true} log p_d, pooled over every fit."""
    out = []
    for s in range(N_SPLITS):
        for d in range(N_DRAWS):
            rec = read_run_record(
                allocation_dir(ds_paths[s], "r1", d),
                splits=("test",),
                array_fields=("labels", "probabilities"),
            )
            if rec is None:
                raise RuntimeError(f"Missing r1 run record at split {s}, draw {d}")
            test = rec["splits"]["test"]
            labels = np.asarray(test["labels"])
            log_probs = np.log(
                np.maximum(np.asarray(test["probabilities"]), _PROBABILITY_FLOOR)
            )
            rows = np.arange(len(labels))
            true_logp = log_probs[rows, labels]
            masked = log_probs.copy()
            masked[rows, labels] = -np.inf
            out.append(true_logp - masked.max(axis=1))
    return np.concatenate(out)


def _run_h3(
    config: dict[str, Any], ds_paths: dict[int, dict[str, Path]], names: list[str]
) -> list[dict[str, Any]]:
    """500 random + 1 greedy-confused 7-class subsets: (classes, ba, d_sim, margin)."""
    rng = np.random.default_rng(SUBSET_SEED)
    ctxs = point_contexts(config)
    r1_cache = load_r1_cache(config, ds_paths, names)
    subsets = random_subsets(rng, names, N_SUBSETS)
    subsets.append(greedy_confused_subset(config, ds_paths, names))
    tags = ["random"] * N_SUBSETS + ["greedy_confused"]
    out = []
    for i, (tag, subset_names) in enumerate(zip(tags, subsets)):
        if i % 100 == 0:
            logger.info("H3 subset %d/%d", i, len(subsets))
        rank_perm = rng.permutation(SUBSET_K)
        ba, d_sim, margin = subset_point(r1_cache, ctxs, names, subset_names, rank_perm)
        out.append(
            {
                "tag": tag,
                "classes": subset_names,
                "ba": ba,
                "d_sim": d_sim,
                "median_margin": margin,
            }
        )
    return out


def _build_state(
    config: dict[str, Any],
    ds_config: dict[str, Any],
    ds_paths: dict[int, dict[str, Path]],
    names: list[str],
) -> AnalysisState:
    """Pool arm accuracy and run the gate, H1, H2, and (on TCGA-UT) H3."""
    acc, fit_split, w, ba = pool(config, ds_config, names)
    dists = build_dists(config, names, ba, w)
    gate = gate_check(config, dists)
    h2_result = h2_curve(dists)
    peer = peer_data(config)
    h1 = h1_gap(config, dists, peer)
    outcome = label(gate[3], h1)

    margins = _full_margins(ds_paths)
    h3 = None
    if len(names) > SUBSET_K:
        logger.info("Running H3: %d random + 1 greedy subset(s)", N_SUBSETS)
        h3 = _run_h3(config, ds_paths, names)

    return AnalysisState(
        acc,
        fit_split,
        dists,
        ba,
        gate,
        h2_result,
        h1,
        outcome,
        margins,
        float(np.median(margins)),
        h3,
        peer,
    )


def _write_outputs(config: dict[str, Any], state: AnalysisState) -> Path:
    """Write analysis.json (via _write_analysis), diagnostics.json, and (if H3 ran) its scatter."""
    gate_lo, gate_hi, d_sim_100, gate_pass = state.gate
    h2, onset_sim, onset_observed = state.h2_result
    path = _write_analysis(config, state.acc, state.fit_split, state.dists)
    diagnostics: dict[str, Any] = {
        "gate": {
            "pass": gate_pass,
            "point": d_sim_100,
            "observed_ci": [gate_lo, gate_hi],
        },
        "label": state.outcome,
        "h1_gap": state.h1,
        "h2_onset": {"D_sim": onset_sim, "observed_P": onset_observed},
        "h2_agreement": h2,
        "own_r1_ba": float(state.ba["r1"][0]),
        "own_median_margin": state.own_median_margin,
    }
    write_json(path.with_name("diagnostics.json"), diagnostics)
    if state.h3 is not None:
        write_json(path.with_name("subset_scatter.json"), {"subsets": state.h3})
    return path


def _write_figures(config: dict[str, Any], state: AnalysisState) -> None:
    """Write the onset curve, margin CDF, and (if H3 ran) the subset scatter figure."""
    figures = output_root(config) / "figures"
    onset_figure(state.dists, figures / "onset_curve.pdf")
    margin_cdf_figure(
        {config["dataset"]["name"]: state.margins},
        MARGIN_RHOS,
        figures / "margin_cdf.pdf",
    )
    if state.h3 is None:
        return
    bracs_point = None
    if state.peer is not None:
        peer_analysis, peer_diag, _ = state.peer
        bracs_point = (
            peer_diag["own_r1_ba"],
            peer_analysis["estimates"]["D_sim_100"]["point"],
            peer_diag["own_median_margin"],
        )
    subset_scatter_figure(state.h3, bracs_point, figures / "subset_scatter.pdf")


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool Q/QT arm accuracy, run the gate and H1-H3, and write analysis, diagnostics, figures."""
    names = canonical_class_names(config)
    ds_config = baseline_config(config, "prevalence_outputs")
    ds_paths = {s: split_paths(ensure_dirs(ds_config), s) for s in range(N_SPLITS)}

    state = _build_state(config, ds_config, ds_paths, names)
    path = _write_outputs(config, state)
    _write_figures(config, state)
    return path
