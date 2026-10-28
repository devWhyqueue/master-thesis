"""Reproduction gate and paired noninferiority of every method's temperature-scaled NLL against CE.

Everything is read from stored run records: the validation-selected candidates
(``analyze.select``), the case-macro NLL/ECE bootstrap distributions
(``analyze.bootstrap.quality_distributions``) and the shared draw weights (exp-40's recipe), so each
contrast is the difference of two paired distributions. Intervals use the replicate columns only;
column zero holds the observed estimate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from analyze.bootstrap import quality_distributions
from analyze.common import FAMILY, METHODS, _setup, imbalance_arms
from analyze.paths import fixed_dirs, run_dir
from analyze.select import select

from breadth import BOOTSTRAP_SEED

from centre import N_DRAWS, N_SPLITS
from centre.analyze import pooled

from decomposition.model import draw_weights

from ts_nll import load_source, source_report

logger = logging.getLogger(__name__)

__all__ = ["verdict", "joint_verdict", "run_gate", "run_analyze"]

GATE_TOLERANCE = 1e-6
REPORT_DIR = Path(__file__).resolve().parents[2] / "report"
R1_CE = "r1_ce"


def verdict(distribution: np.ndarray, margin: float, alpha: float) -> dict[str, Any]:
    """Interval of a delta (observed estimate at index zero) and its three-way decision."""
    lower, upper = (
        float(q) for q in np.quantile(distribution[1:], (alpha / 2, 1 - alpha / 2))
    )
    if lower > -margin:
        decision = "no_meaningful_gain"
    elif upper < -margin:
        decision = "method_better"
    else:
        decision = "inconclusive"
    return {
        "estimate": float(distribution[0]),
        "lower": lower,
        "upper": upper,
        "decision": decision,
    }


def joint_verdict(decisions: dict[str, str]) -> str:
    """The claim is confirmed only if every contrast is ``no_meaningful_gain``."""
    ok = all(d == "no_meaningful_gain" for d in decisions.values())
    return "confirmed" if ok else "not_confirmed"


def _interval(distribution: np.ndarray) -> dict[str, float]:
    lower, upper = (float(q) for q in np.quantile(distribution[1:], (0.025, 0.975)))
    return {"estimate": float(distribution[0]), "lower": lower, "upper": upper}


def _pooled_quality(
    config: dict[str, Any], arm: str
) -> dict[str, dict[str, np.ndarray]]:
    """Pooled (R,) quality distributions of CE and each validation-selected method at one arm."""
    _, n_classes, paths, ctxs, _ = _setup(config)
    fit_split = np.repeat(np.arange(N_SPLITS), N_DRAWS)
    w = draw_weights(
        fit_split,
        N_DRAWS,
        ctxs[0].n_replicates,
        np.random.default_rng(BOOTSTRAP_SEED),
    )
    dirs = {
        R1_CE: fixed_dirs(paths, "r1", "ce", None),
        f"{arm}_ce": fixed_dirs(paths, arm, "ce", None),
    }
    for m in METHODS:
        selection = select(config, paths, arm, FAMILY[m], m)
        dirs[m] = {
            k: run_dir(paths[k[0]], arm, m, p, k[1]) for k, p in selection.items()
        }
    return {
        label: {
            key: pooled(dist, w)
            for key, dist in quality_distributions(d, ctxs, n_classes).items()
        }
        for label, d in dirs.items()
    }


def run_gate(protocol: dict[str, Any]) -> None:
    """Fail unless every recomputed ``nll_ts`` point equals the source report's."""
    for source in protocol["sources"]:
        config = load_source(source)
        stored = json.loads(source_report(source).read_text(encoding="utf-8"))["arms"]
        for arm in imbalance_arms(config):
            for label, quality in _pooled_quality(config, arm).items():
                expected = stored[arm]["probability_quality"][label]["nll_ts"]["point"]
                diff = abs(float(quality["nll_ts"][0]) - expected)
                if diff >= GATE_TOLERANCE:
                    raise SystemExit(
                        f"gate failed: {source} {arm}/{label} diff {diff:.3e}"
                    )
        logger.info(f"gate ok: {source}")


def _setting(config: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    margin = protocol["protocol"]["margin_nats"]
    alpha = 0.05 / (len(METHODS) * len(protocol["sources"]))
    arms = {}
    for arm in imbalance_arms(config):
        q = _pooled_quality(config, arm)
        ce = q[f"{arm}_ce"]
        methods = {}
        for m in METHODS:
            delta = q[m]["nll_ts"] - ce["nll_ts"]
            methods[m] = {
                "nll_ts_delta": (
                    verdict(delta, margin, alpha)
                    if arm == protocol["protocol"]["primary_arm"]
                    else _interval(delta)
                ),
                "nll_minus_ce_ts": _interval(q[m]["nll"] - ce["nll_ts"]),
                "ece_ts_delta": _interval(q[m]["ece_ts"] - ce["ece_ts"]),
            }
        arms[arm] = {"ce_nll_ts": float(ce["nll_ts"][0]), "methods": methods}
    return arms


def _summarize(protocol: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    primary = protocol["protocol"]["primary_arm"]
    contrasts = {
        f"{name}/{m}": arms[primary]["methods"][m]["nll_ts_delta"]
        for name, arms in settings.items()
        for m in METHODS
    }
    decisions = {k: v["decision"] for k, v in contrasts.items()}
    return {
        "protocol": protocol,
        "settings": settings,
        "decisions": decisions,
        "joint_verdict": joint_verdict(decisions),
        "largest_positive_delta": max(
            ({"contrast": k, **v} for k, v in contrasts.items()),
            key=lambda c: c["estimate"],
        ),
    }


def run_analyze(protocol: dict[str, Any]) -> Path:
    """Write ``report/analysis.json``: 32 primary decisions, the joint verdict, descriptive secondaries."""
    settings = {
        Path(s).parents[1].name: _setting(load_source(s), protocol)
        for s in protocol["sources"]
    }
    analysis = _summarize(protocol, settings)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "analysis.json"
    path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    return path
