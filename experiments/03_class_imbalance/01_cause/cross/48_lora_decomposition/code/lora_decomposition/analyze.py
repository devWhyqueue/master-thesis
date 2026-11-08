"""Paired four-arm contrasts, the conditional recovery share, and uncertainty diagnostics.

Each fit's test patient-macro balanced accuracy comes from exp-40's ``selected_ba`` over the
crossed patient bootstrap (one weight per patient across the overlapping test splits; 10,000
replicates). Draws are resampled within split with one set of draw weights shared by all four
arms, so every contrast is a difference of paired distributions. Column zero holds the observed
estimate and never enters an interval. Intervals are marginal 95% estimation intervals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.analysis.inference.context import BootstrapContext

from analyze.bootstrap import selected_ba
from analyze.paths import fixed_dirs, result_paths, shard_keys

from breadth import BOOTSTRAP_SEED, exp2_split_paths
from breadth.analyze.canonical import canonical_class_names, canonical_permutation

from centre import N_DRAWS, N_SPLITS
from centre.analyze import pooled

from decomposition.model import draw_weights

from lora_decomposition import protocol
from lora_decomposition.fit import validate_run
from lora_decomposition.protocol import ARMS, PROTOCOL_ID, config_digest, is_smoke
from lora_decomposition.provenance import source_digest

__all__ = [
    "N_REPLICATES",
    "analyze_dataset",
    "arm_distributions",
    "contrasts",
    "require_complete",
    "run_analyze",
    "share_summary",
    "summarize",
    "summarize_dataset",
]

N_REPLICATES = 10_000
MATERIALITY_PP = 1.0
_COMPONENTS = ("D", "D_F", "D_C", "I", "G")
_INTERPRETATION = (
    "Matched LoRA decomposition under one fixed training recipe (fixed splits, one seed per "
    "allocation draw, so draw variation includes initialization). h = G/D is the recovery "
    "benchmark for restoring balanced class influence, conditional on this learner and recipe; "
    "not an information limit or an upper bound on other methods. Intervals are marginal 95% "
    "estimation intervals, not a simultaneous test of every component."
)


def contrasts(ba: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Positive-damage contrasts of the four arms (pp) and the share h = G / D (signed)."""
    out = {
        "D": ba["B"] - ba["R"],
        "D_F": ba["B"] - ba["F"],
        "D_C": ba["B"] - ba["C"],
        "G": ba["C"] - ba["R"],
    }
    out["I"] = out["D"] - out["D_F"] - out["D_C"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["h"] = out["G"] / out["D"]
    return out


def summarize(distribution: np.ndarray) -> dict[str, float]:
    """Observed estimate (index 0) and the 95% interval of the replicates (index 1 on)."""
    values = np.asarray(distribution, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("need a finite 1-D distribution: estimate, then replicates")
    lower, upper = np.quantile(values[1:], (0.025, 0.975))
    return {"estimate": float(values[0]), "lower": float(lower), "upper": float(upper)}


def share_summary(damage: np.ndarray, recovery: np.ndarray) -> dict[str, Any]:
    """Share status: usable only for D >= 1 pp with a strictly positive interval; never cleaned."""
    damage, recovery = np.asarray(damage, float), np.asarray(recovery, float)
    reasons = []
    if damage[0] < MATERIALITY_PP:
        reasons.append("D point estimate below the 1 pp materiality threshold")
    if np.quantile(damage[1:], 0.025) <= 0:
        reasons.append("95% interval of D is not strictly positive")
    zeros = int(np.sum(damage == 0))
    interval = None
    if zeros:
        reasons.append("ratio undefined: a denominator is exactly zero")
    else:
        interval = summarize(recovery / damage)
    return {
        "usable_for_hypothesis": not reasons,
        "reason": "; ".join(reasons) or None,
        "nonpositive_denominators": int(np.sum(damage[1:] <= 0)),
        "zero_denominators": zeros,
        "interval": interval,
    }


def arm_distributions(
    per_fit: dict[str, np.ndarray], w: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    """Full paired, patient-only (equal draws), and draw-only (observed patients) pooling."""
    ones = np.ones_like(w)
    return {
        "full": {arm: pooled(a, w) for arm, a in per_fit.items()},
        "patient_only": {arm: pooled(a, ones) for arm, a in per_fit.items()},
        "draw_only": {
            arm: pooled(np.broadcast_to(a[:, :1], a.shape), w)
            for arm, a in per_fit.items()
        },
    }


def _block(ba: dict[str, np.ndarray]) -> dict[str, Any]:
    """Accuracies, contrast intervals, and share status of one set of pooled distributions."""
    parts = contrasts(ba)
    return {
        "accuracy": {arm: summarize(ba[arm]) for arm in ARMS},
        "contrasts": {key: summarize(parts[key]) for key in _COMPONENTS},
        "share": share_summary(parts["D"], parts["G"]),
    }


def summarize_dataset(
    per_fit: dict[str, np.ndarray], w: np.ndarray, fit_split: np.ndarray
) -> dict[str, Any]:
    """Full results, per-split results, and patient-only/draw-only diagnostics of one dataset."""
    if set(per_fit) != set(ARMS):
        raise ValueError(f"need all arms {ARMS}, got {sorted(per_fit)}")
    dists = arm_distributions(per_fit, w)
    per_split = {
        str(s): _block(
            {
                arm: pooled(a[fit_split == s], w[fit_split == s])
                for arm, a in per_fit.items()
            }
        )
        for s in np.unique(fit_split)
    }
    return {
        **_block(dists["full"]),
        "per_split": per_split,
        "diagnostics": {
            kind: _block(dists[kind]) for kind in ("patient_only", "draw_only")
        },
    }


def require_complete(config: dict[str, Any]) -> None:
    """Raise unless every arm of every block has a compatible completed run."""
    if is_smoke(config):
        raise ValueError("smoke runs are never analyzed")
    missing = [
        (s, d, arm)
        for s, d in shard_keys()
        for arm in ARMS
        if not validate_run(config, s, d, arm)
    ]
    if missing:
        raise ValueError(f"{len(missing)} runs missing or incomplete: {missing[:12]}")


def _per_fit_ba(config: dict[str, Any]) -> tuple[dict[str, np.ndarray], int]:
    """(30, R+1) test BA per arm over the crossed patient bootstrap, and the column count."""
    names = canonical_class_names(config)
    ctxs = {
        s: BootstrapContext(
            exp2_split_paths(config, s), False, N_REPLICATES, BOOTSTRAP_SEED
        )
        for s in range(N_SPLITS)
    }
    perms = {s: canonical_permutation(config, s, names) for s in range(N_SPLITS)}
    paths = result_paths(config)
    per_fit = {
        arm: selected_ba(fixed_dirs(paths, arm, "ce", None), ctxs, perms, len(names))
        for arm in ARMS
    }
    return per_fit, ctxs[0].n_replicates


def analyze_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Paired bootstrap results of one dataset's 30 complete four-arm blocks."""
    require_complete(config)
    per_fit, n_columns = _per_fit_ba(config)
    fit_split = np.repeat(np.arange(N_SPLITS), N_DRAWS)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    w = draw_weights(fit_split, N_DRAWS, n_columns, rng)
    return {
        "config_sha256": config_digest(config),
        "n_blocks": len(shard_keys()),
        "n_fits": len(shard_keys()) * len(ARMS),
        "n_replicates": N_REPLICATES,
        **summarize_dataset(per_fit, w, fit_split),
    }


def run_analyze(configs: list[dict[str, Any]]) -> Path:
    """Analyze BRACS and TCGA-UT together; write ``report/analysis.json``."""
    by_name = {c["dataset"]["name"]: c for c in configs}
    if sorted(by_name) != ["bracs", "tcga_ut"] or len(configs) != 2:
        raise ValueError(
            f"analyze needs exactly bracs and tcga_ut, got {sorted(by_name)}"
        )
    result = {
        "protocol_id": PROTOCOL_ID,
        "source_sha256": source_digest(),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "materiality_pp": MATERIALITY_PP,
        "interpretation": _INTERPRETATION,
        "datasets": {name: analyze_dataset(c) for name, c in sorted(by_name.items())},
    }
    path = protocol.EXPERIMENT_DIR / "report" / "analysis.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return path
