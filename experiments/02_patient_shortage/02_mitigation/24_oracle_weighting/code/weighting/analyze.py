"""Analyze stage: pool reused and oracle-weighted arms with the exp-24 ladder contrasts and diagnostics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import (
    ensure_dirs,
    read_run_record,
    split_paths,
    write_json,
)

from breadth import BOOTSTRAP_SEED
from breadth.analyze.canonical import canonical_class_names

from decomposition.model import draw_weights

from sites import allocation_dir

from centre import N_DRAWS, N_SPLITS
from centre.analyze import arm_accuracy, pooled

from directions import KAPPA_FACTORS
from directions.analyze import _write_analysis

from weighting import ARMS, G, REUSED_ARM_FAMILIES, baseline_config

__all__ = ["derived", "run_analyze"]

_NEW = ("Wo", "Ro", "Po")
_CONTRASTS = (
    ("Wo", "Wt"),  # primary: oracle weights vs exp-23 patch-level weights
    ("Wo", "Ro"),  # value of the within-patient directions once weights are correct
    ("Po", "Wo"),  # direction loss of V against the pool's best complement directions
    ("RW", "Po"),  # cost of keeping the cohort block and spectrum
)


def derived(arm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Gains over RWc, ladder contrasts, and headroom shares (x - RWc) / (RW - RWc) at G = 5."""
    out: dict[str, np.ndarray] = {}
    for fam in _NEW:
        out[f"{fam}{G}_minus_RWc{G}"] = arm[f"{fam}{G}"] - arm[f"RWc{G}"]
    for a, b in _CONTRASTS:
        out[f"{a}{G}_minus_{b}{G}"] = arm[f"{a}{G}"] - arm[f"{b}{G}"]
    gap = arm[f"RW{G}"] - arm[f"RWc{G}"]
    out[f"RW{G}_minus_RWc{G}"] = gap
    for fam in ("Wt", *_NEW):
        out[f"headroom_share_{fam}{G}"] = (arm[f"{fam}{G}"] - arm[f"RWc{G}"]) / gap
    return out


def _records(config: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    """Run records of one arm over all (split, draw) fits, without arrays."""
    paths = {s: split_paths(ensure_dirs(config), s) for s in range(N_SPLITS)}
    records = [
        read_run_record(allocation_dir(paths[s], arm, d), array_fields=())
        for s in range(N_SPLITS)
        for d in range(N_DRAWS)
    ]
    return [r for r in records if r is not None]


def _kappa_factor(record: dict[str, Any]) -> float:
    """Unscaled kappa factor: the anchored factor's position in the score dict indexes ``KAPPA_FACTORS``."""
    scores = list(record["kappa_validation_scores"])
    return KAPPA_FACTORS[scores.index(str(record["kappa_factor"]))]


def _diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    """Kappa counts, added rank, oracle-implied s, added pool share, and Spearman(omega, oracle) per arm."""
    diag: dict[str, Any] = {}
    for fam in _NEW:
        recs = _records(config, f"{fam}{G}")
        diag[f"kappa_counts_{fam}{G}"] = dict(
            Counter(str(_kappa_factor(r)) for r in recs)
        )
        for key in ("added_rank", "oracle_implied_s", "added_pool_share"):
            diag[f"mean_{key}_{fam}{G}"] = float(np.mean([r[key] for r in recs]))
    wo = _records(config, f"Wo{G}")
    diag[f"mean_spearman_omega_oracle_{G}"] = float(
        np.mean([r["spearman_omega_oracle"] for r in wo])
    )
    return diag


def run_analyze(config: dict[str, Any]) -> Path:
    """Pool reused and new arm accuracy over splits and draws and write estimates with 95% intervals."""
    names = canonical_class_names(config)
    acc: dict[str, np.ndarray] = {}
    for key, families in REUSED_ARM_FAMILIES.items():
        arms = tuple(f"{fam}{G}" for fam in families)
        acc.update(arm_accuracy(baseline_config(config, key), names, arms=arms))
    acc.update(arm_accuracy(config, names, arms=ARMS))
    n_replicates = next(iter(acc.values())).shape[-1]
    fit_split = np.repeat(np.arange(N_SPLITS), N_DRAWS)
    w = draw_weights(
        fit_split, N_DRAWS, n_replicates, np.random.default_rng(BOOTSTRAP_SEED)
    )
    arm = {name: pooled(a, w) for name, a in acc.items()}
    dists = {f"arm_{name}": dist for name, dist in arm.items()} | derived(arm)
    path = _write_analysis(config, acc, fit_split, dists)
    write_json(path.with_name("diagnostics.json"), _diagnostics(config))
    return path
