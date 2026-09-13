"""Constants, path helpers, and addressing for the patient-influence experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from decodability import (
    BOOTSTRAP_SEED,
    INPUT_DIM,
    MIN_TRAIN_PATCHES,
    N_REPLICATES,
    RELEVANCE_THRESHOLD_PP,
)
from imbalance_benchmark.common import output_root, verify_signed_file

__all__ = [
    "SUPPORTS",
    "OBJECTIVES",
    "BOOTSTRAP_SEED",
    "N_REPLICATES",
    "MIN_TRAIN_PATCHES",
    "RELEVANCE_THRESHOLD_PP",
    "INPUT_DIM",
    "TOLERANCE",
    "MAX_ITER",
    "exp3_root",
    "inherited_lambda",
    "cell_dir",
]

SUPPORTS: tuple[str, ...] = ("balanced", "balanced_spread")
OBJECTIVES: tuple[str, ...] = ("patch", "patient")
TOLERANCE: float = 1e-8
MAX_ITER: int = 10000


def exp3_root(config: dict[str, Any]) -> Path:
    """Resolve exp-3's output root, which holds the frozen patch-average baselines."""
    exp3_out = config.get("slurm", {}).get("exp3_outputs")
    if not exp3_out:
        raise ValueError("Config missing slurm.exp3_outputs")
    return output_root({"paths": {"outputs": str(exp3_out)}})


def inherited_lambda(config: dict[str, Any], support: str) -> tuple[float, str]:
    """Return exp-3's validation-selected lambda and its param string for ``support``."""
    sel_path = exp3_root(config) / "data" / "probe_selection.json"
    verify_signed_file(sel_path)
    selection = json.loads(sel_path.read_text(encoding="utf-8"))
    logreg_sel = selection.get("supports", {}).get(support, {}).get("logreg", {})
    if "selected" not in logreg_sel:
        raise RuntimeError(
            f"Missing selected logreg lambda for {support} in {sel_path}"
        )
    return float(logreg_sel["selected"]), str(logreg_sel["selected_param_str"])


def cell_dir(paths: dict[str, Path], support: str, objective: str) -> Path:
    """Return destination directory for one support/objective result cell."""
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective!r}")
    return paths["results"] / support / objective
