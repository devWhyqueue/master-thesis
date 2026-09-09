"""Constants, path helpers, and addressing for exp-4 class decodability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imbalance_benchmark.common import ensure_dirs, split_paths

__all__ = [
    "SUPPORTS",
    "MLP_ASSIGNMENT",
    "READOUTS",
    "LAMBDAS",
    "K_VALUES",
    "NEIGHBOUR_TOP",
    "TIE_TOLERANCE",
    "BOOTSTRAP_SEED",
    "N_REPLICATES",
    "MIN_TRAIN_PATCHES",
    "RELEVANCE_THRESHOLD_PP",
    "INPUT_DIM",
    "exp2_split_paths",
    "allocation_manifest",
    "probe_dir",
]

SUPPORTS: tuple[str, ...] = ("balanced", "balanced_spread")
MLP_ASSIGNMENT: dict[str, str] = {
    "balanced": "unassigned",
    "balanced_spread": "native",
}
READOUTS: tuple[str, ...] = ("mlp", "logreg", "knn")
LAMBDAS: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
K_VALUES: tuple[int, ...] = (1, 5, 20, 100, 250, 500)
NEIGHBOUR_TOP: int = 500
TIE_TOLERANCE: float = 1e-10  # proportion scale
BOOTSTRAP_SEED: int = 20260909
N_REPLICATES: int = 2000
MIN_TRAIN_PATCHES: int = 100
RELEVANCE_THRESHOLD_PP: float = 1.0
INPUT_DIM: int = 2560


def exp2_split_paths(config: dict[str, Any], split_index: int) -> dict[str, Path]:
    """Resolve exp-2 split paths for an exp-4 config."""
    exp2_out = config.get("slurm", {}).get("exp2_outputs")
    if not exp2_out:
        raise ValueError("Config missing slurm.exp2_outputs")
    base = ensure_dirs({"paths": {"outputs": str(exp2_out)}})
    return split_paths(base, split_index)


def allocation_manifest(exp2_paths: dict[str, Path], support: str) -> Path:
    """Return path to the training allocation manifest under exp-2 outputs."""
    if support == "balanced":
        return exp2_paths["data"] / "manifest_balanced.csv"
    if support == "balanced_spread":
        return exp2_paths["data"] / "manifest_native_balanced_spread.csv"
    raise ValueError(f"Unknown support: {support!r}")


def probe_dir(
    paths: dict[str, Path], support: str, readout: str, param: str | None = None
) -> Path:
    """Return destination directory for probe run records."""
    base = paths["results"] / support / readout
    return base / param if param else base
