"""Frozen validation-only selection: candidate ranking, fingerprint, and fingerprinted on-disk record."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, NamedTuple, cast

import numpy as np
from centre.cohort import TrainingTable

from uncertainty import (
    COVARIANCE_KINDS,
    FIXED_T,
    LAMBDAS,
    MAX_ITER,
    TIE_TOLERANCE,
    TOLERANCE,
)
from uncertainty.loss import UncertainFit

logger = logging.getLogger(__name__)

__all__ = [
    "Candidate",
    "SELECTION_NAME",
    "WINNERS_NAME",
    "fingerprint",
    "select_arms",
    "select_candidates",
    "write_selection",
    "read_selection",
    "t_grid",
]

logger = logging.getLogger(__name__)

SELECTION_NAME = "selection.json"
WINNERS_NAME = "winners.npz"


class Candidate(NamedTuple):
    """One fitted (covariance, t, lambda) with its validation score and solver diagnostics."""

    kind: str
    t: float
    lam: float
    score: float
    diag: dict[str, Any]


def select_candidates(
    candidates: list[Candidate], baseline_score: float | None
) -> Candidate | None:
    """Best converged candidate by validation score; ties go to smaller t, then larger lambda.

    ``baseline_score`` is the t = 0 arm (R); if it wins, ``None`` is returned so the caller reuses
    R's record. Pass ``None`` to exclude the baseline.
    """
    best: Candidate | None = None
    best_score = -np.inf if baseline_score is None else baseline_score
    for cand in sorted(candidates, key=lambda c: (c.t, -c.lam)):
        if cand.diag["converged"] and cand.score > best_score + TIE_TOLERANCE:
            best, best_score = cand, cand.score
    if best is None and baseline_score is None:
        raise RuntimeError("No candidate converged during validation tuning")
    return best


def fingerprint(
    config: dict[str, Any],
    table: TrainingTable,
    g: int,
    source: dict[str, Any],
    t_grid: tuple[float, ...],
) -> str:
    """Hash of everything a frozen selection depends on: config, cohort features, grids, source records."""
    meta = {
        "dataset": config.get("dataset", {}),
        "feature_extraction": config.get("feature_extraction", {}),
        "g": g,
        "t": t_grid,
        "lambdas": LAMBDAS,
        "tol": TOLERANCE,
        "max_iter": MAX_ITER,
        "kinds": COVARIANCE_KINDS,
        "source": source,
    }
    h = hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode())
    h.update(np.ascontiguousarray(table.y).tobytes())
    h.update(np.ascontiguousarray(table.x).tobytes())
    return h.hexdigest()


def _pick(
    kind: str, pool: list[Candidate], baseline: float | None
) -> dict[str, Any] | None:
    """Winner among ``kind`` candidates as a plain dict, or None when the baseline wins."""
    best = select_candidates([c for c in pool if c.kind == kind], baseline)
    return None if best is None else {"kind": kind, "t": best.t, "lam": best.lam}


def select_arms(
    candidates: list[Candidate], r_score: float
) -> dict[str, dict[str, Any] | None]:
    """Arm family -> chosen (kind, t, lambda), or None where the t = 0 baseline R wins."""
    fixed = [c for c in candidates if c.t == FIXED_T]
    return {
        "U": _pick("patient", fixed, None),
        "Ut": _pick("patient", candidates, r_score),
        "It": _pick("isotropic", candidates, r_score),
    }


def write_selection(
    out_dir: Path,
    fp: str,
    candidates: list[Candidate],
    selected: dict[str, dict[str, Any] | None],
    geometry: dict[str, Any],
    r_score: float,
    by_key: dict[tuple[str, float, float], UncertainFit],
) -> None:
    """Freeze validation-only evidence: winners' coefficients first, the fingerprinted record last."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for family, sel in selected.items():
        if sel is not None:
            fit = by_key[(sel["kind"], sel["t"], sel["lam"])]
            arrays[f"{family}_coef"] = fit.coef
            arrays[f"{family}_intercept"] = fit.intercept
    np.savez_compressed(out_dir / WINNERS_NAME, **cast(Any, arrays))
    payload = {
        "fingerprint": fp,
        "geometry": geometry,
        "r_validation_score": r_score,
        "selected": selected,
        "candidates": [c._asdict() for c in candidates],
    }
    tmp = out_dir / (SELECTION_NAME + ".tmp")
    logger.info("Freezing selection %s", out_dir)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, out_dir / SELECTION_NAME)


def t_grid(record: dict[str, Any]) -> tuple[float, ...]:
    """Strengths a frozen selection covers: t = 0 (the baseline) plus every fitted candidate's t."""
    return tuple(sorted({0.0} | {c["t"] for c in record["candidates"]}))


def read_selection(
    out_dir: Path, fp_for: Callable[[tuple[float, ...]], str]
) -> dict[str, Any] | None:
    """Frozen selection matching ``fp_for(its own t grid)``; ``None`` if absent. Raises on a stale or damaged one."""
    path = out_dir / SELECTION_NAME
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    winners = out_dir / WINNERS_NAME
    if record.get("fingerprint") != fp_for(t_grid(record)) or not winners.exists():
        raise RuntimeError(
            f"Stale or incomplete selection in {out_dir}; delete to refit"
        )
    needed = {
        f"{fam}_{part}"
        for fam, sel in record["selected"].items()
        if sel
        for part in ("coef", "intercept")
    }
    with np.load(winners) as npz:
        if needed - set(npz.files):
            raise RuntimeError(f"Incomplete winners in {out_dir}; delete to refit")
    return record
