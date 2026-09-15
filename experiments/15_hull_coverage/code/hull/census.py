"""Census stage: cohort search for every (split, draw, class) and the manipulation check."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imbalance_benchmark.common import output_root, verify_signed_file

from breadth.analyze.canonical import canonical_class_names
from breadth.sampling import derive_draw_seed

from similarity.candidates import candidates
from similarity.census import _load_inputs
from similarity.geometry import SplitInputs

from decomposition.census import _gate, _write_split_outputs, load_allocations

from hull import N_DRAWS_DEFAULT, N_SPLITS, SEARCH_BASE_SEED
from hull.checks import manipulation_check
from hull.design import CELLS, Cell, Target, Targets, cell_target, offsets, targets
from hull.geometry import HullGeometry, build_hull_geometry, cohort_values
from hull.search import search_cohort

__all__ = ["run_census_shard", "run_recheck"]

logger = logging.getLogger(__name__)


def _n_draws(config: dict[str, Any]) -> int:
    """Draws per split: from a signed precision.json if present, else the default."""
    precision_p = output_root(config) / "data" / "precision.json"
    if not precision_p.exists():
        return N_DRAWS_DEFAULT
    verify_signed_file(precision_p)
    selected = json.loads(precision_p.read_text(encoding="utf-8"))["selected_draws"]
    if not selected:
        raise RuntimeError(
            "precision.json has no selected_draws; cannot run the census"
        )
    return int(selected)


def _cohort_row(
    geo: HullGeometry,
    idx: np.ndarray,
    cell: Cell,
    tg: Targets,
    target: Target | None,
    key: tuple[int, str],
) -> dict[str, Any]:
    values = cohort_values(geo, np.asarray(idx)[None, :])
    return {
        "draw": key[0],
        "class": key[1],
        **cell._asdict(),
        "r_target": None if target is None else target.r,
        "h_target": None if target is None else target.h,
        "omega_target": None if target is None else target.omega,
        "delta_r": tg.delta_r,
        "delta_h": tg.delta_h,
        **{name: float(v[0]) for name, v in values._asdict().items()},
        "patients": [geo.base.pool[i] for i in idx],
    }


def _cell_row(
    geo: HullGeometry,
    cands: list[np.ndarray],
    cell: Cell,
    tg: Targets,
    key: tuple[int, str],
) -> dict[str, Any]:
    """A searched cohort for a designed cell; the first (random) candidate for a random cell."""
    if cell.mean_level is None:
        return _cohort_row(geo, cands[0], cell, tg, None, key)
    target = cell_target(tg, cell)
    return _cohort_row(geo, search_cohort(geo, cands, target), cell, tg, target, key)


def _class_draw_rows(
    geo: HullGeometry,
    split_idx: int,
    key: tuple[int, str],
    c_idx: int,
    cells: dict[int, Cell],
) -> list[dict[str, Any]]:
    """The G=5 and G=10 cell cohorts and the random G=20 cohort of one (split, draw, class)."""
    draw_idx = key[0]
    seeds = {
        g: derive_draw_seed(
            SEARCH_BASE_SEED, split_idx, g, CELLS[g][0].m, draw_idx, c_idx
        )
        for g in CELLS
    }
    cands = {g: candidates(geo.base, g, seeds[g]) for g in (5, 10)}
    tg = targets(geo, cands[5], cands[10])
    rows = [_cell_row(geo, cands[g], cells[g], tg, key) for g in (5, 10)]
    pool_size = len(geo.base.pool)
    if pool_size < 20:
        raise RuntimeError(f"Class {key[1]} has only {pool_size} eligible patients")
    idx20 = np.random.default_rng(seeds[20]).choice(pool_size, size=20, replace=False)
    rows.append(_cohort_row(geo, idx20, CELLS[20][0], tg, None, key))
    return rows


def _split_rows(
    config: dict[str, Any],
    split_idx: int,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    shared: SplitInputs,
    canonical_names: list[str],
    n_draws: int,
) -> list[dict[str, Any]]:
    """Every (draw, class) cohort of one split, assigned to cells by rotating offsets."""
    geometry = build_hull_geometry(config, split_idx, full_df, train_df, shared)
    offs = {g: offsets(split_idx, g, len(canonical_names)) for g in (5, 10)}
    rows: list[dict[str, Any]] = []
    for draw_idx in range(n_draws):
        logger.info("Census split %d, draw %d", split_idx, draw_idx)
        for c_idx, c_name in enumerate(canonical_names):
            cells = {
                g: CELLS[g][int((offs[g][c_idx] + draw_idx) % len(CELLS[g]))]
                for g in (5, 10)
            }
            rows += _class_draw_rows(
                geometry[c_name], split_idx, (draw_idx, c_name), c_idx, cells
            )
    return rows


def run_census_shard(config: dict[str, Any], split_idx: int) -> Path:
    """Search one split's cohorts and run its manipulation check; write, sign, then gate."""
    n_draws = _n_draws(config)
    canonical_names = canonical_class_names(config)
    _, full_dfs, train_dfs, shared = _load_inputs(config)
    rows = _split_rows(
        config,
        split_idx,
        full_dfs[split_idx],
        train_dfs[split_idx],
        shared,
        canonical_names,
        n_draws,
    )
    check = manipulation_check(rows)
    split_p = _write_split_outputs(config, split_idx, n_draws, check, rows)
    _gate(check, split_idx)
    return split_p


def run_recheck(config: dict[str, Any]) -> list[Path]:
    """Re-evaluate every split's manipulation check on its stored cohorts with the current thresholds."""
    written: list[Path] = []
    for split_idx in range(N_SPLITS):
        record = load_allocations(config, split_idx)
        check = manipulation_check(record["rows"])
        written.append(
            _write_split_outputs(
                config, split_idx, record["draws"], check, record["rows"]
            )
        )
        _gate(check, split_idx)
    return written
