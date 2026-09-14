"""Census stage: cohort search and the manipulation check (report Sec. "check")."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from imbalance_benchmark.common import (
    output_root,
    sign_file,
    verify_signed_file,
    write_json,
)

from breadth.analyze.canonical import canonical_class_names
from breadth.sampling import derive_draw_seed

from similarity.candidates import candidates
from similarity.census import _load_inputs
from similarity.geometry import (
    ClassGeometry,
    SplitInputs,
    build_split_geometry,
    omega_of,
    r_of,
)
from similarity.search import _corner_loss_fn, _search

from decomposition import N_DRAWS_DEFAULT
from decomposition.checks import manipulation_check
from decomposition.design import CELLS, N_CELLS, SEARCH_BASE_SEED, offsets, targets

__all__ = ["run_census_shard", "load_split", "load_allocations"]


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


def _class_draw_row(
    geo: ClassGeometry,
    split_idx: int,
    draw_idx: int,
    c_idx: int,
    c_name: str,
    cell_idx: int,
) -> dict[str, Any]:
    """One (split, draw, class)'s cohort: searched if designed, drawn if random."""
    seed5 = derive_draw_seed(SEARCH_BASE_SEED, split_idx, 5, 32, draw_idx, c_idx)
    seed10 = derive_draw_seed(SEARCH_BASE_SEED, split_idx, 10, 16, draw_idx, c_idx)
    cands5 = candidates(geo, 5, seed5)
    cands10 = candidates(geo, 10, seed10)
    tg = targets(geo, cands5, cands10)
    cell = CELLS[cell_idx]
    cands_g = cands5 if cell.g == 5 else cands10

    if cell.r_level is None or cell.omega_level is None:
        idx = cands_g[0]
        r_target = omega_target = None
    else:
        r_target = tg.r_levels[cell.r_level]
        omega_target = tg.omega_levels[cell.omega_level]
        idx = _search(cands_g, geo, _corner_loss_fn(r_target, omega_target))

    return {
        "draw": draw_idx,
        "class": c_name,
        "cell": cell_idx,
        "g": cell.g,
        "m": cell.m,
        "r_level": cell.r_level,
        "omega_level": cell.omega_level,
        "r_target": r_target,
        "omega_target": omega_target,
        "delta_prime": tg.delta_prime,
        "r_train": r_of(geo.d_pool, idx),
        "r_val": r_of(geo.d_val, idx),
        "omega": omega_of(geo, idx),
        "patients": [geo.pool[i] for i in idx],
    }


def _split_rows(
    config: dict[str, Any],
    split_idx: int,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    shared: SplitInputs,
    canonical_names: list[str],
    n_draws: int,
) -> list[dict[str, Any]]:
    """Every (draw, class) cohort of one split, assigned to cells by its offset."""
    geometry_by_class = build_split_geometry(
        config, split_idx, full_df, train_df, shared
    )
    offs = offsets(split_idx, len(canonical_names))
    rows: list[dict[str, Any]] = []
    for draw_idx in range(n_draws):
        for c_idx, c_name in enumerate(canonical_names):
            cell_idx = int((offs[c_idx] + draw_idx) % N_CELLS)
            rows.append(
                _class_draw_row(
                    geometry_by_class[c_name],
                    split_idx,
                    draw_idx,
                    c_idx,
                    c_name,
                    cell_idx,
                )
            )
    return rows


def _write_split_outputs(
    config: dict[str, Any],
    split_idx: int,
    n_draws: int,
    check: dict[str, Any],
    rows: list[dict[str, Any]],
) -> Path:
    census_dir = output_root(config) / "data" / "census"
    split_p = census_dir / f"split_{split_idx}.json"
    write_json(split_p, {"split": split_idx, "draws": n_draws, "check": check})
    sign_file(split_p)

    allocations_p = census_dir / f"allocations_split_{split_idx}.json"
    write_json(allocations_p, {"split": split_idx, "draws": n_draws, "rows": rows})
    sign_file(allocations_p)
    return split_p


def _gate(check: dict[str, Any], split_idx: int) -> None:
    if not check["pass"]:
        raise RuntimeError(
            f"Manipulation check failed for split {split_idx}; ending the experiment "
            "before fitting, with the original tolerances and cell grid retained."
        )


def run_census_shard(config: dict[str, Any], split_idx: int) -> Path:
    """Search one split's cohorts and run its manipulation check; write, sign, and gate.

    Writes before gating so a failing split is still inspectable, and raises
    afterwards so the array job's failure blocks the downstream ``afterok``
    precision stage.
    """
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


def load_split(config: dict[str, Any], split_idx: int) -> dict[str, Any]:
    """Load and verify one split's signed manipulation-check summary."""
    split_p = output_root(config) / "data" / "census" / f"split_{split_idx}.json"
    verify_signed_file(split_p)
    return json.loads(split_p.read_text(encoding="utf-8"))


def load_allocations(config: dict[str, Any], split_idx: int) -> dict[str, Any]:
    """Load and verify one split's signed cohort rows."""
    allocations_p = (
        output_root(config) / "data" / "census" / f"allocations_split_{split_idx}.json"
    )
    verify_signed_file(allocations_p)
    return json.loads(allocations_p.read_text(encoding="utf-8"))
