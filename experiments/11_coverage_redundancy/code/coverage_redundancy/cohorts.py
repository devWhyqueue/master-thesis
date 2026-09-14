"""Per-split cohort assembly: random and composition cohort records, guarded."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from decodability.evidence import load_freeze_meta

from breadth import BREADTH_LADDER, DEPTH_LADDER, exp2_split_paths
from breadth.sampling import eligible_patients_by_class, sample_class_patches

from redundancy.correlate import _load_train_manifest

from neighbours.census import load_census
from neighbours.embedding import embed, training_mu

from coverage_redundancy import COMPOSITION_ALLOCATIONS, N_DRAWS, exp10_config
from coverage_redundancy.quantities import (
    SplitContext,
    class_quantities,
    class_tau2,
    cohort_record,
)

__all__ = ["CensusContext", "split_cohorts"]

_MAX_DEPTH = max(DEPTH_LADDER)  # 32: the eligibility floor of mu_rc and the pool
_ICC_TOLERANCE = 1e-6
_R_TOLERANCE = 1e-9


class CensusContext(NamedTuple):
    """Cross-split state that every split's cohort assembly shares."""

    means: dict[tuple[str, str], np.ndarray]
    class_names: list[str]
    rho_by_class: dict[str, float]
    raw_full: np.ndarray
    composition_allocations: dict[str, Any]


def _guard_tau2_icc(
    icc_check: dict[str, float],
    raw_full: np.ndarray,
    class_names: list[str],
    split_idx: int,
) -> None:
    """Guard: the reproduced ones-weighted ICC must match exp-6's stored value."""
    for class_idx, class_name in enumerate(class_names):
        expected = float(raw_full[0, split_idx, class_idx])
        actual = icc_check[class_name]
        if abs(actual - expected) > _ICC_TOLERANCE:
            raise ValueError(
                "Reproduced ICC does not match exp-6 correlations.npz at "
                f"split={split_idx} class={class_name}: {actual} vs {expected}"
            )


def _guard_composition_r(
    config: dict[str, Any], split_idx: int, r_by_allocation: dict[str, list[float]]
) -> None:
    """Guard: our per-class draw-mean r must match exp-10's own census.json."""
    split_summary = load_census(exp10_config(config))["splits"][str(split_idx)]
    for allocation, values in r_by_allocation.items():
        expected = float(split_summary[f"r_{allocation}"])
        actual = float(np.mean(values))
        if abs(actual - expected) > _R_TOLERANCE:
            raise ValueError(
                f"Composition r_{allocation} mismatch at split={split_idx}: "
                f"{actual} vs {expected}"
            )


def _random_cohort_patients(
    ctx: SplitContext, g: int, draw_idx: int
) -> dict[str, list[str]]:
    """Ordered unique cohort patients per class, replaying exp-5's own draw seed."""
    patients_by_class: dict[str, list[str]] = {}
    for c_name in ctx.class_names:
        c_idx = ctx.split_class_names.index(c_name)
        patch_idx = sample_class_patches(
            ctx.train_df,
            ctx.split_class_names,
            c_idx,
            (g, _MAX_DEPTH),
            (ctx.split_idx, draw_idx),
        )
        cases = ctx.train_df.loc[patch_idx, "case_id"].astype(str).tolist()
        ordered: list[str] = []
        seen: set[str] = set()
        for case in cases:
            if case not in seen:
                seen.add(case)
                ordered.append(case)
        patients_by_class[c_name] = ordered
    return patients_by_class


def _random_cohorts_for_split(ctx: SplitContext) -> list[dict[str, Any]]:
    """Every (G, m, draw) random cohort record of one split."""
    cohorts = []
    for g in BREADTH_LADDER:
        for draw_idx in range(N_DRAWS):
            patients_by_class = _random_cohort_patients(ctx, g, draw_idx)
            per_class = {
                c: class_quantities(ctx, c, patients_by_class[c])
                for c in ctx.class_names
            }
            for m in DEPTH_LADDER:
                cohorts.append(
                    cohort_record(ctx, "random", g, m, None, draw_idx, per_class)
                )
    return cohorts


def _composition_cohorts_for_split(
    ctx: SplitContext, composition_allocations: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    """Every (allocation, draw) composition cohort record of one split."""
    cohorts = []
    r_by_allocation: dict[str, list[float]] = {
        name: [] for name in COMPOSITION_ALLOCATIONS
    }
    for draw_idx in range(N_DRAWS):
        draw_alloc = composition_allocations[str(ctx.split_idx)]["allocations"][
            str(draw_idx)
        ]
        for allocation in COMPOSITION_ALLOCATIONS:
            per_class = {
                c: class_quantities(ctx, c, draw_alloc[c][allocation])
                for c in ctx.class_names
            }
            r_by_allocation[allocation].extend(
                per_class[c]["r"] for c in ctx.class_names
            )
            cohorts.append(
                cohort_record(
                    ctx, "composition", 20, 8, allocation, draw_idx, per_class
                )
            )
    return cohorts, r_by_allocation


def _build_split_context(
    config: dict[str, Any],
    split_idx: int,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    means: dict[tuple[str, str], np.ndarray],
    class_names: list[str],
    rho_by_class: dict[str, float],
    raw_full: np.ndarray,
) -> SplitContext:
    """Load one split's own class order, embeddings, tau^2, and eligible pools."""
    split_class_names = list(
        load_freeze_meta(exp2_split_paths(config, split_idx))["class_names"]
    )
    embeddings = embed(means, training_mu(means, train_df))
    tau2_by_class, icc_check = class_tau2(
        _load_train_manifest(config, split_idx), class_names, split_idx
    )
    _guard_tau2_icc(icc_check, raw_full, class_names, split_idx)
    eligible_by_class = {
        c: eligible_patients_by_class(train_df, c, _MAX_DEPTH) for c in class_names
    }
    return SplitContext(
        split_idx,
        full_df,
        train_df,
        split_class_names,
        class_names,
        means,
        embeddings,
        eligible_by_class,
        tau2_by_class,
        rho_by_class,
    )


def split_cohorts(
    config: dict[str, Any],
    split_idx: int,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    census: CensusContext,
) -> list[dict[str, Any]]:
    """Every random-cohort and composition-cohort record of one split."""
    ctx = _build_split_context(
        config,
        split_idx,
        full_df,
        train_df,
        census.means,
        census.class_names,
        census.rho_by_class,
        census.raw_full,
    )
    cohorts = _random_cohorts_for_split(ctx)
    comp_cohorts, r_by_allocation = _composition_cohorts_for_split(
        ctx, census.composition_allocations
    )
    _guard_composition_r(config, split_idx, r_by_allocation)
    return cohorts + comp_cohorts
