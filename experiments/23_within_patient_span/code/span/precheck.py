"""Pre-fit gate: does the within-patient complement capture pool variance better than random directions?"""

from __future__ import annotations

from typing import Any

import numpy as np

from breadth.fit import init_shard

from centre.cohort import draw_cohorts, training_table
from centre.pool import Pool, load_pool

from directions.basis import cohort_eigenbasis

from spectrum.basis import pool_variance_captured

from span import CAPTURE_MULTIPLES, baseline_config
from span.basis import complement_oracle_share, complement_within, random_complement

__all__ = ["run_precheck"]

GATE = 0.1
G = 5
SPLIT = 0


def _draw_rows(
    train_df: Any, names: list[str], pool: Pool, draw_idx: int
) -> dict[int, tuple[float, float, float]]:
    """(within, random, oracle) pool variance shares per k multiple for one draw."""
    cohorts = draw_cohorts(train_df, names, SPLIT, draw_idx)
    table = training_table(train_df, names, cohorts.nested, G)
    u_b, _ = cohort_eigenbasis(table, len(names), G)
    v, _ = complement_within(table, len(names), G, u_b)
    rand = random_complement(u_b, len(v), np.random.default_rng([SPLIT, draw_idx, G]))
    ks = {m: min(m * len(u_b), len(v)) for m in CAPTURE_MULTIPLES}
    return {
        m: (
            pool_variance_captured(v[:k], pool),
            pool_variance_captured(rand[:k], pool),
            complement_oracle_share(u_b, pool, k),
        )
        for m, k in ks.items()
    }


def run_precheck(config: dict[str, Any], n_draws: int = 10) -> dict[str, Any]:
    """Mean gate fraction (W - rand) / (oracle - rand) per k over split-0 draws at G = 5."""
    train_df, names, _, _ = init_shard(config, SPLIT)
    pool = load_pool(baseline_config(config, "baseline_outputs"), SPLIT, names)
    draws = [_draw_rows(train_df, names, pool, d) for d in range(n_draws)]
    out: dict[str, Any] = {}
    for m in CAPTURE_MULTIPLES:
        vals = [d[m] for d in draws]
        w, r, o = np.mean(vals, axis=0)
        out[f"k_{m}rB"] = {
            "within": float(w),
            "random": float(r),
            "oracle": float(o),
            "f": float(np.mean([(a - b) / (c - b) for a, b, c in vals])),
        }
    out["stop"] = all(out[f"k_{m}rB"]["f"] < GATE for m in CAPTURE_MULTIPLES)
    return out
