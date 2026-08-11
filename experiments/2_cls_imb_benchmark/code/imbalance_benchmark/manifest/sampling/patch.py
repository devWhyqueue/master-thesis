from __future__ import annotations

import logging
import time
from typing import cast

import numpy as np
import pandas as pd

from imbalance_benchmark.manifest import log_every

logger = logging.getLogger(__name__)


def _build_patch_hierarchy(
    df_class: pd.DataFrame, rng: np.random.Generator
) -> tuple[list[str], dict[str, dict[str, list[int]]]]:
    patients = cast(np.ndarray, df_class["case_id"].unique())
    rng.shuffle(patients)
    # Two-column `.groups` ignores `sort=False` for the second level (pandas
    # still sorts slide_id per case_id), unlike a single-column groupby.
    slide_order = df_class.groupby("case_id", sort=False)["slide_id"].unique()
    groups = df_class.groupby(["case_id", "slide_id"], sort=False).groups
    h: dict[str, dict[str, list[int]]] = {}
    for pat in patients:
        h[pat] = {}
        slides = np.asarray(slide_order[pat], dtype=object)
        rng.shuffle(slides)
        for sld in slides:
            pids = np.asarray(groups[(pat, sld)])
            rng.shuffle(pids)
            h[pat][sld] = list(pids)
    return list(patients), h


def _loop_patches(
    patients: list[str], h: dict, max_p: int, max_s: int, n: int
) -> tuple[list[int], dict, dict]:
    selected, pat_counts, sld_counts = [], {p: 0 for p in patients}, {}
    slide_cursor = {p: 0 for p in patients}
    prog = True
    last_logged = time.perf_counter()
    while len(selected) < n and prog:
        prog = False
        for p in patients:
            if len(selected) >= n or pat_counts[p] >= max_p:
                continue
            slides = list(h[p])
            for offset in range(len(slides)):
                s = slides[(slide_cursor[p] + offset) % len(slides)]
                if sld_counts.get(s, 0) >= max_s or not h[p][s]:
                    continue
                selected.append(h[p][s].pop(0))
                pat_counts[p] += 1
                sld_counts[s] = sld_counts.get(s, 0) + 1
                slide_cursor[p] = (slides.index(s) + 1) % len(slides)
                prog = True
                break
        last_logged = log_every(
            last_logged, logger, "freeze: selecting %d/%d patches", len(selected), n
        )
    return selected, pat_counts, sld_counts


def _contribution_cap(n_examples: int, fraction: float, unit: str) -> int:
    cap = int(np.floor(n_examples * fraction))
    if cap < 1:
        raise ValueError(
            f"{n_examples} examples cannot satisfy the {fraction:.0%} {unit} cap"
        )
    return cap


def _select_from_hierarchy(
    df_class: pd.DataFrame,
    patients: list[str],
    h: dict[str, dict[str, list[int]]],
    n_patches: int,
) -> pd.DataFrame:
    """Consume a hierarchy's round-robin cursors to pick ``n_patches``."""
    selected, _, _ = _loop_patches(
        patients,
        h,
        _contribution_cap(n_patches, 0.10, "patient"),
        _contribution_cap(n_patches, 0.05, "slide"),
        n_patches,
    )
    if len(selected) < n_patches:
        raise ValueError(
            "Patch allocation is infeasible under the 10% patient and 5% slide caps"
        )
    return df_class.loc[selected]


def select_patches_round_robin(
    df_class: pd.DataFrame, n_patches: int, seed: int
) -> pd.DataFrame:
    """Sample patches with round-robin patient and slide caps (10% patient, 5% slide)."""
    if df_class.empty or n_patches <= 0:
        return pd.DataFrame()
    patients, h = _build_patch_hierarchy(df_class, np.random.default_rng(seed))
    return _select_from_hierarchy(df_class, patients, h, n_patches)
