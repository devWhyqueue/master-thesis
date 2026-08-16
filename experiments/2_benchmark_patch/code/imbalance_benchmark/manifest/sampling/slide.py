from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from imbalance_benchmark.manifest.sampling.patch import _contribution_cap


def _loop_slides(patients: list[str], h: dict, max_p: int, n: int) -> list[int]:
    selected, pat_counts = [], {p: 0 for p in patients}
    prog = True
    while len(selected) < n and prog:
        prog = False
        for p in patients:
            if len(selected) >= n or pat_counts[p] >= max_p:
                continue
            if h[p]:
                selected.append(h[p].pop(0))
                pat_counts[p] += 1
                prog = True
    return selected


def _build_slide_hierarchy(
    df_slides: pd.DataFrame, rng: np.random.Generator
) -> tuple[list[str], dict[str, list[int]]]:
    """Build a randomized per-patient slide-index dictionary."""
    patients = list(cast(np.ndarray, df_slides["case_id"].unique()))
    rng.shuffle(patients)
    h = {
        p: list(df_slides[df_slides["case_id"] == p].index.to_numpy()) for p in patients
    }
    for p in h:
        rng.shuffle(h[p])
    return patients, h


def max_feasible_slide_level(df_class: pd.DataFrame) -> int:
    """Largest slide count `select_slides_round_robin` can satisfy for one class.

    Mirrors that function's 10% patient-contribution cap so callers can size
    candidate levels that are always reachable, instead of discovering
    infeasibility only after a training run at a proposed level.
    """
    counts = df_class.drop_duplicates("slide_id").groupby("case_id").size().to_numpy()
    for n in range(int(counts.sum()), 0, -1):
        try:
            cap = _contribution_cap(n, 0.10, "patient")
        except ValueError:
            continue
        if int(np.minimum(counts, cap).sum()) >= n:
            return n
    return 0


def select_slides_round_robin(
    df_class: pd.DataFrame, n_slides: int, seed: int
) -> pd.DataFrame:
    """Sample MIL slides under the 10% patient contribution cap."""
    if df_class.empty or n_slides <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    df_slides = df_class.drop_duplicates("slide_id")
    patients, h = _build_slide_hierarchy(df_slides, rng)
    patient_cap = _contribution_cap(n_slides, 0.10, "patient")
    selected = _loop_slides(patients, h, patient_cap, n_slides)
    if len(selected) < n_slides:
        raise ValueError("Slide allocation is infeasible under the 10% patient cap")
    return cast(
        pd.DataFrame,
        df_class[df_class["slide_id"].isin(df_slides.loc[selected, "slide_id"])],
    )
