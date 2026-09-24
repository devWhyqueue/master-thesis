"""Unit and small integration tests for exp-39's phase-04 fit stage.

Uses tiny synthetic data throughout (a fake feature cache, no GPU/UNI2-h/real
datasets); mirrors ``test_schedule.py``'s synthetic manifest and
``test_features.py``'s fake-cache pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from prevalence import BALANCED, DEPTH
from prevalence.fit import _shard_context

from transfer import ARMS, LAMBDAS
from transfer.fit import EvalPartition, decode_shard_index, run_fit_shard, shard_count
from transfer.fit.tuning import _select_best_lambda, fit_arm

_NAMES = ["clsA", "clsB", "clsC"]
_G = 3


def _synthetic_train_df() -> pd.DataFrame:
    """More than G eligible patients per class, each with exactly DEPTH patches."""
    rows = []
    for name in _NAMES:
        for p in range(6):
            case = f"{name}-P{p}"
            for patch in range(DEPTH):
                rows.append(
                    {
                        "cancer_type": name,
                        "case_id": case,
                        "slide_id": f"{case}-S0",
                        "patch_id": f"{case}-p{patch:04d}",
                    }
                )
    return pd.DataFrame(rows)


def _attach_features(df: pd.DataFrame, cache_dir: Path, dim: int) -> pd.DataFrame:
    """Fake feature cache: one random tensor per row, one file per case_id."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    rng = np.random.default_rng(0)
    paths, indices = [], []
    for case_id, group in df.groupby("case_id", sort=False):
        path = cache_dir / f"{case_id}.pt"
        torch.save(torch.from_numpy(rng.normal(size=(len(group), dim)).astype("float32")), path)
        paths.extend([str(path)] * len(group))
        indices.extend(range(len(group)))
    df["feature_path"] = paths
    df["feature_index"] = indices
    return df


def _eval_partition(
    dim: int, n_classes: int, per_class: int = 2
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(1)
    rows = []
    for ci in range(n_classes):
        for p in range(per_class):
            rows.append((ci, f"cls{ci}-eval-P{p}", f"cls{ci}-eval-P{p}-S0"))
    y = np.array([r[0] for r in rows], dtype=np.int64)
    x = rng.normal(size=(len(rows), dim)).astype(np.float64)
    ident = pd.DataFrame({"case_id": [r[1] for r in rows], "slide_id": [r[2] for r in rows]})
    return x, y, ident


def _evals(dim: int, n_classes: int) -> EvalPartition:
    val = _eval_partition(dim, n_classes)
    test = _eval_partition(dim, n_classes)
    return EvalPartition(*val, *test)


def _shard(dim: int, cache_dir: Path):
    df = _attach_features(_synthetic_train_df(), cache_dir, dim)
    return _shard_context(df, _NAMES, split_idx=0, draw_idx=10, g=_G)


_CONFIG: dict[str, Any] = {
    "dataset": {},
    "feature_extraction": {},
    "prevalence": {"patients_per_class": _G},
}


@pytest.mark.parametrize("dim", [5, 9])
def test_fit_arm_dimension_agnostic_and_stores_every_candidate(tmp_path: Path, dim: int) -> None:
    shard = _shard(dim, tmp_path / "cache")
    evals = _evals(dim, len(_NAMES))
    out_dir = tmp_path / "out"
    fit_arm(_CONFIG, out_dir, "B", shard, evals, draw_idx=10, fit_source=("r1", None))

    from imbalance_benchmark.common import read_run_record

    rec = read_run_record(out_dir)
    assert rec is not None
    assert len(rec["candidates"]) == len(LAMBDAS)
    assert {c["lambda"] for c in rec["candidates"]} == set(LAMBDAS)
    assert rec["arm"] == "B"
    assert sum(rec["class_counts"].values()) == _G * BALANCED * len(_NAMES)

    with np.load(out_dir / "candidates.npz") as data:
        assert data["coef"].shape == (len(LAMBDAS), len(_NAMES), dim)
        assert data["intercept"].shape == (len(LAMBDAS), len(_NAMES))

    temperature = (out_dir / "temperature.json").exists()
    assert temperature


def test_p_arm_stores_prior_counts_and_positive_weights(tmp_path: Path) -> None:
    dim = 4
    shard = _shard(dim, tmp_path / "cache")
    evals = _evals(dim, len(_NAMES))
    out_dir = tmp_path / "out"
    fit_arm(_CONFIG, out_dir, "P100", shard, evals, draw_idx=10, fit_source=("r1", "r100"))

    from imbalance_benchmark.common import read_run_record

    rec = read_run_record(out_dir)
    assert rec is not None
    assert "prior_counts" in rec
    # P100's own rows are r1 (balanced): same total as B's rows.
    assert sum(rec["class_counts"].values()) == _G * BALANCED * len(_NAMES)
    assert sum(rec["prior_counts"].values()) == _G * BALANCED * len(_NAMES)


def test_select_best_lambda_raises_when_nothing_converges(monkeypatch) -> None:
    import transfer.fit.tuning as tuning_mod

    class _Never:
        converged = False
        coef = np.zeros((2, 3))
        intercept = np.zeros(2)

    monkeypatch.setattr(tuning_mod, "fit_multinomial_logistic", lambda *a, **k: _Never())
    x = np.zeros((4, 3))
    y = np.array([0, 1, 0, 1])
    ident = pd.DataFrame({"case_id": ["a", "b", "c", "d"], "slide_id": ["a", "b", "c", "d"]})
    with pytest.raises(RuntimeError):
        _select_best_lambda(x, y, x, y, ident)


def test_ties_break_toward_the_larger_lambda(monkeypatch) -> None:
    import transfer.fit.tuning as tuning_mod
    from decodability.linear import LinearFitResult

    def _flat_fit(features, labels, lambda_val, tol, max_iter, sample_weight=None):
        return LinearFitResult(
            coef=np.zeros((2, 3)),
            intercept=np.zeros(2),
            lambda_val=lambda_val,
            c_val=1.0,
            solver="lbfgs",
            precision="float64",
            tolerance=tol,
            solver_tolerance=tol,
            max_iter=max_iter,
            n_iter=1,
            objective=0.0,
            converged=True,
            weighted=sample_weight is not None,
        )

    monkeypatch.setattr(tuning_mod, "fit_multinomial_logistic", _flat_fit)
    x = np.zeros((4, 3))
    y = np.array([0, 1, 0, 1])
    ident = pd.DataFrame({"case_id": ["a", "b", "c", "d"], "slide_id": ["a", "b", "c", "d"]})
    _, best_lam, _, candidates = _select_best_lambda(x, y, x, y, ident)
    assert best_lam == max(LAMBDAS)
    assert len(candidates) == len(LAMBDAS)


def test_run_fit_shard_skips_already_fit_arms(tmp_path: Path, monkeypatch) -> None:
    dim = 4
    cache = tmp_path / "cache"
    df = _attach_features(_synthetic_train_df(), cache, dim)
    evals = _evals(dim, len(_NAMES))

    import transfer.fit as fit_mod
    from imbalance_benchmark.common import ensure_dirs, split_paths

    config = {**_CONFIG, "paths": {"outputs": str(tmp_path / "outputs")}}
    monkeypatch.setattr(
        fit_mod,
        "init_shard",
        lambda cfg, split_idx, encoder: (
            df,
            _NAMES,
            evals,
            split_paths(ensure_dirs(cfg), split_idx),
        ),
    )
    calls: list[str] = []
    real_fit_arm = fit_mod.fit_arm

    def _tracking_fit_arm(config, out_dir, arm, *rest, **kw):
        calls.append(arm)
        return real_fit_arm(config, out_dir, arm, *rest, **kw)

    monkeypatch.setattr(fit_mod, "fit_arm", _tracking_fit_arm)
    shard_index = next(
        i for i in range(shard_count()) if decode_shard_index(i)[:2] == ("uni2h", 0)
    )
    run_fit_shard(config, shard_index)
    assert set(calls) == set(ARMS)

    calls.clear()
    run_fit_shard(config, shard_index)
    assert calls == []


def test_decode_shard_index_covers_encoder_split_draw() -> None:
    from transfer import ENCODERS, MAIN_DRAWS
    from imbalance_benchmark.common import N_PATIENT_SPLITS

    assert shard_count() == len(ENCODERS) * N_PATIENT_SPLITS * len(MAIN_DRAWS)
    seen = {decode_shard_index(i) for i in range(shard_count())}
    assert len(seen) == shard_count()
    for encoder, split_idx, draw_idx in seen:
        assert encoder in ENCODERS
        assert 0 <= split_idx < N_PATIENT_SPLITS
        assert draw_idx in MAIN_DRAWS
    with pytest.raises(ValueError):
        decode_shard_index(-1)
    with pytest.raises(ValueError):
        decode_shard_index(shard_count())
