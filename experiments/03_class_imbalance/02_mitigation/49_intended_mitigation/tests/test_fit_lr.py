"""CPU logistic-regression candidates: CE reproduction (gate 3), LA, BS, DisAlign, resume."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from imbalance_benchmark.common import EVAL_ARRAYS_NAME, RUN_RECORD_NAME, read_run_record

from breadth import LAMBDAS
from breadth.fit import EvalPartition

from analyze import select as select_module
from analyze.paths import run_dir

from intended import damage_dir, lr_paths
from intended import fit_lr
from intended.fit_lr import (
    LAMBDA_NAME,
    PATH_NAME,
    ArmData,
    balanced_weights,
    disalign_logits,
    fit_lr_shard,
)
from intended.lam import fit_lambda_path, load_path, logits
from intended.reproduction import MISMATCH_NAME, ReproductionError

from intended_world import build_world, patch_freeze_loader, write_damage_records


@pytest.fixture
def world(tmp_path, monkeypatch) -> dict:
    patch_freeze_loader(monkeypatch)
    config = build_world(tmp_path)
    write_damage_records(config, [(0, 0)])
    return config


def _dir(config: dict, arm: str, method: str, param: float | None):
    return run_dir(lr_paths(config)[0], arm, method, param, 0)


def _test(directory, field: str = "preds") -> np.ndarray:
    record = read_run_record(directory, splits=("test",), array_fields=(field,))
    assert record is not None
    return np.asarray(record["splits"]["test"][field])


def test_ce_reproduces_stored_record(world: dict) -> None:
    fit_lr_shard(world, 0, 0, "ce")
    for arm in ("r1", "r100"):
        stored = read_run_record(damage_dir(world, 0, arm, 0), splits=("test",), array_fields=("preds",))
        own = _dir(world, arm, "ce", None)
        np.testing.assert_array_equal(_test(own), stored["splits"]["test"]["preds"])
        selection = json.loads((own / LAMBDA_NAME).read_text())
        assert selection["selected_lambda"] == stored["selected_lambda"]
        assert len(selection["val_scores"]) == len(LAMBDAS)
        assert len(load_path(own / PATH_NAME)) == len(LAMBDAS)


def test_ce_mismatch_raises(world: dict) -> None:
    stored_dir = damage_dir(world, 0, "r100", 0)
    with np.load(stored_dir / EVAL_ARRAYS_NAME) as arrays:
        corrupted = {k: arrays[k].copy() for k in arrays.files}
    corrupted["test_preds"][0] = 1 - corrupted["test_preds"][0]
    np.savez_compressed(stored_dir / EVAL_ARRAYS_NAME, **corrupted)
    with pytest.raises(ReproductionError, match=r"split 0 draw 0 arm r100.*1 test rows differ"):
        fit_lr_shard(world, 0, 0, "ce")
    own = _dir(world, "r100", "ce", None)
    assert (own / MISMATCH_NAME).is_file() and not (own / RUN_RECORD_NAME).exists()


def test_ce_missing_stored_record_raises(world: dict) -> None:
    shutil.rmtree(damage_dir(world, 0, "r1", 0))
    with pytest.raises(ReproductionError, match="missing"):
        fit_lr_shard(world, 0, 0, "ce")


def test_rest_requires_ce(world: dict) -> None:
    with pytest.raises(RuntimeError, match="stage ce"):
        fit_lr_shard(world, 0, 0, "rest")


@pytest.fixture
def fitted(world: dict) -> dict:
    fit_lr_shard(world, 0, 0, "ce")
    fit_lr_shard(world, 0, 0, "rest")
    return world


def test_la_is_shifted_ce(fitted: dict) -> None:
    fits = load_path(_dir(fitted, "r100", "ce", None) / PATH_NAME)
    counts = np.asarray(list(read_run_record(_dir(fitted, "r100", "ce", None))["class_counts"].values()))
    test_x = _test_features(fitted)
    for tau in fitted["mitigation"]["grid"]["r100"]["lr"]["la"]:
        la_dir = _dir(fitted, "r100", "la", tau)
        index = json.loads((la_dir / LAMBDA_NAME).read_text())["selected_index"]
        expected = logits(fits[index], test_x) - tau * np.log(counts / counts.sum())
        np.testing.assert_allclose(_test(la_dir, "logits"), expected, atol=1e-9)


def _test_features(config: dict) -> np.ndarray:
    from breadth.fit import init_shard

    return init_shard(config, 0)[2].test_x


def test_bs_weights() -> None:
    y = np.array([0, 0, 0, 0, 1, 2, 2])
    counts = np.bincount(y)
    for s in (0.5, 1.0):
        weights = balanced_weights(y, counts, s)
        assert weights.sum() == pytest.approx(len(y))
        assert weights[0] / weights[4] == pytest.approx((4 / 1) ** -s)
        assert weights[5] / weights[4] == pytest.approx((2 / 1) ** -s)
    np.testing.assert_array_equal(balanced_weights(y, counts, 0.0), np.ones(len(y)))


def test_every_candidate_recorded(fitted: dict) -> None:
    for arm, stages in fitted["mitigation"]["grid"].items():
        for method, params in stages["lr"].items():
            for param in params:
                directory = _dir(fitted, arm, method, param)
                assert (directory / RUN_RECORD_NAME).is_file() and (directory / LAMBDA_NAME).is_file()


def test_disalign_large_logits_finite(fitted: dict) -> None:
    fits = fit_lambda_path(np.eye(4)[[0, 1, 2, 3] * 5], np.array([0, 1, 2] * 6 + [0, 1]))
    huge = fits[0].__class__(**{**fits[0].__dict__, "coef": fits[0].coef * 1e3, "intercept": fits[0].intercept * 1e3})
    rng = np.random.default_rng(0)
    arm = ArmData("r100", rng.normal(size=(40, 4)) * 50, np.array([0] * 34 + [1] * 4 + [2] * 2), np.array([34, 4, 2]))
    evals = EvalPartition(rng.normal(size=(9, 4)) * 50, np.array([0, 1, 2] * 3), None, rng.normal(size=(6, 4)) * 50, np.array([0, 1, 2] * 2), None)
    val, test = disalign_logits(fitted, arm, evals, huge, 2.0, 7)
    assert np.isfinite(val).all() and np.isfinite(test).all()
    assert np.abs(val).max() > 1e3  # the large-logit regime was actually exercised


def test_resume_skips_complete(fitted: dict, monkeypatch) -> None:
    before = _test(_dir(fitted, "r100", "bs", 0.5), "logits")
    monkeypatch.setattr(fit_lr, "fit_lambda_path", lambda *a, **k: pytest.fail("refit a complete record"))
    monkeypatch.setattr(fit_lr, "run_disalign", lambda *a, **k: pytest.fail("refit a complete record"))
    fit_lr_shard(fitted, 0, 0, "ce")
    fit_lr_shard(fitted, 0, 0, "rest")
    np.testing.assert_array_equal(_test(_dir(fitted, "r100", "bs", 0.5), "logits"), before)


def test_records_selectable(fitted: dict, monkeypatch) -> None:
    monkeypatch.setattr(select_module, "shard_keys", lambda: [(0, 0)])
    for arm, stages in fitted["mitigation"]["grid"].items():
        for method, params in stages["lr"].items():
            for param in params:
                (_dir(fitted, arm, method, param) / EVAL_ARRAYS_NAME).unlink()  # test arrays gone
            chosen = select_module.select(fitted, lr_paths(fitted), arm, "lr", method)
            assert chosen[(0, 0)] in params
