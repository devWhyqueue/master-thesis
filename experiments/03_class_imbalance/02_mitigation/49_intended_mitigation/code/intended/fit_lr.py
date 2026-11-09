"""CPU logistic-regression candidates of one (split, draw): CE refit, LA, BS, DisAlign.

Stage ``ce`` refits exp-25's CE per arm over the full lambda grid; its record is written only if
it reproduces the stored exp-25/26 record (gate 3, ``intended.reproduction``), with every
lambda's fit cached for stage ``rest``. Stage ``rest`` scores logit adjustment on that path
(lambda re-selected per tau), refits class-weighted CE for BS, and trains exp-40's DisAlign on
the CE head at its selected lambda. Existing run records are skipped; the record is written last.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from decodability.linear import LinearFitResult
from imbalance_benchmark.common import RUN_RECORD_NAME, write_json

from breadth import LAMBDAS
from breadth.fit import EvalPartition, init_shard

from prevalence import patients_per_class
from prevalence.fit import _Shard, _arm_rows, _shard_context, class_counts

from analyze.paths import run_dir
from mitigation.train import EvalSplit, RunMeta, Stage1Artifacts, evaluate_and_record
from mitigation.train.stage2 import run_disalign

from lora_ablation import init_seed

from intended import ARMS, lr_paths
from intended.lam import (
    fit_lambda_path,
    load_path,
    logits,
    save_path,
    select_lambda,
    validation_scores,
)
from intended.reproduction import ReproductionError, reproduce_ce

__all__ = [
    "LAMBDA_NAME",
    "PATH_NAME",
    "ArmData",
    "balanced_weights",
    "disalign_logits",
    "fit_lr_shard",
]

LAMBDA_NAME = "lambda.json"
PATH_NAME = "ce_path.npz"


class ArmData(NamedTuple):
    """One arm's realized training rows: float64 features, targets, per-class counts."""

    name: str
    x: np.ndarray
    y: np.ndarray
    counts: np.ndarray


class _Context(NamedTuple):
    config: dict[str, Any]
    split: int
    draw: int
    names: list[str]
    evals: EvalPartition
    paths: dict[str, Path]


def balanced_weights(y: np.ndarray, counts: np.ndarray, s: float) -> np.ndarray:
    """Row weights proportional to ``n_c^(-s)``, rescaled to sum to N (exp-05's solver contract)."""
    weights = np.asarray(counts, dtype=np.float64)[y] ** -s
    return weights * len(y) / weights.sum()


def disalign_logits(
    config: dict[str, Any],
    arm: ArmData,
    evals: EvalPartition,
    fit: LinearFitResult,
    gamma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """exp-40 DisAlign trained on frozen features over one logistic-regression CE head."""
    artifacts = Stage1Artifacts(
        torch.from_numpy(arm.x).float(),
        arm.y,
        torch.from_numpy(np.asarray(evals.val_x)).float(),
        evals.val_y,
        torch.from_numpy(np.asarray(evals.test_x)).float(),
        evals.test_y,
        arm.counts,
        {
            "weight": torch.from_numpy(fit.coef).float(),
            "bias": torch.from_numpy(fit.intercept).float(),
        },
        False,
        None,
        None,
    )
    return run_disalign(
        artifacts, gamma, len(arm.counts), config, torch.device("cpu"), seed
    )


def _out(ctx: _Context, arm: str, method: str, param: float | None) -> Path:
    return run_dir(ctx.paths, arm, method, param, ctx.draw)


def _done(ctx: _Context, arm: str, method: str, param: float | None) -> bool:
    return (_out(ctx, arm, method, param) / RUN_RECORD_NAME).is_file()


def _selection(
    ctx: _Context, fits: list[LinearFitResult], offset: np.ndarray | None = None
) -> dict[str, Any]:
    evals = ctx.evals
    cases = evals.val_id["case_id"].astype(str).to_numpy()
    index = select_lambda(fits, evals.val_x, evals.val_y, cases, offset)
    return {
        "selected_index": index,
        "selected_lambda": LAMBDAS[index],
        "val_scores": validation_scores(fits, evals.val_x, evals.val_y, cases, offset),
        "converged": [bool(fit.converged) for fit in fits],
    }


def _record(
    ctx: _Context,
    arm: ArmData,
    key: tuple[str, float | None],
    pair: tuple,
    selection: dict,
) -> None:
    out_dir = _out(ctx, arm.name, *key)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / LAMBDA_NAME, selection)
    counts = dict(zip(ctx.names, (int(c) for c in arm.counts)))
    evaluate_and_record(
        out_dir,
        RunMeta(ctx.config, key[0], key[1], arm.name, counts),
        EvalSplit(pair[0], ctx.evals.val_y, ctx.evals.val_id),
        EvalSplit(pair[1], ctx.evals.test_y, ctx.evals.test_id),
    )


def _fit_ce(ctx: _Context, arm: ArmData) -> None:
    if _done(ctx, arm.name, "ce", None):
        return
    fits = fit_lambda_path(arm.x, arm.y)
    selection = _selection(ctx, fits)
    best = fits[selection["selected_index"]]
    test_logits = logits(best, ctx.evals.test_x)
    out_dir = _out(ctx, arm.name, "ce", None)
    cases = ctx.evals.test_id["case_id"].astype(str).to_numpy()
    refit = (selection["selected_lambda"], test_logits.argmax(1), cases)
    where = (ctx.split, ctx.draw, arm.name)
    selection["reproduction"] = reproduce_ce(ctx.config, where, refit, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path(out_dir / PATH_NAME, fits)
    _record(
        ctx, arm, ("ce", None), (logits(best, ctx.evals.val_x), test_logits), selection
    )


def _fit_la(
    ctx: _Context, arm: ArmData, fits: list[LinearFitResult], tau: float
) -> None:
    offset = -tau * np.log(arm.counts / arm.counts.sum())
    selection = _selection(ctx, fits, offset)
    best = fits[selection["selected_index"]]
    pair = (
        logits(best, ctx.evals.val_x) + offset,
        logits(best, ctx.evals.test_x) + offset,
    )
    _record(ctx, arm, ("la", tau), pair, selection)


def _fit_bs(ctx: _Context, arm: ArmData, s: float) -> None:
    fits = fit_lambda_path(arm.x, arm.y, balanced_weights(arm.y, arm.counts, s))
    selection = _selection(ctx, fits)
    best = fits[selection["selected_index"]]
    _record(
        ctx,
        arm,
        ("bs", s),
        (logits(best, ctx.evals.val_x), logits(best, ctx.evals.test_x)),
        selection,
    )


def _fit_disalign(
    ctx: _Context, arm: ArmData, ce: tuple[LinearFitResult, dict], gamma: float
) -> None:
    fit, selection = ce
    seed = init_seed(ctx.split, ctx.draw)
    pair = disalign_logits(ctx.config, arm, ctx.evals, fit, gamma, seed)
    _record(ctx, arm, ("disalign", gamma), pair, {**selection, "seed": seed})


def _fit_rest(ctx: _Context, arm: ArmData) -> None:
    ce_dir = _out(ctx, arm.name, "ce", None)
    if not _done(ctx, arm.name, "ce", None):
        raise RuntimeError(
            f"CE record missing at {ce_dir}; run the fit --stage ce first"
        )
    fits = load_path(ce_dir / PATH_NAME)
    ce_selection = json.loads((ce_dir / LAMBDA_NAME).read_text(encoding="utf-8"))
    fitters = {
        "la": lambda tau: _fit_la(ctx, arm, fits, tau),
        "bs": lambda s: _fit_bs(ctx, arm, s),
        "disalign": lambda g: _fit_disalign(
            ctx, arm, (fits[ce_selection["selected_index"]], ce_selection), g
        ),
    }
    for method, params in ctx.config["mitigation"]["grid"][arm.name]["lr"].items():
        for param in params if method in fitters else []:
            if not _done(ctx, arm.name, method, param):
                fitters[method](param)


def _arm(shard: _Shard, arm: str) -> ArmData:
    counts = class_counts(arm, shard.perm, shard.available, shard.pool_counts, shard.g)
    x, y = _arm_rows(shard.train_df, shard.names, shard.patients, counts)
    return ArmData(arm, x, y, np.asarray(counts, dtype=np.int64))


def fit_lr_shard(config: dict[str, Any], split: int, draw: int, stage: str) -> None:
    """Fit every pending ``ce`` (gate 3) or ``rest`` (LA, BS, DisAlign) candidate of one shard."""
    if stage not in ("ce", "rest"):
        raise ValueError(f"stage must be 'ce' or 'rest', not {stage!r}")
    train_df, names, evals, _ = init_shard(config, split)
    shard = _shard_context(train_df, names, split, draw, patients_per_class(config))
    ctx = _Context(config, split, draw, names, evals, lr_paths(config)[split])
    failures: list[str] = []
    for arm in ARMS:
        try:
            (_fit_ce if stage == "ce" else _fit_rest)(ctx, _arm(shard, arm))
        except ReproductionError as error:
            failures.append(str(error))
    if failures:
        raise ReproductionError("; ".join(failures))
