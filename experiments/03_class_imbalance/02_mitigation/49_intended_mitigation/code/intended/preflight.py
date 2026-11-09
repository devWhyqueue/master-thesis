"""Zero-fit gates before any fit: row identity (1), records present (2), LA equivalence (4).

Gate 1 re-derives every shard's training rows through exp-25's allocator on the damage config
and through exp-40's ``train_arm`` on the LoRA config (which exp-45's frozen CUDA runs also
used, via its ``source_config``) and requires identical rows, targets and class order. Gate 2
requires every reused record with test labels in manifest order. Gate 4 compares train-time and
post-hoc logit adjustment on one shard. Reports are JSON; a failed gate never raises here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from decodability.evidence import load_freeze_meta
from decodability.linear import fit_multinomial_logistic
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    _macro_recall_mean,
)
from imbalance_benchmark.common import (
    REPO_ROOT,
    RUN_RECORD_NAME,
    load_config,
    read_run_record,
    write_json,
)

from breadth import MAX_ITER, TOLERANCE, exp2_split_paths
from breadth.fit import init_shard

from prevalence import patients_per_class
from prevalence.fit import _arm_rows, _shard_context, arm_row_index, class_counts

from analyze.paths import run_dir, shard_keys
from mitigation.data import load_shard, train_arm

from intended import ARMS, REPORT_DIR, cuda_r100_dirs, damage_dir, lora_dirs
from intended.lam import fit_offset_logistic, logits

__all__ = ["reused_runs", "run_preflight"]

_GATE_ERRORS = (ValueError, RuntimeError, OSError, KeyError)
TIE_GAP = 1e-6
MAX_BA_DIFF_PP = 0.01


def reused_runs(config: dict[str, Any]) -> list[tuple[str, Path]]:
    """Every reused record: exp-25/26 CE, exp-41/43 LoRA CE/cRT/GCL/GCL2, exp-45 CE_SGD/CUDA r100."""
    grid = config["sources"]["mitigation_grid"]
    lora = [
        (arm, method, param)
        for arm in ARMS
        for method, params in (
            ("ce", [None]),
            ("crt", [None]),
            ("gcl", grid[arm]["stage1"]["gcl"]),
            ("gcl2", grid[arm]["stage2"]["gcl2"]),
        )
        for param in params
    ]
    frozen = [("ce", None)] + [
        ("cuda", p) for p in config["sources"]["frozen_cuda_grid"]["stage1"]["cuda"]
    ]
    runs: list[tuple[str, Path]] = []
    for s, d in shard_keys():
        runs += [("damage", damage_dir(config, s, arm, d)) for arm in ARMS]
        runs += [("lora", run_dir(lora_dirs(config)[s], *job, d)) for job in lora]
        runs += [
            ("frozen_cuda", run_dir(cuda_r100_dirs(config)[s], "r100", *job, d))
            for job in frozen
        ]
    return runs


def _train_split(config: dict[str, Any], split: int) -> tuple[pd.DataFrame, list[str]]:
    """Train rows and locked class order exactly as exp-25's ``init_shard`` reads them."""
    paths = exp2_split_paths(config, split)
    classes = list(load_freeze_meta(paths)["class_names"])
    manifest = pd.read_csv(paths["data"] / "manifest.csv")
    return manifest.query("split == 'train'").reset_index(drop=True), classes


def _exp25_rows(
    damage: dict[str, Any], split_data: tuple, where: tuple[int, int, str]
) -> tuple[list[str], np.ndarray]:
    (train_df, classes), (split, draw, arm) = split_data, where
    shard = _shard_context(train_df, classes, split, draw, patients_per_class(damage))
    counts = class_counts(arm, shard.perm, shard.available, shard.pool_counts, shard.g)
    rows, y = arm_row_index(train_df, classes, shard.patients, counts)
    return train_df.loc[rows, "image_path"].astype(str).tolist(), y


def _frozen_source_problem(config: dict[str, Any]) -> list[str]:
    source = Path(load_config(config["frozen_cuda_config"])["source_config"])
    source = source if source.is_absolute() else REPO_ROOT / source
    if source != Path(config["mitigation_config"]):
        return [
            f"exp-45 source config {source} is not the LoRA config {config['mitigation_config']}"
        ]
    return []


def _row_identity(config: dict[str, Any]) -> dict[str, Any]:
    damage, mitigation = (
        load_config(config["damage_config"]),
        load_config(config["mitigation_config"]),
    )
    problems: list[Any] = _frozen_source_problem(config)
    checked = 0
    for split in sorted({s for s, _ in shard_keys()}):
        split_data, shard = _train_split(damage, split), load_shard(mitigation, split)
        if split_data[1] != shard.classes:
            problems.append(
                {
                    "split": split,
                    "problem": "class order differs",
                    "exp25": split_data[1],
                    "exp40": shard.classes,
                }
            )
            continue
        for draw, arm in [(d, a) for s, d in shard_keys() if s == split for a in ARMS]:
            checked += 1
            ours = _exp25_rows(damage, split_data, (split, draw, arm))
            theirs = train_arm(shard, split, draw, patients_per_class(mitigation), arm)[
                :2
            ]
            if ours[0] != theirs[0] or not np.array_equal(ours[1], theirs[1]):
                problems.append(
                    {
                        "split": split,
                        "draw": draw,
                        "arm": arm,
                        "problem": "training rows differ",
                    }
                )
    return {"passed": not problems, "checked": checked, "problems": problems}


def _test_targets(config: dict[str, Any], split: int) -> np.ndarray:
    paths = exp2_split_paths(config, split)
    classes = list(load_freeze_meta(paths)["class_names"])
    test = pd.read_csv(paths["data"] / "manifest.csv").query("split == 'test'")
    return test["cancer_type"].map(classes.index).to_numpy()


def _records_present(config: dict[str, Any]) -> dict[str, Any]:
    targets = {
        s: _test_targets(config, s) for s in sorted({s for s, _ in shard_keys()})
    }
    problems: list[str] = []
    runs = reused_runs(config)
    for _, directory in runs:
        split = int(directory.parents[2].name.split("=")[1])
        record = read_run_record(
            directory, splits=("test",), array_fields=("labels", "preds")
        )
        if record is None:
            problems.append(f"missing: {directory / RUN_RECORD_NAME}")
            continue
        test = record["splits"].get("test", {})
        if "preds" not in test or not np.array_equal(
            np.asarray(test.get("labels", [])), targets[split]
        ):
            problems.append(
                f"test labels/preds absent or not in manifest order: {directory}"
            )
    return {"passed": not problems, "checked": len(runs), "problems": problems}


def _la_equivalence(config: dict[str, Any]) -> dict[str, Any]:
    train_df, names, evals, _ = init_shard(config, 0)
    shard = _shard_context(train_df, names, 0, 0, patients_per_class(config))
    counts = np.asarray(
        class_counts("r100", shard.perm, shard.available, shard.pool_counts, shard.g)
    )
    x, y = _arm_rows(train_df, names, shard.patients, list(counts))
    stored = read_run_record(
        damage_dir(config, 0, "r100", 0), splits=(), array_fields=()
    )
    if stored is None:
        return {
            "passed": False,
            "problems": ["stored exp-25/26 r100 record of shard (0, 0) missing"],
        }
    lam, log_pi = float(stored["selected_lambda"]), np.log(counts / counts.sum())
    posthoc = (
        logits(fit_multinomial_logistic(x, y, lam, TOLERANCE, MAX_ITER), evals.test_x)
        - log_pi
    )
    coef, intercept = fit_offset_logistic(x, y, lam, log_pi)
    trained = np.asarray(evals.test_x, dtype=np.float64) @ coef.T + intercept
    top2 = np.sort(posthoc, axis=1)[:, -2:]
    clear = top2[:, 1] - top2[:, 0] > TIE_GAP
    differing = int((posthoc.argmax(1) != trained.argmax(1))[clear].sum())
    cases = evals.test_id["case_id"].astype(str).to_numpy()
    ba = [
        _macro_recall_mean(evals.test_y, z.argmax(1), cases) for z in (posthoc, trained)
    ]
    diff_pp = abs(ba[0] - ba[1]) * 100
    return {
        "passed": differing == 0 and diff_pp < MAX_BA_DIFF_PP,
        "lambda": lam,
        "tau": 1.0,
        "differing_clear_rows": differing,
        "near_tie_rows": int((~clear).sum()),
        "ba_difference_pp": diff_pp,
    }


def _guarded(
    gate: Callable[[dict[str, Any]], dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    try:
        return gate(config)
    except _GATE_ERRORS as error:
        return {"passed": False, "problems": [f"{type(error).__name__}: {error}"]}


def run_preflight(config: dict[str, Any], out_dir: Path = REPORT_DIR) -> Path:
    """Gates 1, 2 and 4; writes ``preflight_<dataset>.json`` and returns its path."""
    gates = {
        "row_identity": _guarded(_row_identity, config),
        "records_present": _guarded(_records_present, config),
        "la_equivalence": _guarded(_la_equivalence, config),
    }
    path = out_dir / f"preflight_{config['dataset']['name']}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        path, {"gates": gates, "passed": all(g["passed"] for g in gates.values())}
    )
    return path
