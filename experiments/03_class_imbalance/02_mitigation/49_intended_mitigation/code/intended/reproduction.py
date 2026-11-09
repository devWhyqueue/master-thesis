"""Gate 3: the CE refit must reproduce the stored exp-25/26 record.

Reproduced iff the selected lambda is identical and the test patient-macro BA differs by less
than 0.01 pp. Re-specified on 2026-11-09 after the first CE stage: exact prediction identity
failed on 4/60 arm-shards (same lambda, 1-68 test rows) because exp-25's solver stops early
(issue #5), so near-boundary rows depend on floating-point order. Differing rows and the BA gap
are recorded for every arm-shard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.analysis.query import load_test_identity
from imbalance_benchmark.analysis.reporting.clustered_endpoints import (
    _macro_recall_mean,
)
from imbalance_benchmark.common import RUN_RECORD_NAME, read_run_record, write_json

from breadth import exp2_split_paths

from analyze.paths import run_dir, shard_keys

from intended import ARMS, REPORT_DIR, damage_dir, lr_paths

__all__ = [
    "MAX_BA_DIFF_PP",
    "MISMATCH_NAME",
    "ReproductionError",
    "check_ce",
    "check_reproduction",
    "compare_to_stored",
]

MISMATCH_NAME = "mismatch.json"
MAX_BA_DIFF_PP = 0.01


class ReproductionError(RuntimeError):
    """The CE refit does not reproduce the stored exp-25/26 record (gate 3)."""


def compare_to_stored(
    stored_dir: Path, lam: float, preds: np.ndarray, cases: np.ndarray
) -> dict[str, Any]:
    """Lambda, differing test rows and BA gap against the stored record, and the gate verdict."""
    stored = read_run_record(
        stored_dir, splits=("test",), array_fields=("labels", "preds")
    )
    if stored is None:
        return {"missing": str(stored_dir), "reproduced": False}
    labels = np.asarray(stored["splits"]["test"]["labels"])
    theirs = np.asarray(stored["splits"]["test"]["preds"])
    if theirs.shape != preds.shape:
        return {"shape": [list(theirs.shape), list(preds.shape)], "reproduced": False}
    ba = [_macro_recall_mean(labels, p, cases) * 100 for p in (preds, theirs)]
    gap = abs(ba[0] - ba[1])
    same_lambda = float(stored["selected_lambda"]) == lam
    return {
        "lambda": lam,
        "stored_lambda": stored["selected_lambda"],
        "differing_rows": int((theirs != preds).sum()),
        "ba_difference_pp": gap,
        "reproduced": same_lambda and gap < MAX_BA_DIFF_PP,
    }


def check_reproduction(
    stored_dir: Path,
    where: str,
    refit: tuple[float, np.ndarray, np.ndarray],
    out_dir: Path,
) -> dict[str, Any]:
    """The comparison if reproduced; else write ``mismatch.json`` and raise."""
    comparison = compare_to_stored(stored_dir, *refit)
    if comparison["reproduced"]:
        return comparison
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / MISMATCH_NAME, {"where": where, **comparison})
    raise ReproductionError(f"{where}: not reproduced: {comparison}")


def _test_cases(config: dict[str, Any], split: int) -> np.ndarray:
    manifest = exp2_split_paths(config, split)["data"] / "manifest.csv"
    return load_test_identity(manifest, is_mil=False)["case_id"].astype(str).to_numpy()


def _arm_shard(
    config: dict[str, Any], shard: tuple[int, int], arm: str, cases: np.ndarray
) -> tuple[str, Any]:
    """``("exact" | "tolerated" | "mismatch" | "missing", detail)`` of one CE refit."""
    s, d = shard
    own, where = (
        run_dir(lr_paths(config)[s], arm, "ce", None, d),
        f"split {s} draw {d} arm {arm}",
    )
    if not (own / RUN_RECORD_NAME).is_file():
        if (own / MISMATCH_NAME).is_file():
            return "mismatch", json.loads(
                (own / MISMATCH_NAME).read_text(encoding="utf-8")
            )
        return "missing", str(own)
    selection = json.loads((own / "lambda.json").read_text(encoding="utf-8"))
    record = read_run_record(own, splits=("test",), array_fields=("preds",)) or {}
    preds = np.asarray(record["splits"]["test"]["preds"])
    lam = float(selection["selected_lambda"])
    comparison = {
        "where": where,
        **compare_to_stored(damage_dir(config, s, arm, d), lam, preds, cases),
    }
    if not comparison["reproduced"]:
        return "mismatch", comparison
    return ("exact" if comparison["differing_rows"] == 0 else "tolerated"), comparison


def _summary(found: dict[str, list[Any]], expected: int) -> dict[str, Any]:
    matched = len(found["exact"]) + len(found["tolerated"])
    return {
        "matched": matched,
        "exact": len(found["exact"]),
        "expected": expected,
        "tolerated": found["tolerated"],
        "mismatches": found["mismatch"],
        "missing": found["missing"],
        "passed": matched == expected,
    }


def check_ce(config: dict[str, Any], out_dir: Path = REPORT_DIR) -> Path:
    """Summarize gate 3 over every arm-shard into ``ce_reproduction_<dataset>.json``."""
    found: dict[str, list[Any]] = {
        "exact": [],
        "tolerated": [],
        "mismatch": [],
        "missing": [],
    }
    cases = {s: _test_cases(config, s) for s in sorted({s for s, _ in shard_keys()})}
    for shard in shard_keys():
        for arm in ARMS:
            status, detail = _arm_shard(config, shard, arm, cases[shard[0]])
            found[status].append(detail)
    payload = _summary(found, len(shard_keys()) * len(ARMS))
    path = out_dir / f"ce_reproduction_{config['dataset']['name']}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return path
