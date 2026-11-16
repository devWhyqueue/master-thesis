"""Gate 3: the CE refit must match the stored exp-25/26 record (selected lambda, test predictions)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import RUN_RECORD_NAME, read_run_record, write_json

from analyze.paths import run_dir, shard_keys

from intended import ARMS, REPORT_DIR, damage_dir, lr_paths

__all__ = [
    "MISMATCH_NAME",
    "check_ce",
    "ReproductionError",
    "check_reproduction",
    "reproduction_problem",
]

MISMATCH_NAME = "mismatch.json"


class ReproductionError(RuntimeError):
    """The CE refit differs from the stored exp-25/26 record (gate 3)."""


def reproduction_problem(
    stored_dir: Path, lam: float, preds: np.ndarray
) -> dict[str, Any] | None:
    """``None`` if the stored record has this lambda and these test predictions, else the reason."""
    stored = read_run_record(stored_dir, splits=("test",), array_fields=("preds",))
    if stored is None:
        return {"missing": str(stored_dir)}
    theirs = np.asarray(stored["splits"]["test"]["preds"])
    same_shape = theirs.shape == preds.shape
    differing = int((theirs != preds).sum()) if same_shape else len(preds)
    if float(stored["selected_lambda"]) == lam and differing == 0:
        return None
    return {
        "lambda": lam,
        "stored_lambda": stored["selected_lambda"],
        "differing_rows": differing,
    }


def check_reproduction(
    stored_dir: Path, where: str, refit: tuple[float, np.ndarray], out_dir: Path
) -> None:
    """Write ``mismatch.json`` next to the would-be CE record and raise if the refit differs."""
    problem = reproduction_problem(stored_dir, *refit)
    if problem is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / MISMATCH_NAME, {"where": where, **problem})
    if "missing" in problem:
        raise ReproductionError(
            f"{where}: stored exp-25/26 record missing at {stored_dir}"
        )
    raise ReproductionError(
        f"{where}: lambda {problem['lambda']} vs stored {problem['stored_lambda']}, "
        f"{problem['differing_rows']} test rows differ"
    )


def _arm_shard(
    config: dict[str, Any], shard: tuple[int, int], arm: str
) -> tuple[str, Any]:
    """``("matched" | "mismatch" | "missing", detail)`` of one arm-shard's CE refit."""
    s, d = shard
    own = run_dir(lr_paths(config)[s], arm, "ce", None, d)
    if (own / MISMATCH_NAME).is_file() and not (own / RUN_RECORD_NAME).is_file():
        return "mismatch", json.loads((own / MISMATCH_NAME).read_text(encoding="utf-8"))
    if not (own / RUN_RECORD_NAME).is_file():
        return "missing", str(own)
    selection = json.loads((own / "lambda.json").read_text(encoding="utf-8"))
    record = read_run_record(own, splits=("test",), array_fields=("preds",)) or {}
    preds = np.asarray(record["splits"]["test"]["preds"])
    problem = reproduction_problem(
        damage_dir(config, s, arm, d), float(selection["selected_lambda"]), preds
    )
    if problem is None:
        return "matched", None
    return "mismatch", {"where": f"split {s} draw {d} arm {arm}", **problem}


def check_ce(config: dict[str, Any], out_dir: Path = REPORT_DIR) -> Path:
    """Summarize gate 3 over every arm-shard into ``ce_reproduction_<dataset>.json``."""
    found: dict[str, list[Any]] = {"matched": [], "mismatch": [], "missing": []}
    for shard in shard_keys():
        for arm in ARMS:
            status, detail = _arm_shard(config, shard, arm)
            found[status].append(detail)
    expected = len(shard_keys()) * len(ARMS)
    payload = {
        "matched": len(found["matched"]),
        "expected": expected,
        "mismatches": found["mismatch"],
        "missing": found["missing"],
        "passed": len(found["matched"]) == expected,
    }
    path = out_dir / f"ce_reproduction_{config['dataset']['name']}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return path
