"""Gate 3: the CE refit must match the stored exp-25/26 record (selected lambda, test predictions)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import read_run_record, write_json

__all__ = [
    "MISMATCH_NAME",
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
