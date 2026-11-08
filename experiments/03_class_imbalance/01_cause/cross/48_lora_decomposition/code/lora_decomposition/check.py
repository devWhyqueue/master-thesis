"""Technical pilot/main check: complete compatible runs and matched audits for every block.

Pass/fail rests only on completeness, provenance compatibility, pairing, allocation/weight
identities, and finite training. Training losses and validation balanced accuracy are recorded
as diagnostics for the human review; no effect size or significance enters the decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from imbalance_benchmark.common import read_run_record

from centre.fit import decode_shard_index

from lora_decomposition import protocol
from lora_decomposition.fit import fit_dir, validate_run
from lora_decomposition.protocol import ARMS, is_smoke
from lora_decomposition.provenance import PROVENANCE_NAME
from lora_decomposition.submit import phase_indices

__all__ = ["block_problems", "check_runs"]

_CRITERIA = (
    "technical only: every arm complete and compatible, shared initialization and seed, paired "
    "row streams (B/F, C/R), allocation and weight identities, finite training"
)


def _pairing_rules(prov: dict[str, dict[str, Any]]) -> list[tuple[bool, str]]:
    audits = {arm: p["training_audit"] for arm, p in prov.items()}
    rules = []
    for left, right in (("B", "F"), ("C", "R")):
        for key in ("sample_sequence_sha256", "unique_rows_per_class"):
            rules.append(
                (
                    audits[left][key] == audits[right][key],
                    f"{key} differs between {left} and {right}",
                )
            )
        for key in ("training_identity_sha256", "data_counts"):
            rules.append(
                (
                    prov[left][key] == prov[right][key],
                    f"{key} differs between {left} and {right}",
                )
            )
    return rules


def _training_ok(p: dict[str, Any]) -> bool:
    audit, recipe = p["training_audit"], p["recipe"]
    finite = np.isfinite([audit["first_loss_mean"], audit["last_loss_mean"]]).all()
    expected_samples = int(recipe["steps"]) * int(recipe["batch_size"])
    return bool(
        audit["loss_finite"] and finite and audit["sample_count"] == expected_samples
    )


def _identity_rules(prov: dict[str, dict[str, Any]]) -> list[tuple[bool, str]]:
    audits = [p["training_audit"] for p in prov.values()]
    recipes = {json.dumps(p["recipe"], sort_keys=True) for p in prov.values()}
    return [
        (
            len({a["initialization_sha256"] for a in audits}) == 1,
            "initialization differs across arms",
        ),
        (len({p["seed"] for p in prov.values()}) == 1, "seed differs across arms"),
        (len(recipes) == 1, "recipe differs across arms"),
    ]


def block_problems(prov: dict[str, dict[str, Any]]) -> list[str]:
    """Every technical rule one block's four provenances break (empty when sound)."""
    rules = [
        *_identity_rules(prov),
        *_pairing_rules(prov),
        (
            prov["F"]["target_counts"] == prov["R"]["data_counts"],
            "F target is not R's allocation",
        ),
        (
            prov["C"]["target_counts"] == prov["B"]["data_counts"],
            "C target is not B's allocation",
        ),
        (
            all(np.allclose(prov[a]["class_weights"], 1.0) for a in ("B", "R")),
            "B/R weights not unit",
        ),
        (
            all(_training_ok(p) for p in prov.values()),
            "nonfinite loss or wrong sample count",
        ),
    ]
    return [message for ok, message in rules if not ok]


def _diagnostic(out_dir: Path, arm: str, prov: dict[str, Any]) -> dict[str, Any]:
    record = read_run_record(out_dir, splits=("validation",), array_fields=())
    endpoints = (record or {})["splits"]["validation"]["endpoints"]
    return {
        "split": prov["split"],
        "draw": prov["draw"],
        "arm": arm,
        "first_loss_mean": prov["training_audit"]["first_loss_mean"],
        "last_loss_mean": prov["training_audit"]["last_loss_mean"],
        "validation_patient_macro_ba": endpoints.get("patient_macro_balanced_accuracy"),
        "device": prov["software"]["device"],
    }


def _check_block(
    config: dict[str, Any], split_idx: int, draw_idx: int
) -> tuple[list[str], list[dict[str, Any]]]:
    """(problems, diagnostics) of one block."""
    try:
        complete = {arm: validate_run(config, split_idx, draw_idx, arm) for arm in ARMS}
    except ValueError as error:
        return [str(error)], []
    missing = [arm for arm, ok in complete.items() if not ok]
    if missing:
        return [f"missing or incomplete arms {missing}"], []
    dirs = {arm: fit_dir(config, split_idx, draw_idx, arm) for arm in ARMS}
    prov = {
        arm: json.loads((d / PROVENANCE_NAME).read_text(encoding="utf-8"))
        for arm, d in dirs.items()
    }
    return block_problems(prov), [
        _diagnostic(dirs[arm], arm, prov[arm]) for arm in ARMS
    ]


def check_runs(configs: list[dict[str, Any]], phase: str) -> Path:
    """Check every block of the phase on every dataset; write ``report/check_<phase>.json``."""
    datasets = {}
    for config in configs:
        if is_smoke(config):
            raise ValueError("smoke runs are not checked as a scientific phase")
        failures, diagnostics = [], []
        for index in phase_indices(phase):
            split_idx, draw_idx = decode_shard_index(index)
            problems, rows = _check_block(config, split_idx, draw_idx)
            failures += [f"block ({split_idx}, {draw_idx}): {p}" for p in problems]
            diagnostics += rows
        datasets[config["dataset"]["name"]] = {
            "passed": not failures,
            "failures": failures,
            "diagnostics": diagnostics,
        }
    result = {
        "phase": phase,
        "criteria": _CRITERIA,
        "passed": all(d["passed"] for d in datasets.values()),
        "datasets": datasets,
    }
    path = protocol.EXPERIMENT_DIR / "report" / f"check_{phase}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return path
