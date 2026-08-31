"""Temperature-scaled tail-group macro-NLL deficits and method effects.

The confirmatory pipeline evaluates the calibration axis on raw probabilities
only (`gate_blocks.py` reads `record["probs"]`), so the report can state a raw
tail-group deficit but not whether it survives post-hoc scaling. This script
recomputes the same two contrasts from the same confirmed predictions with
`temperature_scaled_probs` substituted:

    deficit  D_cal = tailNLL(deprived CE) - tailNLL(balanced-reference CE)
    effect   E     = tailNLL(deprived CE) - tailNLL(method)

Both keep the pipeline's higher-is-better orientation: a positive deficit is
damage, a positive effect is a method improving on deprived CE.

Both use the deprived condition's own tail classes and the frozen crossed
patient bootstrap, and are averaged over the three locked splits exactly as
`aggregate.py` does (mean of the bootstrap-effect arrays, then percentiles).

Raw columns are recomputed alongside so the output can be checked against the
pipeline's own `cross_split_gates_and_recovery.json` (`--check`).

No permutation p-values: the crossed permutation is the expensive part and the
scaled contrast is reported as an interval-only diagnostic.

Read-only over `results/`; writes one CSV per dataset root.

    python3 scaled_tail_deficit.py --config ../configs/bracs_patch.yaml
    python3 scaled_tail_deficit.py --config ../configs/bracs_patch.yaml --check
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from imbalance_benchmark.analysis.inference.confirmatory.holm import PRIMARY_METHODS
from imbalance_benchmark.analysis.inference.context import (
    CONDITION_REFERENCE,
    BootstrapContext,
    _tail_classes,
)
from imbalance_benchmark.analysis.inference.gates import (
    calibration_gate,
    confidence_interval,
)
from imbalance_benchmark.analysis.query import load_seed_predictions
from imbalance_benchmark.common import ensure_dirs, load_config, split_paths
from imbalance_benchmark.manifest.seeds import derive_seed

logger = logging.getLogger(__name__)

METHODS = ("ce", *sorted(PRIMARY_METHODS))
PROBS = {"raw": "probs", "scaled": "temperature_scaled_probs"}


def _predictions(
    paths: dict[str, Path], condition: str, method: str, assignment: str
) -> dict[str, Any] | None:
    """Load one confirmation block, or None when this dataset never realized it.

    A frozen condition without confirmation runs (TCGA-UT has no
    `balanced_spread`) is a unit the pipeline also skips, not an error here.
    """
    try:
        return load_seed_predictions(paths, condition, method, assignment)
    except RuntimeError:
        logger.info(
            "scaled-tail: skipping absent %s/%s/%s", assignment, condition, method
        )
        return None


def _split_contrasts(
    paths: dict[str, Path],
    config: dict[str, Any],
    n_replicates: int,
    seed: int,
) -> dict[tuple[str, str, str, str, str], np.ndarray]:
    """One split's bootstrap contrast arrays, keyed by unit/method/scale/kind."""
    freeze = json.loads((paths["data"] / "manifest_freeze.json").read_text())
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    context = BootstrapContext(paths, is_mil, n_replicates, seed)
    contrasts: dict[tuple[str, str, str, str, str], np.ndarray] = {}
    for assignment, conditions in freeze.get("assignment_conditions", {}).items():
        for severity in conditions:
            reference = CONDITION_REFERENCE.get(severity)
            if reference is None or reference == severity:
                continue  # a reference condition is not its own deprived arm
            balanced = _predictions(paths, reference, "ce", assignment)
            deprived = _predictions(paths, severity, "ce", assignment)
            if balanced is None or deprived is None:
                continue
            tail = _tail_classes(
                freeze, list(balanced["class_names"]), assignment, severity
            )
            if not tail:
                continue
            methods = {
                method: _predictions(paths, severity, method, assignment)
                for method in METHODS
                if method != "ce"
            }
            for scale, key in PROBS.items():
                base = context.tail_nll_distribution(
                    balanced["labels"], balanced[key], tail
                )
                ce = context.tail_nll_distribution(
                    deprived["labels"], deprived[key], tail
                )
                assert base is not None and ce is not None  # tail is non-empty
                contrasts[(assignment, severity, "ce", scale, "deficit")] = ce - base
                for method, record in methods.items():
                    if record is None:
                        continue
                    values = context.tail_nll_distribution(
                        record["labels"], record[key], tail
                    )
                    assert values is not None
                    contrasts[(assignment, severity, method, scale, "effect")] = (
                        ce - values
                    )
            logger.info("scaled-tail: %s/%s done", assignment, severity)
    return contrasts


def _rows(
    base_paths: dict[str, Path],
    config: dict[str, Any],
    n_replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Equal-split contrasts: mean the per-split bootstrap arrays, then take percentiles."""
    per_split = [
        _split_contrasts(split_paths(base_paths, index), config, n_replicates, seed)
        for index in range(3)
    ]
    shared = set.intersection(*(set(split) for split in per_split))
    dataset = config.get("dataset", {}).get("name", "")
    rows = []
    for key in sorted(shared):
        assignment, severity, method, scale, kind = key
        pooled = np.mean(np.stack([split[key] for split in per_split]), axis=0)
        low, high = confidence_interval(pooled)
        rows.append(
            {
                "dataset": dataset,
                "assignment": assignment,
                "condition": severity,
                "method": method,
                "scale": scale,
                "kind": kind,
                "effect": float(pooled[0]),
                "ci_low": low,
                "ci_high": high,
                # Only a deficit is gated; the gate reads the (deprived - balanced)
                # magnitude against this dataset's prespecified threshold.
                "gate_passed": kind == "deficit"
                and calibration_gate(float(pooled[0]), (low, high), dataset),
            }
        )
    return rows


def _check(base_paths: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    """Assert the recomputed raw deficits reproduce the pipeline's own numbers."""
    path = base_paths["data"] / "cross_split_gates_and_recovery.json"
    published = {
        (c["assignment"], c["severity"]): c
        for c in json.loads(path.read_text())["comparisons"]
        if c.get("gate") == "calibration" and c.get("method") == "ce"
    }
    checked = 0
    for row in rows:
        if row["scale"] != "raw" or row["kind"] != "deficit":
            continue
        reference = published.get((row["assignment"], row["condition"]))
        assert reference is not None, f"no published deficit for {row}"
        assert abs(reference["effect"] - row["effect"]) < 1e-9, (
            f"raw deficit mismatch for {row['assignment']}/{row['condition']}: "
            f"published {reference['effect']} vs recomputed {row['effect']}"
        )
        checked += 1
    assert checked, "no raw deficits were compared"
    logger.info("scaled-tail: %d raw deficits match the published values", checked)


def main() -> None:
    """Recompute one dataset's raw and scaled tail-NLL contrasts and write the CSV."""
    logging.basicConfig(
        level=logging.INFO, format="|%(asctime)s| [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    base_paths = ensure_dirs(config)
    n_replicates = int(config.get("analysis", {}).get("bootstrap_replicates", 10_000))
    rows = _rows(base_paths, config, n_replicates, derive_seed(args.seed, "resampling"))
    if args.check:
        _check(base_paths, rows)
    destination = base_paths["data"] / "scaled_tail_deficit.csv"
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("scaled-tail: wrote %d rows to %s", len(rows), destination)


if __name__ == "__main__":
    main()
