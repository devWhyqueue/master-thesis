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

Every roster method is evaluated; each effect is also emitted as a recovery
ratio R = effect / deficit per replicate of the split-pooled arrays.

`--classes all` swaps the tail group for every class, giving the case-macro
NLL over all classes (no gate: the calibration thresholds are tail-specific).

Raw columns are recomputed alongside so the output can be checked against the
pipeline's own `cross_split_gates_and_recovery.json` (`--check`, tail only).

No permutation p-values: the crossed permutation is the expensive part and the
scaled contrast is reported as an interval-only diagnostic.

Read-only over `results/`; writes one CSV per dataset root.

    python3 scaled_tail_deficit.py --config ../configs/bracs_patch.yaml
    python3 scaled_tail_deficit.py --config ../configs/bracs_patch.yaml --check
    python3 scaled_tail_deficit.py --config ../configs/bracs_patch.yaml --classes all
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

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
from imbalance_benchmark.modeling.context import roster_for_regime

logger = logging.getLogger(__name__)

PROBS = {"raw": "probs", "scaled": "temperature_scaled_probs"}
OUTPUTS = {"tail": "scaled_tail_deficit.csv", "all": "nll_recovery_all_classes.csv"}


def _predictions(
    paths: dict[str, Path], condition: str, method: str, assignment: str
) -> dict[str, Any] | None:
    """Load one confirmation block, or None for a unit the pipeline also skips.

    Example: TCGA-UT never realized `balanced_spread`."""
    try:
        return load_seed_predictions(
            paths, condition, method, assignment, fields=tuple(PROBS.values())
        )
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
    classes: str,
) -> dict[tuple[str, str, str, str, str], np.ndarray]:
    """One split's bootstrap contrast arrays, keyed by unit/method/scale/kind."""
    freeze = json.loads((paths["data"] / "manifest_freeze.json").read_text())
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    context = BootstrapContext(paths, is_mil, n_replicates, seed)
    methods_roster = [m for m in roster_for_regime(is_mil) if m != "ce"]
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
            class_names = list(balanced["class_names"])
            tail = (
                list(range(len(class_names)))
                if classes == "all"
                else _tail_classes(freeze, class_names, assignment, severity)
            )
            if not tail:
                continue
            methods = {
                method: _predictions(paths, severity, method, assignment)
                for method in methods_roster
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
    classes: str,
) -> list[dict[str, Any]]:
    """Equal-split contrasts: mean the per-split bootstrap arrays, then take percentiles."""
    per_split = [
        _split_contrasts(
            split_paths(base_paths, index), config, n_replicates, seed, classes
        )
        for index in range(3)
    ]
    shared = set.intersection(*(set(split) for split in per_split))
    dataset = config.get("dataset", {}).get("name", "")
    pooled_arrays = {
        key: np.mean(np.stack([split[key] for split in per_split]), axis=0)
        for key in shared
    }
    for key, effect in list(pooled_arrays.items()):
        assignment, severity, method, scale, kind = key
        deficit_key = (assignment, severity, "ce", scale, "deficit")
        if kind == "effect" and deficit_key in pooled_arrays:
            deficit = pooled_arrays[deficit_key]
            with np.errstate(divide="ignore", invalid="ignore"):
                pooled_arrays[(assignment, severity, method, scale, "recovery")] = (
                    np.where(deficit != 0, effect / deficit, np.nan)
                )
    rows = []
    for key in sorted(pooled_arrays):
        assignment, severity, method, scale, kind = key
        pooled = pooled_arrays[key]
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
                "gate_passed": classes == "tail"
                and kind == "deficit"
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
    parser.add_argument("--classes", choices=sorted(OUTPUTS), default="tail")
    args = parser.parse_args()
    assert not args.check or args.classes == "tail", "--check is tail-only"

    config = load_config(args.config)
    base_paths = ensure_dirs(config)
    n_replicates = int(config.get("analysis", {}).get("bootstrap_replicates", 10_000))
    seed = derive_seed(args.seed, "resampling")
    rows = _rows(base_paths, config, n_replicates, seed, args.classes)
    if args.check:
        _check(base_paths, rows)
    destination = base_paths["data"] / OUTPUTS[args.classes]
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("scaled-tail: wrote %d rows to %s", len(rows), destination)


if __name__ == "__main__":
    main()
