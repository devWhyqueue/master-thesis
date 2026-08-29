"""PLAN_3 Plan §2 / Verification: the noise-floor term of the deficit thresholds.

One-time, run before any gate row is read. Per dataset (and, for tail NLL,
per severity), reports the seed-dispersion sigma of balanced-condition CE
and the resulting DISCRIMINATION_THRESHOLDS / CALIBRATION_THRESHOLDS to paste
into gates.py -- grouped by dataset, not pooled, since a global max
calibrates every dataset to the noisiest one. Blind: only reads the balanced
condition, so it never touches a gate row or a recovery ratio.

    uv run python experiments/2_benchmark_patch/code/derive_deficit_thresholds.py
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imbalance_benchmark.analysis.inference.bootstrap import (
    weighted_balanced_accuracy,
    weighted_macro_nll,
)
from imbalance_benchmark.analysis.inference.context import (
    BootstrapContext,
    _tail_classes,
)
from imbalance_benchmark.analysis.inference.crossed_permutation import load_freeze
from imbalance_benchmark.analysis.query import load_seed_predictions
from imbalance_benchmark.common import ensure_dirs, load_config, split_paths

logger = logging.getLogger(__name__)

CONFIGS = (
    "bracs_patch.yaml",
    "camelyon16_patch.yaml",
    "panda_patch.yaml",
    "tcga_ut_patch.yaml",
)
# Report §app:construction-support pilot stability floor: the benchmark's own
# prespecified minimum-material balanced-accuracy change (PLAN_3 §2).
STABILITY_FLOOR = 0.01
CALIBRATION_ANCHOR = 0.05  # nats; kept unless the noise floor exceeds half of it


@dataclass
class DispersionRow:
    dataset: str
    split: int
    severity: str | None
    endpoint: str
    sigma_seed: float


@dataclass
class DatasetThresholds:
    dataset: str
    max_ba_sigma: float
    max_nll_sigma: float
    discrimination_threshold: float
    calibration_threshold: float
    calibration_raised: bool


def _seed_point_values(fn, n_seeds: int) -> np.ndarray:
    """Replicate-0 (unit-weight, observed-cohort) value of one metric per seed."""
    return np.array([fn(i)[0] for i in range(n_seeds)])


def _ba_sigma(
    balanced: dict, ctx: BootstrapContext, n_classes: int, n_seeds: int
) -> float:
    """Seed dispersion of balanced-condition CE case-macro balanced accuracy."""
    ba_seeds = _seed_point_values(
        lambda i: weighted_balanced_accuracy(
            balanced["labels"],
            balanced["preds"][i],
            ctx.weights,
            n_classes,
            ctx.case_class_divisor,
        ),
        n_seeds,
    )
    return float(np.std(ba_seeds, ddof=1))


def _tail_nll_sigma(
    balanced: dict, ctx: BootstrapContext, tail_classes: list[int], n_seeds: int
) -> float:
    """Seed dispersion of balanced-condition CE case-macro tail-group macro NLL."""
    nll_seeds = _seed_point_values(
        lambda i: weighted_macro_nll(
            balanced["labels"],
            balanced["probs"][i],
            ctx.weights,
            tail_classes,
            ctx.case_class_divisor,
        ),
        n_seeds,
    )
    return float(np.std(nll_seeds, ddof=1))


def _split_dispersion(
    dataset: str, split_index: int, paths: dict[str, Path], is_mil: bool
) -> list[DispersionRow]:
    """Dispersion rows for one dataset split, or [] if confirm isn't there yet."""
    if not (paths["data"] / "manifest.csv").exists():
        return []
    try:
        balanced = load_seed_predictions(paths, "balanced", "ce")
    except RuntimeError:
        return []  # balanced/ce confirm not run yet for this split
    if balanced is None:
        return []
    freeze = load_freeze(paths)
    ctx = BootstrapContext(paths, is_mil, n_replicates=1, seed=0)
    n_classes = len(balanced["class_names"])
    n_seeds = balanced["preds"].shape[0]
    rows = [
        DispersionRow(
            dataset,
            split_index,
            None,
            "ba",
            _ba_sigma(balanced, ctx, n_classes, n_seeds),
        )
    ]
    for severity in freeze.get("assignment_conditions", {}).get("native", {}):
        tail_classes = _tail_classes(
            freeze, balanced["class_names"], "native", severity
        )
        if not tail_classes:
            continue
        rows.append(
            DispersionRow(
                dataset,
                split_index,
                severity,
                "tail_nll",
                _tail_nll_sigma(balanced, ctx, tail_classes, n_seeds),
            )
        )
    return rows


def dataset_dispersion(config_path: Path) -> list[DispersionRow]:
    """Per-(split[, severity]) seed dispersion rows for one dataset config."""
    config = load_config(config_path)
    is_mil = config.get("dataset", {}).get("regime", "patch") == "wsi"
    base_paths = ensure_dirs(config)
    dataset_name = config["dataset"]["name"]
    rows: list[DispersionRow] = []
    for split_index in range(3):
        paths = split_paths(base_paths, split_index)
        rows += _split_dispersion(dataset_name, split_index, paths, is_mil)
    return rows


def _log_rows(all_rows: list[DispersionRow]) -> None:
    for row in all_rows:
        tag = f"{row.dataset} split={row.split}" + (
            f" {row.severity}" if row.severity else ""
        )
        logger.info("%-45s %-9s sigma_seed=%.5f", tag, row.endpoint, row.sigma_seed)


def _dataset_thresholds(dataset: str, rows: list[DispersionRow]) -> DatasetThresholds:
    """PLAN_3 §2 rule applied to one dataset's own dispersion rows, not the pooled max."""
    max_ba = max((r.sigma_seed for r in rows if r.endpoint == "ba"), default=0.0)
    max_nll = max((r.sigma_seed for r in rows if r.endpoint == "tail_nll"), default=0.0)
    raised = max_nll > 0.025
    return DatasetThresholds(
        dataset,
        max_ba,
        max_nll,
        max(STABILITY_FLOOR, 2 * max_ba),
        2 * max_nll if raised else CALIBRATION_ANCHOR,
        raised,
    )


def _log_dataset_threshold(t: DatasetThresholds) -> None:
    note = " (raised: noise floor exceeded 0.025)" if t.calibration_raised else ""
    logger.info(
        "%-12s max sigma ba=%.5f tail_nll=%.5f -> DISCRIMINATION=%.5f CALIBRATION=%.5f%s",
        t.dataset,
        t.max_ba_sigma,
        t.max_nll_sigma,
        t.discrimination_threshold,
        t.calibration_threshold,
        note,
    )
    if t.max_ba_sigma > 0.005 or t.max_nll_sigma > 0.005:
        logger.warning(
            "%s: sigma_seed exceeds 0.005; pipeline is noisier than the "
            "design assumed (PLAN_3 Verification).",
            t.dataset,
        )


def report_thresholds(all_rows: list[DispersionRow]) -> dict[str, DatasetThresholds]:
    """Apply the PLAN_3 §2 rule per dataset to the collected dispersion rows and log it."""
    by_dataset: dict[str, list[DispersionRow]] = {}
    for row in all_rows:
        by_dataset.setdefault(row.dataset, []).append(row)
    thresholds = {
        dataset: _dataset_thresholds(dataset, rows)
        for dataset, rows in sorted(by_dataset.items())
    }
    for t in thresholds.values():
        _log_dataset_threshold(t)
    return thresholds


def log_paste_ready(thresholds: dict[str, DatasetThresholds]) -> None:
    """Log the two dicts ready to paste into gates.py's threshold tables."""
    disc = {
        t.dataset: round(t.discrimination_threshold, 5) for t in thresholds.values()
    }
    cal = {t.dataset: round(t.calibration_threshold, 5) for t in thresholds.values()}
    logger.info("DISCRIMINATION_THRESHOLDS = %s", disc)
    logger.info("CALIBRATION_THRESHOLDS = %s", cal)


def main() -> None:
    """Compute and report the derived deficit-gate thresholds across all four datasets."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs",
    )
    args = parser.parse_args()

    all_rows: list[DispersionRow] = []
    for name in CONFIGS:
        all_rows += dataset_dispersion(args.config_dir / name)

    _log_rows(all_rows)
    log_paste_ready(report_thresholds(all_rows))


if __name__ == "__main__":
    main()
