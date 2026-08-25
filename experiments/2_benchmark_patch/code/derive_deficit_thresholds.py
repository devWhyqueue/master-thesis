"""PLAN_3 Plan §2 / Verification: the noise-floor term of the deficit thresholds.

One-time computation, run before any gate row is read. For every dataset and
split (and, for tail NLL, every severity, since the tail group is
severity-specific), reports the standard deviation across the five
confirmation seeds of balanced-condition CE case-macro balanced accuracy and
tail-group macro NLL, then the resulting DISCRIMINATION_THRESHOLD /
CALIBRATION_THRESHOLD to paste into gates.py and the protocol.

Blind: only reads the balanced condition, never a balanced-minus-severity
contrast, so it never touches a gate row or a recovery ratio.

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
SEVERITIES = ("moderate", "severe")


@dataclass
class DispersionRow:
    dataset: str
    split: int
    severity: str | None
    endpoint: str
    sigma_seed: float


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
    for severity in SEVERITIES:
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
    rows: list[DispersionRow] = []
    for split_index in range(3):
        paths = split_paths(base_paths, split_index)
        rows += _split_dispersion(config_path.stem, split_index, paths, is_mil)
    return rows


def _log_rows(all_rows: list[DispersionRow]) -> None:
    for row in all_rows:
        tag = f"{row.dataset} split={row.split}" + (
            f" {row.severity}" if row.severity else ""
        )
        logger.info("%-45s %-9s sigma_seed=%.5f", tag, row.endpoint, row.sigma_seed)


def _report_thresholds(all_rows: list[DispersionRow]) -> None:
    """Apply the PLAN_3 §2 rule to the collected dispersion rows and log the result."""
    ba_sigmas = [r.sigma_seed for r in all_rows if r.endpoint == "ba"]
    nll_sigmas = [r.sigma_seed for r in all_rows if r.endpoint == "tail_nll"]
    max_ba_sigma = max(ba_sigmas, default=0.0)
    max_nll_sigma = max(nll_sigmas, default=0.0)

    discrimination_threshold = max(STABILITY_FLOOR, 2 * max_ba_sigma)
    raise_calibration = max_nll_sigma > 0.025
    calibration_threshold = (
        2 * max_nll_sigma if raise_calibration else CALIBRATION_ANCHOR
    )

    logger.info(
        "max sigma_seed (balanced-condition case-macro BA):       %.5f", max_ba_sigma
    )
    logger.info(
        "max sigma_seed (balanced-condition case-macro tail NLL): %.5f", max_nll_sigma
    )
    logger.info(
        "DISCRIMINATION_THRESHOLD = max(0.01, 2*sigma) = %.5f", discrimination_threshold
    )
    note = " (raised: noise floor exceeded 0.025)" if raise_calibration else ""
    logger.info("CALIBRATION_THRESHOLD    = %.5f%s", calibration_threshold, note)
    if max_ba_sigma > 0.005 or max_nll_sigma > 0.005:
        logger.warning(
            "sigma_seed exceeds 0.005 somewhere; pipeline is noisier than the "
            "design assumed (PLAN_3 Verification)."
        )


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
    _report_thresholds(all_rows)


if __name__ == "__main__":
    main()
