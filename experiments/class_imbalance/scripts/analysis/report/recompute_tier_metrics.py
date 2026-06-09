import logging
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.common import ensure_dirs, load_config, read_run_record, write_run_record
from scripts.modeling.mil.metrics import _tier_metrics
from scripts.modeling.training.support_tiers import tier_support_for_classes
from scripts.analysis.results import connect, init_schema, load_class_distribution

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse tier-metric recomputation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=["patch", "wsi_bag"],
        dest="benchmarks",
    )
    return parser.parse_args()


def _benchmark_settings(
    config: dict, benchmark: str
) -> tuple[list[str], list[int], str, str]:
    if benchmark == "patch":
        return (
            list(config["patch_methods"]),
            list(config["patch_training"]["seeds"]),
            "patch_results",
            "patch_image",
        )
    return (
        list(config["wsi_bag_methods"]),
        list(config["wsi_training"]["seeds"]),
        "wsi_results",
        "wsi_bag",
    )


def _recompute_payload(payload: dict, tier_support: np.ndarray) -> bool:
    precision = np.asarray(payload["precision_per_class"], dtype=np.float64)
    recall = np.asarray(payload["recall_per_class"], dtype=np.float64)
    f1 = np.asarray(payload["f1_per_class"], dtype=np.float64)
    support = np.asarray(payload["support_per_class"], dtype=np.int64)
    updated = _tier_metrics(precision, recall, f1, support, tier_support)
    if payload.get("support_tier_metrics") == updated["support_tier_metrics"]:
        return False
    payload["support_tier_metrics"] = updated["support_tier_metrics"]
    return True


def _update_db_split(connection, run_id: str, split: str, payload: dict) -> None:
    extended = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "negative_log_likelihood",
            "brier_score",
            "expected_calibration_error",
            "labels",
            "preds",
            "probabilities",
            "class_names",
        }
    }
    connection.execute(
        """
        UPDATE eval_results
        SET extended_json = ?
        WHERE run_id = ? AND split = ?
        """,
        (json.dumps(extended, sort_keys=True), run_id, split),
    )


def _tier_support(payload: dict, distribution) -> np.ndarray:
    slide_counts = dict(
        zip(
            distribution["cancer_type"].astype(str),
            distribution["n_slides"].astype(int),
            strict=True,
        )
    )
    return tier_support_for_classes(list(payload["class_names"]), slide_counts)


def _recompute_record(
    connection,
    result_dir: Path,
    record: dict,
    storage_benchmark: str,
    method: str,
    seed: int,
    distribution,
) -> bool:
    splits = record.get("splits", {})
    if not isinstance(splits, dict):
        return False
    changed = False
    for split, payload in splits.items():
        if not isinstance(payload, dict):
            continue
        tier_support = _tier_support(payload, distribution)
        if _recompute_payload(payload, tier_support):
            changed = True
            run_id = f"{storage_benchmark}:{method}:seed={seed}"
            _update_db_split(connection, run_id, str(split), payload)
    if changed:
        write_run_record(result_dir, record)
    return changed


def _load_distribution(paths: dict[str, Path]):
    connection = connect(paths["db"])
    init_schema(connection)
    distribution = load_class_distribution(connection, paths)
    connection.close()
    if distribution.empty:
        raise FileNotFoundError(
            "Missing class distribution required for tier assignment."
        )
    return distribution


def _scan_runs(
    connection,
    paths: dict[str, Path],
    methods: list[str],
    seeds: list[int],
    result_key: str,
    storage_benchmark: str,
    distribution,
) -> tuple[int, int]:
    updated = 0
    scanned = 0
    for method in methods:
        for seed in seeds:
            result_dir = paths[result_key] / method / f"seed={seed}"
            record = read_run_record(result_dir)
            if record is None:
                continue
            splits = record.get("splits", {})
            if isinstance(splits, dict):
                scanned += len(splits)
            if _recompute_record(
                connection,
                result_dir,
                record,
                storage_benchmark,
                method,
                seed,
                distribution,
            ):
                updated += 1
    return scanned, updated


def recompute_benchmark(
    paths: dict[str, Path], config: dict, benchmark: str
) -> tuple[int, int]:
    """Refresh support-tier summaries in stored run records and the database."""
    methods, seeds, result_key, storage_benchmark = _benchmark_settings(
        config, benchmark
    )
    connection = connect(paths["db"])
    init_schema(connection)
    scanned, updated = _scan_runs(
        connection,
        paths,
        methods,
        seeds,
        result_key,
        storage_benchmark,
        _load_distribution(paths),
    )
    connection.commit()
    connection.close()
    return scanned, updated


def main() -> None:
    """Recompute support-tier metrics from stored per-class results."""
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    benchmarks = args.benchmarks or ["patch", "wsi_bag"]
    for benchmark in benchmarks:
        scanned, updated = recompute_benchmark(paths, config, benchmark)
        logger.info(f"{benchmark}: updated {updated}/{scanned} result files")


if __name__ == "__main__":
    main()
