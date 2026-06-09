import logging
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.common import ensure_dirs, load_config
from scripts.modeling.mil.metrics import _tier_metrics
from scripts.modeling.training.support import dataset_tier_support

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
) -> tuple[list[str], list[int], str]:
    if benchmark == "patch":
        return (
            list(config["patch_methods"]),
            list(config["patch_training"]["seeds"]),
            "patch_results",
        )
    return (
        list(config["wsi_bag_methods"]),
        list(config["wsi_training"]["seeds"]),
        "wsi_results",
    )


def _recompute_file(path: Path, tier_support: np.ndarray) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    precision = np.asarray(payload["precision_per_class"], dtype=np.float64)
    recall = np.asarray(payload["recall_per_class"], dtype=np.float64)
    f1 = np.asarray(payload["f1_per_class"], dtype=np.float64)
    support = np.asarray(payload["support_per_class"], dtype=np.int64)
    updated = _tier_metrics(precision, recall, f1, support, tier_support)
    if payload.get("support_tier_metrics") == updated["support_tier_metrics"]:
        return False
    payload["support_tier_metrics"] = updated["support_tier_metrics"]
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return True


def recompute_benchmark(
    paths: dict[str, Path], config: dict, benchmark: str
) -> tuple[int, int]:
    """Refresh support-tier summaries in stored result JSON files."""
    methods, seeds, result_key = _benchmark_settings(config, benchmark)
    updated = 0
    scanned = 0
    for method in methods:
        for seed in seeds:
            result_dir = paths[result_key] / method / f"seed={seed}"
            for split in ("val", "test"):
                path = result_dir / f"{split}_results.json"
                if not path.exists():
                    continue
                scanned += 1
                payload = json.loads(path.read_text(encoding="utf-8"))
                tier_support = dataset_tier_support(
                    list(payload["class_names"]),
                    paths["tables"] / "class_distribution.csv",
                )
                if tier_support is None:
                    raise FileNotFoundError(
                        "Missing class_distribution.csv required for tier assignment."
                    )
                if _recompute_file(path, tier_support):
                    updated += 1
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
