"""Audit WSI-bag calibration metrics for cross-method consistency."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from scripts.common import ensure_dirs, load_config
from scripts.modeling.mil.metrics import _calibration_metrics

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    return parser.parse_args()


def _load_test(result_root: Path, method: str, seed: int) -> dict[str, object] | None:
    path = result_root / method / f"seed={seed}" / "test_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_seed(result_root: Path, methods: list[str], seed: int) -> list[str]:
    issues: list[str] = []
    reference = _load_test(result_root, methods[0], seed)
    if reference is None:
        return issues
    ref_labels = reference["labels"]
    ref_names = reference["class_names"]
    n_classes = len(ref_names)

    for method in methods:
        payload = _load_test(result_root, method, seed)
        if payload is None:
            continue
        if payload["class_names"] != ref_names:
            issues.append(f"seed={seed} {method}: class_names mismatch")
        if payload["labels"] != ref_labels:
            issues.append(f"seed={seed} {method}: label sequence mismatch")
        probs = np.asarray(payload["probabilities"], dtype=np.float64)
        sums = probs.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-4):
            issues.append(f"seed={seed} {method}: probabilities do not sum to 1")
        if probs.min() < -1e-6 or probs.max() > 1.0 + 1e-6:
            issues.append(f"seed={seed} {method}: probabilities outside [0, 1]")
        recomputed = _calibration_metrics(
            list(map(int, payload["labels"])),
            payload["probabilities"],
            n_classes,
        )
        for metric in (
            "negative_log_likelihood",
            "expected_calibration_error",
            "brier_score",
        ):
            stored = float(payload[metric])
            actual = float(recomputed[metric])
            if abs(stored - actual) > 1e-4:
                issues.append(
                    f"seed={seed} {method}: {metric} stored={stored:.6f} "
                    f"recomputed={actual:.6f}"
                )
    return issues


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    methods = list(config["wsi_bag_methods"])
    seeds = args.seeds or list(config["wsi_training"]["seeds"])
    issues: list[str] = []
    for seed in seeds:
        issues.extend(_audit_seed(paths["wsi_results"], methods, seed))

    logger.info(f"methods={methods}")
    logger.info(f"issues={len(issues)}")
    for issue in issues:
        logger.info(issue)

    reference_method = methods[0]
    rankmix = "rankmix_mil"
    for seed in seeds:
        baseline = _load_test(paths["wsi_results"], reference_method, seed)
        mixed = _load_test(paths["wsi_results"], rankmix, seed)
        if baseline is None or mixed is None:
            continue
        labels = np.asarray(baseline["labels"], dtype=np.int64)
        base_probs = np.asarray(baseline["probabilities"], dtype=np.float64)
        mix_probs = np.asarray(mixed["probabilities"], dtype=np.float64)
        base_true = base_probs[np.arange(len(labels)), labels]
        mix_true = mix_probs[np.arange(len(labels)), labels]
        print(
            f"seed={seed} n={len(labels)} "
            f"labels_match={baseline['labels'] == mixed['labels']} | "
            f"{reference_method}: NLL={baseline['negative_log_likelihood']:.4f} "
            f"ECE={baseline['expected_calibration_error']:.4f} "
            f"mean_conf={base_probs.max(axis=1).mean():.3f} "
            f"mean_p_true={base_true.mean():.3f} | "
            f"{rankmix}: NLL={mixed['negative_log_likelihood']:.4f} "
            f"ECE={mixed['expected_calibration_error']:.4f} "
            f"mean_conf={mix_probs.max(axis=1).mean():.3f} "
            f"mean_p_true={mix_true.mean():.3f}"
        )


if __name__ == "__main__":
    main()
