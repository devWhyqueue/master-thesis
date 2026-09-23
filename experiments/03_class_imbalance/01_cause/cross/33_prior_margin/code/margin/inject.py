"""Inject stage: post-hoc prior injection into stored r1 test probabilities (0 new fits).

Q{rho} = argmax(log p_r1 + log(n_c / sum n)), n_c from r{rho}'s realized ``class_counts``
(exp-27/28's ``Lr`` adjustment, sign flipped, source fixed at r1). QT{rho} is the same injection
on r1's validation-temperature-scaled probabilities (sensitivity check).

H3's 7-class subset screen reuses ``injected_preds``/``rho_shares`` from here; see
``margin.subsets``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from decodability.evidence import load_freeze_meta
from imbalance_benchmark.common import (
    RUN_RECORD_NAME,
    ensure_dirs,
    read_run_record,
    split_paths,
    write_run_record,
)

from breadth import exp2_split_paths
from breadth.calibrate import scaled_test_probabilities

from sites import allocation_dir

from spectrum import baseline_config

from centre import N_DRAWS, N_SPLITS

from margin import RATIOS_NEW

__all__ = ["run_inject", "local_class_names", "rho_shares", "injected_preds"]

logger = logging.getLogger(__name__)

_PROBABILITY_FLOOR = np.finfo(np.float64).tiny


def local_class_names(config: dict[str, Any], split_idx: int) -> list[str]:
    """This split's own frozen class order, matching its stored run records' label ints."""
    return list(load_freeze_meta(exp2_split_paths(config, split_idx))["class_names"])


def rho_shares(k: int, rho: float) -> np.ndarray:
    """Exponential head-to-tail share profile (exp-02's allocator's weight formula, k classes)."""
    if k == 1:
        return np.ones(1)
    w = np.array([rho ** (-i / (k - 1)) for i in range(k)], dtype=np.float64)
    return w / w.sum()


def injected_preds(probs: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """argmax(log p + log shares): inject a class-share prior into stored probabilities."""
    adjusted = np.log(np.maximum(probs, _PROBABILITY_FLOOR)) + np.log(shares)
    return np.argmax(adjusted, axis=1)


def _target_shares(class_counts: dict[str, int], names: list[str]) -> np.ndarray:
    counts = np.array([class_counts[n] for n in names], dtype=np.float64)
    return counts / counts.sum()


def _write_one(
    out_dir: Path,
    arm: str,
    source_arm: str,
    labels: np.ndarray,
    probs: np.ndarray,
    class_counts: dict[str, int],
    names: list[str],
) -> None:
    if (out_dir / RUN_RECORD_NAME).exists():
        return
    preds = injected_preds(probs, _target_shares(class_counts, names))
    write_run_record(
        out_dir,
        {
            "arm": arm,
            "injected_from": "r1",
            "prior_source": source_arm,
            "class_counts": class_counts,
            # No stored probabilities: downstream readers of Q{rho}/QT{rho} (arm_accuracy /
            # _recall_matrix) only ever load labels+preds, and r1's own probabilities already
            # live in its own record -- storing them again here would just duplicate ~50MB/draw.
            "splits": {"test": {"labels": labels, "preds": preds}},
        },
        keep_arrays=True,
    )


def _inject_split_draw(
    config: dict[str, Any],
    own_paths: dict[str, Path],
    ds_paths: dict[str, Path],
    split_idx: int,
    draw_idx: int,
) -> None:
    names = local_class_names(config, split_idx)
    r1_dir = allocation_dir(ds_paths, "r1", draw_idx)
    r1 = read_run_record(
        r1_dir, splits=("test",), array_fields=("labels", "probabilities")
    )
    if r1 is None:
        raise RuntimeError(f"Missing r1 run record at {r1_dir}")
    test = r1["splits"]["test"]
    labels = np.asarray(test["labels"])
    probs_raw = np.asarray(test["probabilities"])
    probs_ts = scaled_test_probabilities(r1_dir, probs_raw)
    for r in RATIOS_NEW:
        rho_rec = read_run_record(
            allocation_dir(ds_paths, f"r{r}", draw_idx), splits=(), array_fields=()
        )
        if rho_rec is None:
            raise RuntimeError(f"Missing r{r} run record for class_counts")
        class_counts = rho_rec["class_counts"]
        _write_one(
            allocation_dir(own_paths, f"Q{r}", draw_idx),
            f"Q{r}",
            f"r{r}",
            labels,
            probs_raw,
            class_counts,
            names,
        )
        _write_one(
            allocation_dir(own_paths, f"QT{r}", draw_idx),
            f"QT{r}",
            f"r{r}",
            labels,
            probs_ts,
            class_counts,
            names,
        )


def run_inject(config: dict[str, Any]) -> None:
    """Write every Q{rho}/QT{rho} run record for every (split, draw) of this dataset."""
    ds_config = baseline_config(config, "prevalence_outputs")
    for s in range(N_SPLITS):
        own_paths = split_paths(ensure_dirs(config), s)
        ds_paths = split_paths(ensure_dirs(ds_config), s)
        for d in range(N_DRAWS):
            logger.info("Injecting split %d, draw %d", s, d)
            _inject_split_draw(config, own_paths, ds_paths, s, d)
