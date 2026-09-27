"""Test-set bootstrap distributions of an explicit per-shard directory map.

Drives exp-16/17/25's own recall/probability-quality primitives
(``_patient_macro_recalls``, ``_probability_quality``) the same way
``centre.analyze.arm_accuracy`` and ``prevalence.analyze._thirds``/``_arm_dist`` do,
but from a directory chosen per shard instead of one arm name valid for every
shard -- the shape a validation-selected method's winning run needs. TS scaling
here is ``softmax(logits / validation_temperature)``, since
``breadth.calibrate.scaled_test_probabilities`` expects a ``temperature.json``
sidecar exp-40's run records never write (the temperature lives inline in the
record instead).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from imbalance_benchmark.analysis.calibration import softmax
from imbalance_benchmark.analysis.inference.context import BootstrapContext
from imbalance_benchmark.common import read_run_record

from breadth.analyze.secondary import _patient_macro_recalls, _probability_quality

from prevalence.fit import class_permutation

from analyze.paths import Shard, shard_keys

__all__ = ["selected_ba", "selected_thirds", "quality_distributions"]


def _labels_preds(result_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rec = read_run_record(
        result_dir, splits=("test",), array_fields=("labels", "preds")
    )
    if rec is None:
        raise RuntimeError(f"Missing run record at {result_dir}")
    test = rec["splits"]["test"]
    return np.asarray(test["labels"]), np.asarray(test["preds"])


def _probabilities(result_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rec = read_run_record(
        result_dir, splits=("test",), array_fields=("labels", "probabilities", "logits")
    )
    if rec is None:
        raise RuntimeError(f"Missing run record at {result_dir}")
    test = rec["splits"]["test"]
    labels = np.asarray(test["labels"])
    probs = np.asarray(test["probabilities"])
    scaled = softmax(np.asarray(test["logits"]) / float(rec["validation_temperature"]))
    return labels, probs, scaled


def selected_ba(
    dirs: dict[Shard, Path],
    ctxs: dict[int, BootstrapContext],
    perms: dict[int, np.ndarray],
    n_classes: int,
) -> np.ndarray:
    """(F, R) patient-macro BA distribution, ``arm_accuracy``'s own recipe over a per-shard dir map."""
    per_shard = [
        _patient_macro_recalls(ctxs[s], *_labels_preds(dirs[(s, d)]), n_classes)[
            perms[s]
        ]
        for s, d in shard_keys()
    ]
    return np.stack(per_shard).mean(axis=1) * 100.0


def selected_thirds(
    dirs: dict[Shard, Path], ctxs: dict[int, BootstrapContext], n_classes: int
) -> dict[str, float]:
    """Mean observed head/body/tail patient-macro recall (%), ``prevalence.analyze._thirds``'s own recipe."""
    vals: dict[str, list[float]] = {"head": [], "body": [], "tail": []}
    for s, d in shard_keys():
        groups = np.array_split(class_permutation(s, d, n_classes), 3)
        labels, preds = _labels_preds(dirs[(s, d)])
        recalls = (
            _patient_macro_recalls(ctxs[s], labels, preds, n_classes)[:, 0] * 100.0
        )
        for key, idx in zip(vals, groups):
            vals[key].append(float(recalls[idx].mean()))
    return {key: float(np.mean(v)) for key, v in vals.items()}


def quality_distributions(
    dirs: dict[Shard, Path], ctxs: dict[int, BootstrapContext], n_classes: int
) -> dict[str, np.ndarray]:
    """Raw and validation-TS macro NLL/ECE (F, R) distributions, ``prevalence.analyze._arm_dist``'s own recipe."""
    cols: dict[str, list[np.ndarray]] = {
        "nll": [],
        "ece": [],
        "nll_ts": [],
        "ece_ts": [],
    }
    for s, d in shard_keys():
        labels, probs, scaled = _probabilities(dirs[(s, d)])
        nll, ece = _probability_quality(ctxs[s], labels, probs, n_classes)
        nll_ts, ece_ts = _probability_quality(ctxs[s], labels, scaled, n_classes)
        cols["nll"].append(nll)
        cols["ece"].append(ece)
        cols["nll_ts"].append(nll_ts)
        cols["ece_ts"].append(ece_ts)
    return {key: np.stack(vals) for key, vals in cols.items()}
