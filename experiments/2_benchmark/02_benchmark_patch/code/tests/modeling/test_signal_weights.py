from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from imbalance_benchmark.datasets.data import ImbalanceDataset
from imbalance_benchmark.datasets.features.cache import reset_feature_bank
from imbalance_benchmark.modeling.context import MATCHED_BETA_METHOD, matched_beta_config
from imbalance_benchmark.modeling.losses import effective_number
from imbalance_benchmark.modeling.training import _init_criterion

DIM = 4


def _base_ctx(n_classes: int) -> dict[str, object]:
    return {
        "class_counts": np.zeros(n_classes),
        "difficulty": {},
        "train_dataset": SimpleNamespace(classes=[]),
    }


def _weight(method: str, param: float, ctx: dict[str, object]) -> torch.Tensor:
    n_classes = len(ctx["class_counts"])  # type: ignore[arg-type]
    criterion = _init_criterion(
        method, param, n_classes, np.zeros(1, dtype=int), torch.device("cpu"), ctx
    )
    return criterion.weight  # type: ignore[attr-defined]


def _case_dataset(tmp_path: Path, classes: list[str], cases: list[int]) -> ImbalanceDataset:
    tmp_path.mkdir(parents=True, exist_ok=True)
    reset_feature_bank()
    rows = []
    for cls, n_cases in zip(classes, cases):
        for i in range(n_cases):
            slide_id = f"{cls}_S{i}"
            feature_path = tmp_path / f"{slide_id}.pt"
            torch.save(torch.randn(1, DIM), feature_path)
            rows.append(
                {
                    "case_id": f"{cls}_P{i}",
                    "slide_id": slide_id,
                    "cancer_type": cls,
                    "feature_path": str(feature_path),
                }
            )
    manifest = tmp_path / f"manifest_{'_'.join(classes)}.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return ImbalanceDataset(manifest, class_names=classes)


def test_class_balanced_ce_weight_is_mean_one_and_favors_the_tail() -> None:
    ctx = _base_ctx(3)
    ctx["class_counts"] = np.array([100.0, 100.0, 100.0])
    w_balanced = _weight("class_balanced_ce", 0.99, ctx)
    ctx["class_counts"] = np.array([100.0, 100.0, 1.0])
    w_severe = _weight("class_balanced_ce", 0.99, ctx)

    assert torch.allclose(w_balanced.sum(), torch.tensor(3.0), atol=1e-4)
    assert torch.allclose(w_severe.sum(), torch.tensor(3.0), atol=1e-4)
    assert torch.allclose(w_balanced, torch.ones(3), atol=1e-3)
    assert w_severe[2] > w_severe[0]


def test_independent_support_ce_weight_is_mean_one_and_favors_the_tail(
    tmp_path: Path,
) -> None:
    classes = ["a", "b", "c"]
    ctx = _base_ctx(3)
    ctx["class_counts"] = np.array([1.0, 1.0, 1.0])
    ctx["train_dataset"] = _case_dataset(tmp_path / "balanced", classes, [50, 50, 50])
    w_balanced = _weight("independent_support_ce", 0.95, ctx)

    ctx["train_dataset"] = _case_dataset(tmp_path / "severe", classes, [50, 50, 5])
    w_severe = _weight("independent_support_ce", 0.95, ctx)

    assert torch.allclose(w_balanced.sum(), torch.tensor(3.0), atol=1e-4)
    assert torch.allclose(w_severe.sum(), torch.tensor(3.0), atol=1e-4)
    assert w_severe[2] > w_severe[0]


def test_independent_support_ce_reduces_to_class_balanced_ce_when_gc_equals_nc(
    tmp_path: Path,
) -> None:
    classes = ["a", "b", "c"]
    counts = [200, 40, 4]
    beta = 0.97

    ctx = _base_ctx(3)
    ctx["class_counts"] = np.array(counts, dtype=float)
    w_cb = _weight("class_balanced_ce", beta, ctx)

    ctx["train_dataset"] = _case_dataset(tmp_path, classes, counts)
    w_isw = _weight("independent_support_ce", beta, ctx)

    assert torch.allclose(w_cb, w_isw, atol=1e-6)


def test_pilot_difficulty_ce_weight_is_mean_one_and_favors_the_hard_class() -> None:
    classes = ["a", "b", "c"]
    ctx = _base_ctx(3)
    ctx["train_dataset"] = SimpleNamespace(classes=classes)
    ctx["difficulty"] = dict(zip(classes, [0.05, 0.1, 0.15]))
    w_easy = _weight("pilot_difficulty_ce", 0.5, ctx)

    ctx["difficulty"] = dict(zip(classes, [0.05, 0.1, 0.9]))
    w_hard = _weight("pilot_difficulty_ce", 0.5, ctx)

    assert torch.allclose(w_easy.sum(), torch.tensor(3.0), atol=1e-4)
    assert torch.allclose(w_hard.sum(), torch.tensor(3.0), atol=1e-4)
    assert w_hard[2] > w_hard[0]


def test_effective_number_floors_zero_count_classes_at_one() -> None:
    eff = effective_number(np.array([0.0, 10.0]), beta=0.9)
    assert eff[0] == effective_number(np.array([1.0]), beta=0.9)[0]


def test_matched_beta_config_borrows_isw_lr_and_cb_beta() -> None:
    configs = {
        "independent_support_ce": {"lr": 3e-4, "parameter": 0.8},
        "class_balanced_ce": {"lr": 1e-3, "parameter": 0.999},
    }
    assert matched_beta_config(configs) == {"lr": 3e-4, "parameter": 0.999}


def test_matched_beta_method_weights_like_independent_support_ce(
    tmp_path: Path,
) -> None:
    classes = ["a", "b", "c"]
    ctx = _base_ctx(3)
    ctx["train_dataset"] = _case_dataset(tmp_path, classes, [200, 40, 4])

    w_isw = _weight("independent_support_ce", 0.999, ctx)
    w_matched = _weight(MATCHED_BETA_METHOD, 0.999, ctx)

    assert torch.allclose(w_isw, w_matched, atol=1e-6)
