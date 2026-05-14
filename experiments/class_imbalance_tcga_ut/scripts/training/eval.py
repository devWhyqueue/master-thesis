from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from torch import nn

from scripts.mil.bags import AttentionMil, BagFeatureDataset, MdeMil, bag_collate
from scripts.training.support import (
    FeatureDataset,
    _evaluate,
    _load_numpy,
    _metric_payload,
)


def _train_sklearn(
    method: str, frame: pd.DataFrame, class_names: list[str], config: dict
) -> dict[str, dict[str, object]]:
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    train = cast(pd.DataFrame, frame[frame["split"] == "train"])
    val = cast(pd.DataFrame, frame[frame["split"] == "val"])
    test = cast(pd.DataFrame, frame[frame["split"] == "test"])
    x_train, y_train = _load_numpy(train, class_to_idx)
    estimator = _build_estimator(method, config, len(x_train))
    estimator.fit(x_train, y_train)
    return _predict_eval_splits(estimator, val, test, class_to_idx, class_names)


def _build_estimator(
    method: str, config: dict, n_train_rows: int
) -> KNeighborsClassifier | NearestCentroid:
    if method == "knn":
        n_neighbors = max(1, min(int(config["training"]["knn_k"]), n_train_rows))
        return KNeighborsClassifier(n_neighbors=n_neighbors)
    return NearestCentroid()


def _predict_eval_splits(
    estimator: KNeighborsClassifier | NearestCentroid,
    val: pd.DataFrame,
    test: pd.DataFrame,
    class_to_idx: dict[str, int],
    class_names: list[str],
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for name, split_frame in [("val", val), ("test", test)]:
        x_split, y_split = _load_numpy(split_frame, class_to_idx)
        preds = estimator.predict(x_split)
        results[name] = _evaluate_predictions(y_split, preds, class_names)
    return results


def _evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict[str, object]:
    payload = _metric_payload(y_true.tolist(), y_pred.tolist(), [], class_names)
    payload["accuracy"] = float((y_true == y_pred).mean())
    return payload


def _save_and_evaluate(
    model: nn.Module,
    val_dataset: FeatureDataset,
    test_dataset: FeatureDataset,
    class_names: list[str],
    device: torch.device,
    result_dir,
) -> dict[str, dict[str, object]]:
    val_results = _evaluate(model, val_dataset, class_names, device)
    test_results = _evaluate(model, test_dataset, class_names, device)
    torch.save(model.state_dict(), result_dir / "model.pt")
    return {"val": val_results, "test": test_results}


def _save_and_evaluate_bags(
    model: nn.Module,
    val_dataset: BagFeatureDataset,
    test_dataset: BagFeatureDataset,
    class_names: list[str],
    device: torch.device,
    result_dir,
) -> dict[str, dict[str, object]]:
    """Evaluate and save a feature-bag model."""
    val_results = _evaluate_bags(model, val_dataset, class_names, device)
    test_results = _evaluate_bags(model, test_dataset, class_names, device)
    torch.save(model.state_dict(), result_dir / "model.pt")
    return {"val": val_results, "test": test_results}


def _evaluate_bags(
    model: nn.Module,
    dataset: BagFeatureDataset,
    class_names: list[str],
    device: torch.device,
) -> dict[str, object]:
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, collate_fn=bag_collate
    )
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for bags, targets in loader:
            logits = _bag_logits(model, [bag.to(device) for bag in bags])
            probs = torch.softmax(logits, dim=1)
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
    return _metric_payload(y_true, y_pred, probabilities, class_names)


def _bag_logits(model: nn.Module, bags: list[torch.Tensor]) -> torch.Tensor:
    if isinstance(model, AttentionMil | MdeMil):
        return model.forward_bags(bags)[0]
    raise TypeError(f"Unsupported bag model: {type(model).__name__}")
