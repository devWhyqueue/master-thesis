from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from scripts.training.support import _load_numpy, _metric_payload


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
