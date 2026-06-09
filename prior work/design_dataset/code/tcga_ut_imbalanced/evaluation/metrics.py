import logging
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from tcga_ut_imbalanced.models.mlp import MLP
from tcga_ut_imbalanced.models.sklearn import SKLearnModel
from tcga_ut_imbalanced.data.dataset import TCGAUTDatasetImbalanced

logger = logging.getLogger(__name__)


def test_model(
    model: object,
    dl_test: DataLoader,
    model_type: str = "mlp",
    class_order: np.ndarray | list[int] | None = None,
) -> dict[str, object]:
    """Evaluate a model and return aggregate metrics and predictions."""
    predictions, labels = _predict(model, dl_test, model_type)
    precision, recall, conf_mat = _classification_arrays(
        labels, predictions, class_order
    )
    return {
        "accuracy": float(np.sum(predictions == labels) / len(labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "confusion_matrix": conf_mat,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "class_order": class_order,
        "class_names": _class_names(dl_test, class_order),
        "preds": predictions,
        "labels": labels,
    }


def _predict(
    model: object, dl_test: DataLoader, model_type: str
) -> tuple[np.ndarray, np.ndarray]:
    if model_type == "mlp":
        return _predict_mlp(model, dl_test)
    if model_type in ("knn", "ncc"):
        return _predict_sklearn(model, dl_test)
    raise ValueError(f"Unknown model type: {model_type}")


def _predict_mlp(model: object, dl_test: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    mlp = cast(MLP, model)
    predictions = np.array([])
    labels = np.array([])
    with torch.no_grad():
        mlp.eval()
        for batch in dl_test:
            first_layer = cast(nn.Linear, mlp.model[0])
            output = mlp(batch["features"].to(first_layer.weight.dtype)).squeeze()
            predictions = np.concatenate(
                [predictions, torch.argmax(output, dim=-1).detach().cpu().numpy()]
            )
            labels = np.concatenate([labels, batch["target"].detach().cpu().numpy()])
        mlp.train()
    return predictions, labels


def _predict_sklearn(
    model: object, dl_test: DataLoader
) -> tuple[np.ndarray, np.ndarray]:
    sklearn_model = cast(SKLearnModel, model)
    predictions = np.array([])
    labels = np.array([])
    for batch in dl_test:
        output = sklearn_model.predict(
            batch["features"].squeeze().detach().cpu().numpy()
        )
        logger.info("output: %s", output)
        predictions = np.concatenate([predictions, output])
        labels = np.concatenate([labels, batch["target"].detach().cpu().numpy()])
    return predictions, labels


def _classification_arrays(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_order: np.ndarray | list[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, _, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average=None,
        labels=class_order,
    )
    conf_mat = confusion_matrix(labels, predictions, labels=class_order)
    return (
        cast(np.ndarray, precision),
        cast(np.ndarray, recall),
        cast(np.ndarray, conf_mat),
    )


def _class_names(
    dl_test: DataLoader, class_order: np.ndarray | list[int] | None
) -> list[str]:
    dataset = cast(TCGAUTDatasetImbalanced, dl_test.dataset)
    int_to_class_map = dataset.get_int_to_class_map()
    if class_order is not None:
        return [int_to_class_map[index] for index in class_order]
    return [int_to_class_map[index] for index in range(dataset.get_n_classes())]
