import logging
from typing import cast

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from common_code.metrics.calibration import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)

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
    predictions, labels, probabilities = _predict(model, dl_test, model_type)
    precision, recall, f1, support, conf_mat = _classification_arrays(
        labels, predictions, class_order
    )
    payload: dict[str, object] = _summary_metrics(
        labels, predictions, probabilities, precision, recall, f1, support
    )
    payload.update(
        {
            "confusion_matrix": conf_mat,
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": f1,
            "support_per_class": support,
            "class_order": class_order,
            "class_names": _class_names(dl_test, class_order),
            "preds": predictions,
            "labels": labels,
            "probabilities": probabilities,
        }
    )
    return payload


def _summary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    support: np.ndarray,
) -> dict[str, object]:
    present = support > 0
    return {
        "accuracy": float(np.sum(predictions == labels) / len(labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(np.mean(precision[present])),
        "macro_recall": float(np.mean(recall[present])),
        "macro_f1": float(np.mean(f1[present])),
        "negative_log_likelihood": negative_log_likelihood(
            labels.astype(int).tolist(), probabilities.tolist(), probabilities.shape[1]
        ),
        "brier_score": brier_score(
            labels.astype(int).tolist(), probabilities.tolist(), probabilities.shape[1]
        ),
        "expected_calibration_error": expected_calibration_error(
            labels.astype(int).tolist(), probabilities.tolist()
        ),
    }


def _predict(
    model: object, dl_test: DataLoader, model_type: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model_type == "mlp":
        return _predict_mlp(model, dl_test)
    if model_type in ("knn", "ncc"):
        return _predict_sklearn(model, dl_test)
    raise ValueError(f"Unknown model type: {model_type}")


def _predict_mlp(
    model: object, dl_test: DataLoader
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mlp = cast(MLP, model)
    predictions = np.array([])
    labels = np.array([])
    n_classes = _n_classes(dl_test)
    probabilities = np.empty((0, n_classes))
    with torch.no_grad():
        mlp.eval()
        dtype = next(mlp.parameters()).dtype
        for batch in dl_test:
            output = mlp(batch["features"].to(dtype)).squeeze()
            output = output.unsqueeze(0) if output.ndim == 1 else output
            probs = torch.softmax(output, dim=-1).detach().cpu().numpy()
            predictions = np.concatenate(
                [predictions, torch.argmax(output, dim=-1).detach().cpu().numpy()]
            )
            labels = np.concatenate([labels, batch["target"].detach().cpu().numpy()])
            probabilities = np.concatenate([probabilities, np.atleast_2d(probs)])
        mlp.train()
    return predictions, labels, probabilities


def _predict_sklearn(
    model: object, dl_test: DataLoader
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sklearn_model = cast(SKLearnModel, model)
    predictions = np.array([])
    labels = np.array([])
    n_classes = _n_classes(dl_test)
    probabilities = np.empty((0, n_classes))
    for batch in dl_test:
        features = np.atleast_2d(batch["features"].squeeze().detach().cpu().numpy())
        output = sklearn_model.predict(features)
        logger.info("output: %s", output)
        predictions = np.concatenate([predictions, output])
        labels = np.concatenate([labels, batch["target"].detach().cpu().numpy()])
        probabilities = np.concatenate(
            [
                probabilities,
                _sklearn_probabilities(sklearn_model, features, output, n_classes),
            ]
        )
    return predictions, labels, probabilities


def _classification_arrays(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_order: np.ndarray | list[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        average=None,
        labels=class_order,
        zero_division=cast(str, 0),
    )
    conf_mat = confusion_matrix(labels, predictions, labels=class_order)
    return (
        cast(np.ndarray, precision),
        cast(np.ndarray, recall),
        cast(np.ndarray, f1),
        cast(np.ndarray, support),
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


def _n_classes(dl_test: DataLoader) -> int:
    dataset = cast(TCGAUTDatasetImbalanced, dl_test.dataset)
    return dataset.get_n_classes()


def _sklearn_probabilities(
    model: SKLearnModel,
    features: np.ndarray,
    predictions: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    if hasattr(model.model, "predict_proba"):
        raw = cast(np.ndarray, model.model.predict_proba(features))
        probs = np.zeros((len(predictions), n_classes))
        for column, class_index in enumerate(model.model.classes_.astype(int)):
            probs[:, class_index] = raw[:, column]
        return probs
    probs = np.zeros((len(predictions), n_classes))
    for index, prediction in enumerate(predictions.astype(int)):
        probs[index, prediction] = 1.0
    return probs


def _labels(probabilities: np.ndarray) -> list[int]:
    return list(range(probabilities.shape[1]))
