from __future__ import annotations

import numpy as np

from imbalance_benchmark.analysis.predictors import separability as sep


def _reference_knn_and_nn(ref_x, ref_y, val_x, val_y, n_classes, k=5):
    """Pre-refactor formulation: one unchunked matrix, full argsort for top-k."""
    d2 = (
        (val_x**2).sum(axis=1, keepdims=True)
        - 2.0 * val_x @ ref_x.T
        + (ref_x**2).sum(axis=1)[None, :]
    )
    k = min(k, ref_x.shape[0])
    neighbor_idx = np.argsort(d2, axis=1)[:, :k]
    neighbor_labels = ref_y[neighbor_idx]
    preds = np.array(
        [np.bincount(row, minlength=n_classes).argmax() for row in neighbor_labels]
    )
    nearest = d2.argmin(axis=1)
    nn_correct = ref_y[nearest] == val_y
    return preds, nn_correct


def _synthetic(seed: int, n_ref: int = 37, n_val: int = 130, n_classes: int = 4, n_features: int = 6):
    rng = np.random.default_rng(seed)
    ref_x = rng.normal(size=(n_ref, n_features))
    ref_y = rng.integers(0, n_classes, size=n_ref)
    val_x = rng.normal(size=(n_val, n_features))
    val_y = rng.integers(0, n_classes, size=n_val)
    return ref_x, ref_y, val_x, val_y


def test_argpartition_matches_argsort_knn_labels():
    ref_x, ref_y, val_x, val_y = _synthetic(seed=0)
    n_classes = 4
    preds, _ = sep._knn_and_nn_probe(ref_x, ref_y, val_x, val_y, n_classes)
    expected_preds, _ = _reference_knn_and_nn(ref_x, ref_y, val_x, val_y, n_classes)
    np.testing.assert_array_equal(preds, expected_preds)


def test_chunked_distances_match_unchunked_1nn():
    ref_x, ref_y, val_x, val_y = _synthetic(seed=1, n_val=9000)
    n_classes = 4
    # Force multiple chunks (val rows >> chunk size) via the module-level constant.
    assert val_x.shape[0] > sep._CHUNK_SIZE
    _, nn_correct = sep._knn_and_nn_probe(ref_x, ref_y, val_x, val_y, n_classes)
    _, expected_nn_correct = _reference_knn_and_nn(ref_x, ref_y, val_x, val_y, n_classes)
    np.testing.assert_array_equal(nn_correct, expected_nn_correct)


def test_shared_pass_matches_public_functions():
    ref_x, ref_y, val_x, val_y = _synthetic(seed=2)
    n_classes = 4
    metrics = sep.probe_metrics(ref_x, ref_y, val_x, val_y, n_classes)
    assert metrics["knn_macro_recall"] == sep.balanced_knn_macro_recall(
        ref_x, ref_y, val_x, val_y, n_classes
    )
    np.testing.assert_array_equal(
        metrics["per_class_nn_error"],
        sep.per_class_nn_error(ref_x, ref_y, val_x, val_y, n_classes).tolist(),
    )
