"""Scientific invariants of exp-47's descriptive metrics (weighting, confusion, margins, transitions)."""

from __future__ import annotations

import numpy as np
import pytest

from gap.metrics import confusion, margins, patient_weights, transitions

# Class 0: patient "a" has 3 patches, "b" has 1. Class 1: patient "a" again (a BRACS patient with
# two lesion classes) with 2 patches, and patient "c" with 2.
LABELS = np.array([0, 0, 0, 0, 1, 1, 1, 1])
PATIENTS = np.array(["a", "a", "a", "b", "a", "a", "c", "c"])


def test_patient_weights_equal_class_and_patient_mass() -> None:
    """Each class sums to one; each patient-class group gets an equal share of its class."""
    w = patient_weights(LABELS, PATIENTS)
    np.testing.assert_allclose(w, [1 / 6, 1 / 6, 1 / 6, 1 / 2, 1 / 4, 1 / 4, 1 / 4, 1 / 4])


def test_patient_in_two_classes_keeps_separate_groups() -> None:
    """Patient "a"'s class-1 rows are not pooled with its class-0 rows."""
    w = patient_weights(LABELS, PATIENTS)
    assert w[LABELS == 1].sum() == pytest.approx(1.0)
    assert w[(PATIENTS == "a") & (LABELS == 1)].sum() == pytest.approx(0.5)


def test_confusion_rows_sum_to_one_and_diagonal_is_patient_macro_ba() -> None:
    """Mean diagonal equals the class mean of patient-mean patch accuracy."""
    preds = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    w = patient_weights(LABELS, PATIENTS)
    m = confusion(LABELS, preds, w, 2)
    np.testing.assert_allclose(m.sum(axis=1), 1.0)
    recall0 = np.mean([1 / 3, 1.0])
    recall1 = np.mean([1 / 2, 1.0])
    np.testing.assert_allclose(np.diag(m), [recall0, recall1])


def test_confusion_rejects_unnormalized_weights() -> None:
    """Weights that do not give each class unit mass are a caller error, not a silent rescale."""
    with pytest.raises(ValueError):
        confusion(LABELS, LABELS, np.ones(len(LABELS)), 2)


def test_margins_positive_exactly_when_correct() -> None:
    """Margin = log p_true - max rival log p: positive iff argmax is the true class."""
    probs = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
    labels = np.array([0, 0, 2])
    m, ties = margins(probs, labels)
    np.testing.assert_allclose(m, [np.log(0.7 / 0.2), np.log(0.2 / 0.5), np.log(0.8 / 0.1)])
    assert not ties.any()


def test_margins_flag_ties_instead_of_counting_them_as_correct() -> None:
    """An exact tie has margin 0 and is flagged, whatever argmax happened to predict."""
    probs = np.array([[0.4, 0.4, 0.2], [0.0, 0.0, 1.0]])
    labels = np.array([1, 0])
    m, ties = margins(probs, labels)
    assert m[0] == 0.0 and ties[0]
    # True class and best rival underflow differently: true is 0, rival 1 -> finite negative.
    assert np.isfinite(m[1]) and m[1] < 0 and not ties[1]


def test_transitions_lost_minus_gained_equals_damage() -> None:
    """Weighted lost minus gained correct predictions reproduces BA(reference) - BA(changed)."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 200)
    patients = rng.integers(0, 15, 200).astype(str)
    reference = np.where(rng.random(200) < 0.6, labels, rng.integers(0, 3, 200))
    changed = np.where(rng.random(200) < 0.5, labels, rng.integers(0, 3, 200))
    w = patient_weights(labels, patients)
    t = transitions(reference, changed, labels, w)
    ba_ref = np.diag(confusion(labels, reference, w, 3)).mean()
    ba_chg = np.diag(confusion(labels, changed, w, 3)).mean()
    assert t["lost"] - t["gained"] == pytest.approx(ba_ref - ba_chg)
    np.testing.assert_allclose(t["lost_per_class"] - t["gained_per_class"],
                               np.diag(confusion(labels, reference, w, 3))
                               - np.diag(confusion(labels, changed, w, 3)))


# --- gap.records: identity checks fail before anything is pooled -------------------------------

from gap.records import canonicalize, check_stored_ba  # noqa: E402

CANONICAL = ["A", "B", "C"]
LOCAL = ["C", "A", "B"]  # this split froze its classes in a different order
PERM = np.array([LOCAL.index(n) for n in CANONICAL])  # canonical_permutation's convention


def test_canonicalize_maps_labels_preds_and_probability_columns() -> None:
    """Local class indices and probability columns are reordered into the canonical order."""
    labels_local = np.array([0, 1, 2])  # C, A, B
    preds_local = np.array([1, 1, 0])  # A, A, C
    probs_local = np.array([[0.6, 0.3, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    names = np.array(["C", "A", "B"])
    labels, preds, probs = canonicalize(labels_local, preds_local, probs_local, PERM, names, CANONICAL)
    np.testing.assert_array_equal(labels, [2, 0, 1])
    np.testing.assert_array_equal(preds, [0, 0, 2])
    np.testing.assert_allclose(probs[0], [0.3, 0.1, 0.6])


def test_canonicalize_rejects_label_order_mismatch_with_manifest() -> None:
    """A record whose labels disagree with the manifest's class names fails loudly."""
    with pytest.raises(RuntimeError, match="label"):
        canonicalize(np.array([0, 1]), np.array([0, 1]), np.full((2, 3), 1 / 3), PERM,
                     np.array(["A", "A"]), CANONICAL)


def test_canonicalize_rejects_length_mismatch() -> None:
    """A record with a different row count than the manifest's test split fails loudly."""
    with pytest.raises(RuntimeError, match="rows"):
        canonicalize(np.array([0, 1]), np.array([0, 1]), np.full((2, 3), 1 / 3), PERM,
                     np.array(["C"]), CANONICAL)


def test_check_stored_ba_rejects_disagreement() -> None:
    """Recomputed patient-macro BA must equal the run's stored endpoint."""
    preds = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    w = patient_weights(LABELS, PATIENTS)
    ba = np.diag(confusion(LABELS, preds, w, 2)).mean()
    assert check_stored_ba(LABELS, preds, w, ba) == pytest.approx(ba)
    with pytest.raises(RuntimeError, match="balanced accuracy"):
        check_stored_ba(LABELS, preds, w, ba + 1e-4)


# --- gap.geometry: deterministic sampling, disjoint patients, neighbour agreement --------------

import pandas as pd  # noqa: E402

from gap.geometry import (  # noqa: E402
    assert_disjoint,
    chance_adjusted,
    knn_agreement,
    sample_rows,
    variance_shares,
)


def _frame() -> pd.DataFrame:
    rows = []
    for label, patient, n in [("A", "p1", 50), ("A", "p2", 10), ("A", "p3", 40), ("B", "p1", 35),
                              ("B", "p4", 60)]:
        rows += [{"cancer_type": label, "case_id": patient}] * n
    return pd.DataFrame(rows)


def test_sample_rows_caps_patients_and_patches_without_replacement() -> None:
    """At most n_patients groups per class, at most n_patches rows each, rows unique."""
    frame = _frame()
    rows = sample_rows(frame, n_patients=2, n_patches=32, seed=0)
    picked = frame.loc[rows]
    assert len(set(rows)) == len(rows)
    counts = picked.groupby(["cancer_type", "case_id"]).size()
    assert counts.max() <= 32
    assert picked.groupby("cancer_type")["case_id"].nunique().max() <= 2
    # p2 has only 10 patches: kept whole when sampled, never padded.
    if ("A", "p2") in counts.index:
        assert counts[("A", "p2")] == 10


def test_sample_rows_is_deterministic_and_keeps_patient_class_groups() -> None:
    """Same seed, same rows; p1 can appear in both classes as two groups."""
    frame = _frame()
    a = sample_rows(frame, n_patients=10, n_patches=32, seed=3)
    assert a == sample_rows(frame, n_patients=10, n_patches=32, seed=3)
    picked = frame.loc[a]
    assert set(picked.loc[picked["case_id"] == "p1", "cancer_type"]) == {"A", "B"}


def test_sample_rows_full_groups_only_gives_equal_class_counts() -> None:
    """Reference mode keeps only full groups and the same patient count in every class."""
    frame = _frame()
    rows = sample_rows(frame, n_patients=10, n_patches=32, seed=0, full_groups=True)
    per_class = frame.loc[rows].groupby("cancer_type").size()
    assert per_class.nunique() == 1 and per_class.iloc[0] == 64


def test_assert_disjoint_rejects_shared_patient() -> None:
    """A patient in both reference and query sets fails before any distance is computed."""
    assert_disjoint(np.array(["a", "b"]), np.array(["c"]))
    with pytest.raises(RuntimeError, match="b"):
        assert_disjoint(np.array(["a", "b"]), np.array(["b", "c"]))


def test_knn_agreement_counts_same_class_neighbours() -> None:
    """Query at the class-0 cluster sees only class-0 neighbours for small k, mixed for large k."""
    ref = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    ref_y = np.array([0, 0, 1, 1])
    query = np.array([[1.0, 0.05]])
    for metric in ("cosine", "euclidean"):
        a = knn_agreement(ref, ref_y, query, np.array([0]), (1, 2, 4), metric)
        np.testing.assert_allclose(a, [[1.0, 1.0, 0.5]])


def test_cosine_agreement_ignores_vector_length() -> None:
    """Cosine geometry is scale-free; raw Euclidean is not."""
    ref = np.array([[10.0, 0.0], [0.0, 1.0]])
    ref_y = np.array([0, 1])
    query = np.array([[1.0, 0.2]])
    assert knn_agreement(ref, ref_y, query, np.array([0]), (1,), "cosine")[0, 0] == 1.0
    assert knn_agreement(ref, ref_y, query, np.array([0]), (1,), "euclidean")[0, 0] == 0.0


def test_chance_adjusted_maps_chance_to_zero_and_perfect_to_one() -> None:
    """(a - 1/K) / (1 - 1/K)."""
    np.testing.assert_allclose(chance_adjusted(np.array([0.25, 1.0]), 4), [0.0, 1.0])


def test_variance_shares_split_within_and_between_patients() -> None:
    """Patients with identical means have zero between-patient share."""
    x = np.array([[0.0], [2.0], [0.0], [2.0], [5.0], [5.0], [7.0], [7.0]])
    patients = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    shares = variance_shares(x, patients, labels, 2)
    np.testing.assert_allclose(shares, [1.0, 0.0])  # within-patient share per class


# --- gap.embedding: historical feature paths are resolved as stored, rows are bounds-checked ----

from gap.embedding import take_rows  # noqa: E402


def test_take_rows_gathers_valid_rows() -> None:
    """Rows come back in the requested order."""
    tensor = np.arange(12.0).reshape(4, 3)
    np.testing.assert_array_equal(take_rows(tensor, np.array([2, 0]), "x.pt"), tensor[[2, 0]])


@pytest.mark.parametrize("bad", [4, -1])
def test_take_rows_rejects_out_of_bounds_and_negative_indices(bad: int) -> None:
    """A negative index would silently wrap in NumPy/torch; both bounds fail loudly instead."""
    with pytest.raises(RuntimeError, match="x.pt"):
        take_rows(np.zeros((4, 3)), np.array([0, bad]), "x.pt")


def test_take_rows_rejects_non_finite_features() -> None:
    """A corrupted feature row is not passed on to distance computations."""
    tensor = np.zeros((2, 3))
    tensor[1, 0] = np.nan
    with pytest.raises(RuntimeError, match="finite"):
        take_rows(tensor, np.array([1]), "x.pt")


# --- weighted margin CDF ------------------------------------------------------------------------

from gap.metrics import weighted_cdf, weighted_median  # noqa: E402


def test_weighted_cdf_uses_weights_not_patch_counts() -> None:
    """One heavy patch outweighs three light ones; total mass is one."""
    values = np.array([-1.0, 1.0, 1.0, 1.0])
    weights = np.array([0.7, 0.1, 0.1, 0.1])
    np.testing.assert_allclose(weighted_cdf(values, weights, np.array([-2.0, 0.0, 2.0])),
                               [0.0, 0.7, 1.0])


def test_weighted_median_follows_weights() -> None:
    """The heavy negative patch carries the median below zero."""
    assert weighted_median(np.array([-1.0, 1.0, 1.0, 1.0]), np.array([0.7, 0.1, 0.1, 0.1])) == -1.0


def test_positive_margin_mass_equals_ba_without_ties() -> None:
    """With no ties, weighted mass of positive margins is the patient-macro BA."""
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(300, 4))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = rng.integers(0, 4, 300)
    patients = rng.integers(0, 20, 300).astype(str)
    w = patient_weights(labels, patients)
    m, ties = margins(probs, labels)
    ba = np.diag(confusion(labels, probs.argmax(axis=1), w, 4)).mean()
    assert not ties.any()
    assert (w[m > 0].sum() / 4) == pytest.approx(ba)


# --- gap.images: selected tiles copied with their provenance -----------------------------------

from gap.images import copy_tiles  # noqa: E402


def test_copy_tiles_flattens_ids_and_records_checksums(tmp_path) -> None:
    """TCGA-UT patch ids contain '/', so file names are flattened; each copy carries its sha256."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jpg").write_bytes(b"one")
    rows = pd.DataFrame({"patch_id": ["SLIDE-1/0_0_5", "SLIDE-1/0_0_5"],
                         "image_path": [str(src / "a.jpg")] * 2})
    out = copy_tiles(rows, tmp_path / "out")
    assert len(out) == 1  # duplicate patch selected twice is copied once
    assert out.loc[0, "file"] == "SLIDE-1__0_0_5.jpg"
    assert (tmp_path / "out" / "SLIDE-1__0_0_5.jpg").read_bytes() == b"one"
    assert out.loc[0, "sha256"] == "7692c3ad3540bb803c020b3aee66cd8887123234ea0c6e7143c0add73ff431ed"


# --- final-review fixes: margin-conditioned flips, required r1 temperature ----------------------

from gap.metrics import lost_by_margin  # noqa: E402
from gap.records import read_temperature  # noqa: E402


def test_lost_by_margin_splits_correct_mass_by_reference_margin() -> None:
    """Correct mass and lost mass per reference-margin bin; bins cover only correct rows."""
    margin = np.array([0.2, 0.8, 1.5, 3.0, -0.5])
    reference_ok = margin > 0
    changed_ok = np.array([False, True, False, True, True])
    weights = np.full(5, 0.5)
    edges = (0.0, 1.0, np.inf)
    correct, lost = lost_by_margin(margin, reference_ok, changed_ok, weights, edges, 2)
    np.testing.assert_allclose(correct, [0.5, 0.5])  # (0.5 + 0.5) / 2 classes per bin
    np.testing.assert_allclose(lost, [0.25, 0.25])


def test_read_temperature_requires_file_when_asked(tmp_path) -> None:
    """A missing r1 temperature fails instead of turning scaled margins into NaN."""
    (tmp_path / "temperature.json").write_text('{"temperature": 0.5}')
    assert read_temperature(tmp_path, required=True) == 0.5
    with pytest.raises(RuntimeError, match="temperature"):
        read_temperature(tmp_path / "missing", required=True)
    assert np.isnan(read_temperature(tmp_path / "missing", required=False))
