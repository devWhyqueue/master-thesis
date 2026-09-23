"""Dense same-cohort centre-correction target, and its composition with the separation shift.

PLAN.md's "dense same-cohort target": the SAME patients already drawn into a shard's cohort, each
measured at ``CENTRE_DEPTH`` = 160 patches instead of the arm's own shallow training depth -- extra
per-patient precision on the real cohort, not a different or larger patient pool (unlike exp-16's
``centre.pool.centres``, which pools every eligible training patient). This is finite-depth
first-moment error reduction only; it supplies no new patients and no covariance information.

Both the separation shift (``separation.geometry.apply_intervention``) and the centre-correction
shift (``centre.arms.move_centres``) are per-class constant translations, so they compose by
addition -- but ``move_centres`` recomputes the *current* class mean from its input each call, so
composing them correctly (for the ``joint`` setting) requires shifting the dense target by the same
alpha-scaled offset applied to the training rows first (:func:`shift_target`), not literal vector
addition after the fact.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from breadth.sampling import load_features_for_df

from centre.arms import class_means, move_centres
from centre.cohort import patient_rows

from separation.geometry import Centres

__all__ = [
    "dense_cohort_target",
    "shift_target",
    "move_to_target",
    "move_to_target_negated",
]


def dense_cohort_target(
    train_df: pd.DataFrame, names: list[str], patients: list[list[str]], depth: int
) -> np.ndarray:
    """(C, d) native-space mean-of-patient-means, over exactly this shard's own drawn patients."""
    means = []
    for name, class_patients in zip(names, patients):
        class_df = cast(pd.DataFrame, train_df[train_df["cancer_type"] == name])
        rows = patient_rows(class_df, class_patients, depth)
        x = load_features_for_df(train_df.loc[rows]).astype(np.float64)
        patient_means = x.reshape(len(class_patients), depth, -1).mean(axis=1)
        means.append(patient_means.mean(axis=0))
    return np.stack(means)


def shift_target(target: np.ndarray, centres: Centres, alpha: float) -> np.ndarray:
    """Apply the same x^(alpha) per-class constant shift used on training rows to a native target."""
    if alpha == 1.0:
        return target
    return target + (alpha - 1.0) * (centres.means - centres.grand_mean)


def move_to_target(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Translate every class to ``target``, preserving within-class residuals (PLAN.md line 33)."""
    return move_centres(x, y, target)


def move_to_target_negated(
    x: np.ndarray, y: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Wrong-direction control: same correction magnitude, opposite direction (PLAN.md line 40)."""
    means = class_means(x, y, len(target))
    return move_centres(x, y, 2.0 * means - target)
