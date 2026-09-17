"""Pure between-patient eigenbasis estimated from one drawn cohort's own patients."""

from __future__ import annotations

import numpy as np

from centre.arms import between_patient_eigenbasis
from centre.cohort import TrainingTable

__all__ = ["cohort_eigenbasis"]


def cohort_eigenbasis(
    table: TrainingTable, n_classes: int, g: int
) -> tuple[np.ndarray, np.ndarray]:
    """(k, d) eigenvectors and (k,) eigenvalues of the cohort's own between-patient deviations.

    Rows of ``table.x`` are ordered class -> patient -> m patches, so patient means are
    ``table.x.reshape(n_classes, g, m, d).mean(axis=2)``. Deviations are taken from each
    class's own cohort mean, pooled across classes, with dof = n_classes * (g - 1).
    """
    m = len(table.x) // (n_classes * g)
    means = table.x.reshape(n_classes, g, m, -1).mean(axis=2)
    dev = (means - means.mean(axis=1, keepdims=True)).reshape(n_classes * g, -1)
    return between_patient_eigenbasis(dev, n_classes * (g - 1))
