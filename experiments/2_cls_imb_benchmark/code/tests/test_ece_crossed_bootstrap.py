from __future__ import annotations

import numpy as np

from imbalance_benchmark.analysis.reporting.calibration_intervals import (
    crossed_ece_distribution,
)


class _Context:
    def __init__(self, values: list[float]) -> None:
        self.values = np.asarray(values)
        self.probability_inputs: list[np.ndarray] = []

    def ece_distribution(
        self, _labels: np.ndarray, probabilities: np.ndarray
    ) -> np.ndarray:
        self.probability_inputs.append(probabilities)
        return self.values


def test_crossed_ece_averages_seed_resampled_split_replicates() -> None:
    first, second = _Context([0.1, 0.5]), _Context([0.3, 0.7])
    first_probs = np.array([[[0.9, 0.1]]])
    second_probs = np.array([[[0.1, 0.9]]])

    distribution = crossed_ece_distribution(
        [(np.array([0]), first_probs), (np.array([1]), second_probs)],
        [first, second],
    )

    assert distribution == [0.2, 0.6]
    assert first.probability_inputs == [first_probs]
    assert second.probability_inputs == [second_probs]
