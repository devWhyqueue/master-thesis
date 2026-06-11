from collections.abc import Iterator

import numpy as np
from torch.utils.data import Sampler


class BatchBalancingSampler(Sampler[int]):
    def __init__(
        self,
        labels: np.ndarray,
        n_classes: int,
        seed: int = 0,
        sampler_power: float = 1.0,
    ) -> None:
        self.labels = labels
        self.n_classes = n_classes
        self.seed = seed
        self.class_indices = [
            np.where(self.labels == class_index)[0]
            for class_index in range(self.n_classes)
        ]
        self.generator = np.random.default_rng(seed)
        self._len = len(labels)
        counts = np.bincount(labels, minlength=n_classes)
        class_weights = np.power(1.0 / np.maximum(counts, 1), sampler_power)
        self.p = class_weights / class_weights.sum()

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[int]:
        return self._sample_indices()

    def _sample_indices(self) -> Iterator[int]:
        for _ in range(self._len):
            class_index = self.generator.choice(self.n_classes, p=self.p)
            yield int(self.generator.choice(self.class_indices[class_index]))
