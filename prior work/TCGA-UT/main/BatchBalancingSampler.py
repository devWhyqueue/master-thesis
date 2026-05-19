from torch.utils.data import Sampler
import numpy as np
from typing import Iterator

class BatchBalancingSampler(Sampler[int]):
    def __init__(self, labels, n_classes:int, seed:int=0) -> None:
        self.labels = labels
        self.n_classes = n_classes
        self.seed = seed
        self.class_indices = [
            np.where(self.labels == c)[0]
            for c in range(self.n_classes)
        ]
        self.generator = np.random.default_rng(seed)
        self._len = len(labels)
        self.p = np.ones(self.n_classes)/self.n_classes
    def __len__(self) -> int:
        return self._len
    def __iter__(self) -> Iterator[int]:
        for _ in range(self._len):
            cls = self.generator.choice(self.n_classes, p = self.p)
            idx = self.generator.choice(self.class_indices[cls])
            yield idx