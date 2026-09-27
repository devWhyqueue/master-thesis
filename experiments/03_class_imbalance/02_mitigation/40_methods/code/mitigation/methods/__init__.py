"""Sampling, losses, and heads for the mitigation methods (``report/40_methods.tex``).

Each function is a direct, unit-tested translation of one report equation; the
control parameter named in the docstring is the one swept across runs (fit.py
passes it as ``--param``), everything else is fixed by the method's definition.
"""

from __future__ import annotations

from mitigation.methods.heads import CosineHead, DisAlign, reinit_linear_head
from mitigation.methods.losses import (
    disalign_class_weights,
    gcl_delta,
    gcl_logits,
    gcl_loss,
    la_loss,
)
from mitigation.methods.mixup import mixup_batch, mixup_loss
from mitigation.methods.sampling import draw_indices, sample_weights

__all__ = [
    "CosineHead",
    "DisAlign",
    "reinit_linear_head",
    "disalign_class_weights",
    "gcl_delta",
    "gcl_logits",
    "gcl_loss",
    "la_loss",
    "mixup_batch",
    "mixup_loss",
    "draw_indices",
    "sample_weights",
]
