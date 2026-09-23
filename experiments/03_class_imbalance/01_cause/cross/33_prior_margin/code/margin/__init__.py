"""Constants for the prior-injection margin experiment (exp-33).

For every stored r1 fit (balanced training, exp-25/26), inject the prior of the paired rho{rho}
fit post hoc: preds = argmax(log p_r1 + log(n_c / sum n)), n_c from rho{rho}'s realized class
counts. This is exp-27/28's ``Lr`` adjustment with the sign flipped and the source fixed at r1
(rather than at r{rho} itself), so it estimates the *prior channel's* damage D_sim(rho) with no
new fits: margins, direction, headroom, K, and the onset nonlinearity all come through the actual
stored softmax, unlike a linear confusion score.

r1/r{rho} are reused from exp-25 (TCGA-UT) / exp-26 (BRACS) via ``spectrum.baseline_config``
against ``slurm.prevalence_outputs``. The observed prior-channel damage P{rho} (exp-27/28,
``slurm.p_outputs``) sets the pre-registered validity gate.
"""

from __future__ import annotations

from prevalence import RATIOS

__all__ = [
    "RATIOS_NEW",
    "Q_ARMS",
    "QT_ARMS",
    "ARMS",
    "N_SUBSETS",
    "SUBSET_K",
    "SUBSET_SEED",
    "H1_SEED",
    "N_H1_DRAWS",
    "MARGIN_RHOS",
    "GATE_P100_CI",
    "AGREEMENT_TOLERANCE_PP",
]

RATIOS_NEW: tuple[int, ...] = RATIOS[1:]  # (2, 5, 10, 20, 50, 100)

# Raw-probability injection (primary) and its validation-TS sensitivity variant.
Q_ARMS: tuple[str, ...] = tuple(f"Q{r}" for r in RATIOS_NEW)
QT_ARMS: tuple[str, ...] = tuple(f"QT{r}" for r in RATIOS_NEW)
ARMS: tuple[str, ...] = Q_ARMS + QT_ARMS

# H3 (K control): TCGA-UT-only 7-class subset screen.
N_SUBSETS: int = 500
SUBSET_K: int = 7
SUBSET_SEED: int = 20260923
# H1: independent-bootstrap combination of the two datasets' D_sim(100) distributions.
H1_SEED: int = 20260923
N_H1_DRAWS: int = 2000
# Intuition figure: mark ln(rho) for these.
MARGIN_RHOS: tuple[int, ...] = (2, 10, 100)

# Gate G (pre-registered, fixed before any run): the observed prior-channel damage P(rho=100),
# 95% CI, from exp-27 (TCGA-UT) / exp-28 (BRACS)'s own stored analysis.json (``delta_P_100``,
# sign-flipped to the positive "damage" convention this experiment uses).
GATE_P100_CI: dict[str, tuple[float, float]] = {
    "tcga_ut": (0.8736685460827334, 1.5250203337247672),
    "bracs": (2.8448732856323082, 6.301268881728336),
}
AGREEMENT_TOLERANCE_PP: float = 1.0
