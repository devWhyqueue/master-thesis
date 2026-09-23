"""Statistics for exp-33's gate G, the primary gap H1, and the onset comparison H2.

Split out of ``margin.analyze`` (which keeps orchestration, H3, and output writing) purely to
stay under this codebase's per-file length limit; the two modules are one analysis stage.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from imbalance_benchmark.common import output_root

from breadth import BOOTSTRAP_SEED
from breadth.analyze.secondary import pack_estimate

from decomposition.model import draw_weights

from centre import N_DRAWS, N_SPLITS
from centre.analyze import arm_accuracy, pooled

from spectrum import baseline_config

from margin import (
    AGREEMENT_TOLERANCE_PP,
    GATE_P100_CI,
    H1_SEED,
    N_H1_DRAWS,
    RATIOS_NEW,
    Q_ARMS,
    QT_ARMS,
)

__all__ = [
    "Gate",
    "H2Result",
    "Peer",
    "pool",
    "build_dists",
    "gate_check",
    "h2_curve",
    "peer_data",
    "h1_gap",
    "label",
]

logger = logging.getLogger(__name__)

Gate = tuple[float, float, float, bool]
H2Result = tuple[dict[str, Any], int | None, int | None]
Peer = tuple[dict[str, Any], dict[str, Any], np.ndarray] | None


def pool(
    config: dict[str, Any], ds_config: dict[str, Any], names: list[str]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Pooled r1/Q/QT arm accuracy and balanced accuracy, with the draw-resampling weights."""
    acc = {
        **arm_accuracy(ds_config, names, arms=("r1",)),
        **arm_accuracy(config, names, arms=Q_ARMS + QT_ARMS),
    }
    n_replicates = next(iter(acc.values())).shape[-1]
    fit_split = np.repeat(np.arange(N_SPLITS), N_DRAWS)
    w = draw_weights(
        fit_split, N_DRAWS, n_replicates, np.random.default_rng(BOOTSTRAP_SEED)
    )
    ba = {arm: pooled(a, w) for arm, a in acc.items()}
    return acc, fit_split, w, ba


def _observed_p(
    config: dict[str, Any], names: list[str], w: np.ndarray
) -> dict[str, np.ndarray]:
    """Observed prior-channel damage P(rho) = ba(r1) - ba(P{rho}), from exp-27/28's own outputs."""
    ds_config = baseline_config(config, "prevalence_outputs")
    p_config = baseline_config(config, "p_outputs")
    acc = {
        **arm_accuracy(ds_config, names, arms=("r1",)),
        **arm_accuracy(p_config, names, arms=tuple(f"P{r}" for r in RATIOS_NEW)),
    }
    ba = {arm: pooled(a, w) for arm, a in acc.items()}
    return {f"observed_P_{r}": ba["r1"] - ba[f"P{r}"] for r in RATIOS_NEW}


def build_dists(
    config: dict[str, Any], names: list[str], ba: dict[str, np.ndarray], w: np.ndarray
) -> dict[str, np.ndarray]:
    """Arm-prefixed BA plus D_sim(rho), D_sim_ts(rho), and the observed P(rho) contrast."""
    dists: dict[str, np.ndarray] = {f"arm_{arm}": d for arm, d in ba.items()}
    for r in RATIOS_NEW:
        dists[f"D_sim_{r}"] = ba["r1"] - ba[f"Q{r}"]
        dists[f"D_sim_ts_{r}"] = ba["r1"] - ba[f"QT{r}"]
    dists |= _observed_p(config, names, w)
    return dists


def gate_check(config: dict[str, Any], dists: dict[str, np.ndarray]) -> Gate:
    """Whether D_sim(100)'s point falls inside the pre-registered observed-P100 interval."""
    gate_lo, gate_hi = GATE_P100_CI[config["dataset"]["name"]]
    d_sim_100 = float(dists["D_sim_100"][0])
    gate_pass = gate_lo <= d_sim_100 <= gate_hi
    logger.info(
        "Gate: D_sim(100)=%.3f, observed P100 CI=[%.3f, %.3f], pass=%s",
        d_sim_100,
        gate_lo,
        gate_hi,
        gate_pass,
    )
    return gate_lo, gate_hi, d_sim_100, gate_pass


def _onset(dists: dict[str, np.ndarray], prefix: str) -> int | None:
    """First rho (in RATIOS_NEW order) whose damage lower CI clears 0."""
    for r in RATIOS_NEW:
        if pack_estimate(dists[f"{prefix}_{r}"])["ci_2_5"] > 0:
            return r
    return None


def h2_curve(dists: dict[str, np.ndarray]) -> H2Result:
    """Per-rho D_sim/observed_P agreement, plus each curve's onset rho."""
    h2 = {
        str(r): {
            "d_sim": pack_estimate(dists[f"D_sim_{r}"]),
            "observed_P": pack_estimate(dists[f"observed_P_{r}"]),
            "agree_within_1pp": bool(
                abs(dists[f"D_sim_{r}"][0] - dists[f"observed_P_{r}"][0])
                <= AGREEMENT_TOLERANCE_PP
            ),
        }
        for r in RATIOS_NEW
    }
    return h2, _onset(dists, "D_sim"), _onset(dists, "observed_P")


def peer_data(config: dict[str, Any]) -> Peer:
    """The other dataset's own analysis.json, diagnostics.json, and D_sim(100) distribution."""
    peer_key = config.get("slurm", {}).get("peer_outputs")
    if not peer_key:
        return None
    peer_config = baseline_config(config, "peer_outputs")
    data_dir = output_root(peer_config) / "data"
    analysis_path, npz_path, diag_path = (
        data_dir / "analysis.json",
        data_dir / "distributions.npz",
        data_dir / "diagnostics.json",
    )
    if not (analysis_path.exists() and npz_path.exists() and diag_path.exists()):
        logger.info(
            "Peer outputs not ready yet at %s; skipping H1/H3 overlay.", data_dir
        )
        return None
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(diag_path.read_text(encoding="utf-8"))
    with np.load(npz_path) as npz:
        peer_d_sim_100 = np.asarray(npz["D_sim_100"])
    return analysis, diagnostics, peer_d_sim_100


def _independent_diff(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Difference distribution of two independent bootstrap distributions (not paired by index).

    Column 0 is the observed difference; the rest independently resample each side's own
    replicates (columns 1..), so the combined CI does not inherit any shared-seed correlation
    between the two datasets' own bootstraps.
    """
    out = np.empty(N_H1_DRAWS, dtype=np.float64)
    out[0] = a[0] - b[0]
    out[1:] = (
        a[rng.integers(1, len(a), size=N_H1_DRAWS - 1)]
        - b[rng.integers(1, len(b), size=N_H1_DRAWS - 1)]
    )
    return out


def h1_gap(
    config: dict[str, Any], dists: dict[str, np.ndarray], peer: Peer
) -> dict[str, Any] | None:
    """Independent-bootstrap BRACS-minus-TCGA-UT gap, once the peer dataset has finished."""
    if peer is None:
        return None
    _, _, peer_d_sim_100 = peer
    rng = np.random.default_rng(H1_SEED)
    bracs_dist, tcga_dist = (
        (dists["D_sim_100"], peer_d_sim_100)
        if config["dataset"]["name"] == "bracs"
        else (peer_d_sim_100, dists["D_sim_100"])
    )
    dists["gap_h1_D_sim_100"] = _independent_diff(bracs_dist, tcga_dist, rng)
    h1 = pack_estimate(dists["gap_h1_D_sim_100"])
    h1["supports_prior_channel_gap"] = bool(abs(h1["point"] - 3.27) <= 1.0)
    return h1


def label(gate_pass: bool, h1: dict[str, Any] | None) -> str:
    """The pre-registered outcome label."""
    if not gate_pass:
        return "proxy_invalid"
    if h1 is not None and h1["supports_prior_channel_gap"]:
        return "margin_explains_prior_gap"
    return "partial"
