"""Train-time logit adjustment, GCL's clouded logit, and DisAlign's class weight."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "la_loss",
    "gcl_delta",
    "gcl_logits",
    "gcl_loss",
    "disalign_class_weights",
]


def la_loss(
    logits: torch.Tensor, y: torch.Tensor, log_prior: torch.Tensor, tau: float
) -> torch.Tensor:
    """Train-time logit adjustment: CE(z + tau*log(pi), y) (Eq. la-train); tau=0 is CE."""
    return F.cross_entropy(logits + tau * log_prior, y)


def gcl_delta(counts: np.ndarray) -> np.ndarray:
    """Per-class cloud size delta_j = log(n_max) - log(n_j) (Eq. gcl-logit)."""
    n_max = counts.max()
    return np.log(n_max) - np.log(np.maximum(counts, 1.0))


def gcl_logits(
    cos_theta: torch.Tensor,
    delta: torch.Tensor,
    s: float,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Clouded cosine logits s*(cos(theta_j) - delta_j*|eps|), eps~N(0,sigma^2) (Eq. gcl-logit).

    ``sigma=0`` recovers ``s*cos(theta)`` (CE with a cosine head, no noise); the class
    with ``delta_j=0`` (the most frequent) always receives zero noise regardless of sigma.
    """
    if sigma == 0:
        return s * cos_theta
    eps = (
        torch.randn(cos_theta.shape, generator=generator, device=cos_theta.device)
        * sigma
    )
    return s * (cos_theta - delta.unsqueeze(0) * eps.abs())


def gcl_loss(
    cos_theta: torch.Tensor,
    y: torch.Tensor,
    delta: torch.Tensor,
    s: float,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """CE on GCL's clouded logits (Eq. gcl-logit); no noise is added at inference."""
    return F.cross_entropy(gcl_logits(cos_theta, delta, s, sigma, generator), y)


def disalign_class_weights(counts: np.ndarray, rho: float) -> np.ndarray:
    """Reweighting w_c(rho) = pi_c^-rho / sum_k pi_k^-rho (Eq. disalign-loss).

    rho=0 weights every patch equally; rho=1 gives inverse-prevalence weights.
    Proportional to n_c^-rho (the constant N^rho common to every class cancels
    in the normalization), so it is the same family as balanced sampling's
    class weight, applied here to the loss instead of the sampler.
    """
    pi = counts / counts.sum()
    weight = pi**-rho
    return weight / weight.sum()
