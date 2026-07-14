from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = ["fit_rq3_model"]


def _optimize_rq3(
    beta_0: torch.Tensor,
    beta: torch.Tensor,
    u: torch.Tensor,
    l_su: torch.Tensor,
    l_s: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    s: torch.Tensor,
    idxs: torch.Tensor,
    is_logistic: bool,
) -> None:
    """Helper to run Adam optimization on Bayesian MAP objective."""
    opt = torch.optim.Adam([beta_0, beta, u, l_su, l_s], lr=0.01)
    for _ in range(1000):
        opt.zero_grad()
        mu = beta_0 + torch.matmul(x, beta) + u[idxs]
        if is_logistic:
            prob = torch.sigmoid(mu)
            loss_like = -torch.sum(
                y * torch.log(prob + 1e-8) + (1.0 - y) * torch.log(1.0 - prob + 1e-8)
            )
        else:
            tot = s.square() + torch.exp(l_s).square()
            loss_like = 0.5 * torch.sum((y - mu).square() / tot + torch.log(tot))
        loss = (
            loss_like
            + 0.5
            * torch.sum(
                u.square() / torch.exp(l_su).square()
                + torch.log(torch.exp(l_su).square())
            )
            + 0.5 * torch.sum(beta.square())
        )
        loss.backward()
        opt.step()


def _standardize(x: np.ndarray) -> torch.Tensor:
    """Zero-center and unit-scale each predictor column."""
    return torch.tensor(
        (x - x.mean(0)) / np.maximum(x.astype(float).std(0), 1e-8), dtype=torch.float32
    )


def _fit_map(
    x_std: torch.Tensor,
    y: np.ndarray,
    s_errors: np.ndarray,
    groups: np.ndarray,
    g: dict[str, int],
    is_logistic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b0, b, u = [torch.zeros(s, requires_grad=True) for s in (1, x_std.shape[1], len(g))]
    l_su, l_s = [torch.zeros(1, requires_grad=True) for _ in range(2)]
    _optimize_rq3(
        b0,
        b,
        u,
        l_su,
        l_s,
        x_std,
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(s_errors, dtype=torch.float32),
        torch.tensor([g[gp] for gp in groups], dtype=torch.long),
        is_logistic,
    )
    return b0, b, u, l_su, l_s


def fit_rq3_model(
    y: np.ndarray,
    x: np.ndarray,
    groups: np.ndarray,
    s_errors: np.ndarray,
    is_logistic: bool = False,
) -> dict[str, Any]:
    """Fit a Bayesian hierarchical random intercept model using PyTorch autograd (MAP)."""
    g = {gp: i for i, gp in enumerate(np.unique(groups))}
    x_std = _standardize(np.atleast_2d(x.T).T if x.ndim == 1 else x)
    b0, b, u, l_su, l_s = _fit_map(x_std, y, s_errors, groups, g, is_logistic)
    return {
        "intercept": float(b0.item()),
        "slopes": b.detach().numpy().tolist(),
        "rand_intercepts": u.detach().numpy().tolist(),
        "sigma_u": float(torch.exp(l_su).item()),
        "sigma": float(torch.exp(l_s).item()) if not is_logistic else 0.0,
    }
