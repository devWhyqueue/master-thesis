"""Centre-uncertainty logistic loss (ISDA upper bound) with analytic gradient, fitted by L-BFGS-B.

For logits ``z_ik = w_k . x_i + b_k`` the loss is the mean of
``log sum_k exp[z_ik - z_iy + (t / 2) (w_k - w_y)' V (w_k - w_y)]`` plus ``lam / 2 ||W||_F^2``,
with ``V`` a low-rank patient covariance (``basis' diag(variances) basis``) plus an optional
isotropic part ``isotropic * I``. Per-class-pair quadratic forms are evaluated through the
basis, so no d x d matrix is ever formed. Inference uses ordinary logits.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from decodability.linear import LinearFitResult
from scipy.optimize import minimize
from scipy.special import logsumexp

__all__ = [
    "Covariance",
    "UncertainFit",
    "as_linear_result",
    "covariance_for",
    "fit_uncertain_logistic",
    "loss_and_grad",
    "pair_quadratics",
]

_MAX_LINE_SEARCH = 50
_FTOL = 64 * np.finfo(float).eps


class Covariance(NamedTuple):
    """Centre covariance ``basis' diag(variances) basis + isotropic * I`` (basis rows orthonormal)."""

    basis: np.ndarray
    variances: np.ndarray
    isotropic: float

    @property
    def is_zero(self) -> bool:
        """True when the covariance contributes nothing to the loss."""
        return not self.isotropic and not np.any(self.variances)


class UncertainFit(NamedTuple):
    """Fitted coefficients (K, d), intercepts (K,), and solver diagnostics."""

    coef: np.ndarray
    intercept: np.ndarray
    objective: float
    grad_norm: float
    n_iter: int
    converged: bool


def covariance_for(
    kind: str, basis: np.ndarray, eigvals: np.ndarray, g: int
) -> Covariance:
    """Centre covariance ``Sigma / g`` of ``g`` patients; ``isotropic`` keeps only its trace, spread evenly."""
    if kind not in ("patient", "isotropic"):
        raise ValueError(f"Unknown covariance kind {kind!r}")
    if g < 1:
        raise ValueError("Patient count must be at least 1")
    if basis.ndim != 2 or len(basis) != len(eigvals) or np.any(eigvals < 0):
        raise ValueError(
            "Basis and eigenvalues must be aligned, eigenvalues non-negative"
        )
    variances = eigvals / g
    if kind == "patient":
        return Covariance(basis, variances, 0.0)
    return Covariance(basis[:0], variances[:0], float(variances.sum()) / basis.shape[1])


def _pair_diff(m: np.ndarray) -> np.ndarray:
    """(K, K, r) differences ``m_k - m_c``."""
    return m[:, None, :] - m[None, :, :]


def pair_quadratics(w: np.ndarray, cov: Covariance) -> np.ndarray:
    """(K, K) matrix of ``(w_k - w_c)' V (w_k - w_c)``."""
    p = _pair_diff(w @ cov.basis.T)
    q = np.einsum("kcr,r->kc", p * p, cov.variances)
    if cov.isotropic:
        d = _pair_diff(w)
        q = q + cov.isotropic * (d * d).sum(axis=2)
    return q


def _quadratic_gradient(w: np.ndarray, cov: Covariance, a: np.ndarray) -> np.ndarray:
    """Gradient in ``w`` of ``sum_kc a[k, c] (w_k - w_c)' V (w_k - w_c)``."""
    s = a + a.T
    rows = s.sum(axis=1)[:, None]
    m = w @ cov.basis.T
    grad = (2.0 * cov.variances * (rows * m - s @ m)) @ cov.basis
    if cov.isotropic:
        grad = grad + 2.0 * cov.isotropic * (rows * w - s @ w)
    return grad


def loss_and_grad(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    lam: float,
    t: float,
    cov: Covariance,
) -> tuple[float, np.ndarray]:
    """Objective and gradient at ``theta = [W.ravel(), b]``."""
    n, d = x.shape
    w = theta[: n_classes * d].reshape(n_classes, d)
    b = theta[n_classes * d :]
    z = x @ w.T + b
    s = z - z[np.arange(n), y][:, None]
    if t:
        s = s + (t / 2.0) * pair_quadratics(w, cov)[:, y].T
    lse = np.asarray(logsumexp(s, axis=1))
    value = float(lse.mean() + lam / 2.0 * np.sum(w * w))
    p = np.exp(s - lse[:, None])
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1.0
    dz = (p - onehot) / n
    grad_w = dz.T @ x + lam * w
    if t:
        grad_w = grad_w + _quadratic_gradient(w, cov, (t / 2.0) * (p.T @ onehot) / n)
    return value, np.concatenate([grad_w.ravel(), dz.sum(axis=0)])


def _check_inputs(
    x: np.ndarray, y: np.ndarray, n_classes: int, lam: float, t: float, cov: Covariance
) -> None:
    """Raise ``ValueError`` on inputs the loss is undefined for."""
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("x must be (n, d) and y (n,)")
    if not np.all(np.isfinite(x)):
        raise ValueError("Features must be finite")
    if y.min() < 0 or y.max() >= n_classes:
        raise ValueError("Labels must lie in [0, n_classes)")
    if not lam > 0 or not np.isfinite(lam):
        raise ValueError("lam must be positive and finite")
    if t < 0 or not np.isfinite(t):
        raise ValueError("t must be non-negative and finite")
    if cov.basis.shape[1:] != x.shape[1:] and len(cov.basis):
        raise ValueError("Covariance basis dimension differs from feature dimension")
    if np.any(cov.variances < 0) or cov.isotropic < 0:
        raise ValueError("Covariance must be positive semi-definite")


def _minimize(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    lam: float,
    t: float,
    cov: Covariance,
    limits: tuple[float, int],
) -> Any:
    """L-BFGS-B from zero with sklearn's line-search and ftol settings."""
    options = {
        "maxiter": limits[1],
        "maxls": _MAX_LINE_SEARCH,
        "gtol": limits[0],
        "ftol": _FTOL,
    }
    return minimize(
        loss_and_grad,
        np.zeros(n_classes * (x.shape[1] + 1)),
        args=(x, y, n_classes, lam, t, cov),
        jac=True,
        method="L-BFGS-B",
        options=options,
    )


def fit_uncertain_logistic(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    lam: float,
    t: float,
    cov: Covariance,
    limits: tuple[float, int],
) -> UncertainFit:
    """L-BFGS-B fit in float64; ``limits`` = (gradient tolerance on the mean objective, max iterations)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.int64)
    _check_inputs(x, y, n_classes, lam, t, cov)
    res = _minimize(x, y, n_classes, lam, t, cov, limits)
    split = n_classes * x.shape[1]
    return UncertainFit(
        res.x[:split].reshape(n_classes, -1),
        res.x[split:],
        float(res.fun),
        float(np.abs(res.jac).max()),
        int(res.nit),
        bool(res.success),
    )


def as_linear_result(
    fit: UncertainFit, lam: float, n: int, tol: float, max_iter: int
) -> LinearFitResult:
    """Wrap a fit as the ``LinearFitResult`` the shared run-record builder expects."""
    return LinearFitResult(
        fit.coef,
        fit.intercept,
        lam,
        1.0 / (n * lam),
        "lbfgs-b-centre-uncertainty",
        "float64",
        tol,
        tol,
        max_iter,
        fit.n_iter,
        fit.objective,
        fit.converged,
        False,
    )
