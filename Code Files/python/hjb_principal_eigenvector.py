"""Principal eigenvector of the autonomous matrix M."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig, eigh_tridiagonal


@dataclass
class HJBParams:

    gamma: float = 2.0e-5   # CARA risk aversion
    sigma: float = 8.0      # realised vol, USD * s^-1/2
    A: float = 0.8          # AS base intensity, fills/s
    k: float = 0.15         # AS decay, per USD
    alpha_ml: float = 0.0   # ML drift, USD/s (0 = baseline)
    Q: int = 10             # inventory capacity
    xi_b: float = 0.0       # bid-fill adverse move, USD
    xi_a: float = 0.0       # ask-fill adverse move, USD

    @property
    def alpha(self) -> float:
        return 0.5 * self.k * self.gamma * self.sigma ** 2

    @property
    def beta_ml(self) -> float:
        return self.k * self.alpha_ml

    @property
    def eta(self) -> float:
        return self.A * (1.0 + self.gamma / self.k) ** (-(1.0 + self.k / self.gamma))


def q_grid(Q: int) -> np.ndarray:
    """Inventory levels ordered as the state vector: q = (Q, Q-1, ..., -Q)."""
    return np.arange(Q, -Q - 1, -1, dtype=np.float64)


def autonomous_diag(p: HJBParams) -> np.ndarray:
    """Diagonal D_q = alpha*q^2 - beta_ML*q of the autonomous M, ordered Q..-Q."""
    q = q_grid(p.Q)
    return p.alpha * q ** 2 - p.beta_ml * q


def build_autonomous_M(p: HJBParams) -> tuple[np.ndarray, np.ndarray]:
    """Return (diag, off) of the symmetric tridiagonal autonomous M (Phi=0).
    `off` has length 2Q (the shared sub/super-diagonal), all equal to -eta."""
    diag = autonomous_diag(p)
    off = np.full(diag.size - 1, -p.eta, dtype=np.float64)
    return diag, off


def as_offdiag_factors(p: HJBParams) -> tuple[np.ndarray, np.ndarray]:
    """Per-row fill-conditional AS reweights on the q-ordered (Q..-Q) off-diagonals
    For row index i (inventory q = Q - i):
        M[i, i-1] = -eta * bid[i]   couples q -> q+1 (bid),  factor exp(-k(q+1)xi_b)
        M[i, i+1] = -eta * ask[i]   couples q -> q-1 (ask),  factor exp(+k(q-1)xi_a)
    Both arrays reduce to ones when xi_b = xi_a = 0."""
    q = q_grid(p.Q)
    bid = np.exp(-p.k * (q + 1.0) * p.xi_b)
    ask = np.exp(+p.k * (q - 1.0) * p.xi_a)
    return bid, ask


def build_modified_M_dense(p: HJBParams) -> np.ndarray:
    """Dense autonomous M with the AS off-diagonal reweighting, ordered
    q = Q..-Q. Non-symmetric for xi != 0; reduces to build_autonomous_M at xi = 0."""
    diag = autonomous_diag(p)
    n = diag.size
    M = np.diag(diag).astype(float)
    bid, ask = as_offdiag_factors(p)
    i = np.arange(1, n)
    M[i, i - 1] = -p.eta * bid[1:]        # super-diagonal in q (bid, q -> q+1)
    j = np.arange(0, n - 1)
    M[j, j + 1] = -p.eta * ask[:-1]       # sub-diagonal in q (ask, q -> q-1)
    return M


def principal_eigvec(p: HJBParams, *, check: bool = True) -> np.ndarray:
    """Exact discrete principal eigenvector f^0 (ordered q = Q..-Q), normalised so max == 1."""
    if p.xi_b == 0.0 and p.xi_a == 0.0:
        diag, off = build_autonomous_M(p)
        _, v = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))
        f0 = v[:, 0]
    else:
        M = build_modified_M_dense(p)
        w, V = eig(M)
        f0 = V[:, int(np.argmin(w.real))].real
    if f0.sum() < 0:
        f0 = -f0
    if check and np.any(f0 < -1e-10):
        print(
            "WARN: f^0 not strictly positive (min={:.3e}); eigensolver sanity "
            "failure, check parameter regime.".format(f0.min())
        )
    return f0 / f0.max()


def gaussian_eigvec(p: HJBParams) -> np.ndarray:
    """Continuous Gaussian approximation of f^0:
        f^0_q ~ exp(-1/2 sqrt(alpha/eta) (q - alpha_ml/(gamma sigma^2))^2).
    Normalised so max == 1. For Regime-I closed form and the discrete-vs-Gaussian
    error check ONLY, never for stitching."""
    q = q_grid(p.Q)
    centre = p.alpha_ml / (p.gamma * p.sigma ** 2)
    g = np.exp(-0.5 * math.sqrt(p.alpha / p.eta) * (q - centre) ** 2)
    return g / g.max()


def discrete_autonomous_quotes(p: HJBParams, f0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if f0 is None:
        f0 = principal_eigvec(p)
    Q = p.Q
    const = (1.0 / p.gamma) * math.log1p(p.gamma / p.k)

    def omega(q: int) -> float:
        return f0[Q - q]  # f0 ordered Q..-Q

    delta_b = np.full(2 * Q + 1, np.nan)
    delta_a = np.full(2 * Q + 1, np.nan)
    for qq in range(-Q, Q + 1):
        i = qq + Q
        if qq != Q:
            delta_b[i] = (const + (1.0 / p.k) * math.log(omega(qq) / omega(qq + 1))
                          + (qq + 1) * p.xi_b)
        if qq != -Q:
            delta_a[i] = (const + (1.0 / p.k) * math.log(omega(qq) / omega(qq - 1))
                          - (qq - 1) * p.xi_a)
    return delta_b, delta_a


