"""Epoch-stitching boundary condition."""
from __future__ import annotations

import numpy as np

from hjb_principal_eigenvector import HJBParams, principal_eigvec, q_grid


def liquidation_terminal(p: HJBParams, nu: float) -> np.ndarray:
    """Finite-horizon liquidation terminal omega_q(T) = exp(-k nu |q|), ordered
    q = (Q, ..., -Q). nu >= 0 is the linear liquidation impact penalty. Strictly
    positive, peaks at q=0."""
    if nu < 0:
        raise ValueError("nu must be >= 0")
    q = q_grid(p.Q)
    return np.exp(-p.k * nu * np.abs(q))


def stitching_terminal(p: HJBParams) -> np.ndarray:
    """Epoch-stitching terminal omega_q(T_fund) = f^0_q, the EXACT DISCRETE principal."""
    return principal_eigvec(p)


