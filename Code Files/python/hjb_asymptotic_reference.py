"""Closed-form Gueant-Lehalle-Tapia asymptotic quotes (Regime I)."""
from __future__ import annotations

import math

import numpy as np

from hjb_principal_eigenvector import (
    HJBParams,
    discrete_autonomous_quotes,
    q_grid,
)


def _const(p: HJBParams) -> float:
    return (1.0 / p.gamma) * math.log1p(p.gamma / p.k)


def asymptotic_scale(p: HJBParams) -> float:
    """scale = sqrt( sigma^2 gamma / (2 k A) * (1 + gamma/k)^{1 + k/gamma} )."""
    return math.sqrt(
        p.sigma ** 2 * p.gamma / (2.0 * p.k * p.A) * (1.0 + p.gamma / p.k) ** (1.0 + p.k / p.gamma)
    )


def delta_b_gaussian(p: HJBParams, q: np.ndarray | float) -> np.ndarray:
    """Closed-form asymptotic bid depth. Undefined at q=Q
    (bid withdrawn); caller masks the capacity edge."""
    q = np.asarray(q, dtype=np.float64)
    centre = p.alpha_ml / (p.gamma * p.sigma ** 2)
    return _const(p) + (-centre + (2.0 * q + 1.0) / 2.0) * asymptotic_scale(p)


def delta_a_gaussian(p: HJBParams, q: np.ndarray | float) -> np.ndarray:
    """Closed-form asymptotic ask depth. Undefined at q=-Q."""
    q = np.asarray(q, dtype=np.float64)
    centre = p.alpha_ml / (p.gamma * p.sigma ** 2)
    return _const(p) + (centre - (2.0 * q - 1.0) / 2.0) * asymptotic_scale(p)


def spread_gaussian(p: HJBParams) -> float:
    """Total asymptotic spread psi: the sum of the bid and ask depths."""
    return 2.0 * _const(p) + asymptotic_scale(p)


def gaussian_quote_grid(p: HJBParams) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian closed-form (delta_b, delta_a) over q in [-Q, Q] (index i=q+Q),
    capacity-edge quote set to np.nan (withdrawn)."""
    q = q_grid(p.Q)[::-1]  # ascending -Q..Q so index = q + Q
    db = delta_b_gaussian(p, q)
    da = delta_a_gaussian(p, q)
    db[-1] = np.nan  # q = Q : bid withdrawn
    da[0] = np.nan   # q = -Q: ask withdrawn
    return db, da


def gaussian_vs_discrete_quote_error(p: HJBParams) -> dict:
    """L_inf between the Gaussian closed-form quotes and the exact discrete-f^0
    quotes, over the interior (capacity edges excluded). The diagnostic for how
    deep into the interior Regime I is trustworthy."""
    db_g, da_g = gaussian_quote_grid(p)
    db_d, da_d = discrete_autonomous_quotes(p)
    mb = np.isfinite(db_g) & np.isfinite(db_d)
    ma = np.isfinite(da_g) & np.isfinite(da_d)
    return {
        "bid_Linf": float(np.max(np.abs(db_g[mb] - db_d[mb]))),
        "ask_Linf": float(np.max(np.abs(da_g[ma] - da_d[ma]))),
    }


def compare_to_lut(lut_delta_b: np.ndarray, lut_delta_a: np.ndarray, p: HJBParams, eps_ticks: float, tick: float = 0.1) -> dict:
    """Assert a LUT's far-from-settlement quotes match the Regime-I closed form to
    within eps_ticks. `lut_delta_b/a` are quote arrays indexed by q in [-Q, Q]
    (the deepest-interior temporal slice of the LUT). Returns per-side max error in
    ticks and a pass flag. (The LUT binary format itself is owned by
    hjb_lut_builder.py, deferred; this works on already-decoded arrays.)"""
    db_g, da_g = gaussian_quote_grid(p)
    mb = np.isfinite(db_g) & np.isfinite(lut_delta_b)
    ma = np.isfinite(da_g) & np.isfinite(lut_delta_a)
    eb = float(np.max(np.abs(db_g[mb] - lut_delta_b[mb]))) / tick
    ea = float(np.max(np.abs(da_g[ma] - lut_delta_a[ma]))) / tick
    return {"bid_err_ticks": eb, "ask_err_ticks": ea, "pass": eb <= eps_ticks and ea <= eps_ticks}


