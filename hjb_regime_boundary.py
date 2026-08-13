"""Operational regime-transition boundary u*."""
from __future__ import annotations

import math
import warnings

import numpy as np

from hjb_principal_eigenvector import HJBParams, discrete_autonomous_quotes
from hjb_riccati_solver import (
    FundingParams,
    RiccatiResult,
)

# BTCUSDT tick size (USDT). Used to express eps in ticks.
DEFAULT_TICK = 0.1


def analytical_u_hat(p: HJBParams, fp: FundingParams) -> float | None:
    """Order-of-magnitude boundary-layer width

        u_hat ~ (1/rho) ln( 2 rho_tilde |F_t| / (gamma sigma^2 Q) ).

    Returns None when rho_tilde |F_t| <= gamma sigma^2 Q / 2 (no boundary layer)
    or rho <= 0.
    """
    weight = fp.drain_scale() * abs(fp.F_t)
    threshold = p.gamma * p.sigma ** 2 * p.Q / 2.0
    if weight <= threshold or fp.rho <= 0.0:
        return None
    return (1.0 / fp.rho) * math.log(2.0 * weight / (p.gamma * p.sigma ** 2 * p.Q))


def deviation_profile(res: RiccatiResult, p: HJBParams) -> np.ndarray:
    """dev[i] = max over q (bid and ask) |delta_exact(u_i, q) - delta_auto(q)|.

    The capacity-edge nan entries are ignored. Returns an array aligned to
    res.u_grid.
    """
    db_auto, da_auto = discrete_autonomous_quotes(p)
    db_ex, da_ex = res.quotes(p)
    mb = np.isfinite(db_auto)
    ma = np.isfinite(da_auto)
    dev_b = np.max(np.abs(db_ex[:, mb] - db_auto[mb]), axis=1)
    dev_a = np.max(np.abs(da_ex[:, ma] - da_auto[ma]), axis=1)
    return np.maximum(dev_b, dev_a)


def find_u_star(
    res: RiccatiResult,
    p: HJBParams,
    *,
    eps_ticks: float = 1.0,
    tick: float = DEFAULT_TICK,
    u_margin: float = 0.0,
    warn: bool = True,
) -> tuple[float, np.ndarray]:
    """Operational u* = sup{ u : dev(u) > eps } (+ optional outward safety margin)."""
    eps = eps_ticks * tick
    dev = deviation_profile(res, p)
    u = res.u_grid
    mask = dev > eps
    if not mask.any():
        return 0.0, dev
    u_star = float(u[mask].max())
    if warn:
        idx = np.flatnonzero(mask)
        if bool(mask[-1]):
            warnings.warn(
                f"find_u_star: dev>eps at the grid edge (u_max={float(u[-1]):.1f}s); "
                f"u* is CLIPPED and under-covers the boundary layer -- extend u_max.",
                RuntimeWarning,
                stacklevel=2,
            )
        if idx[-1] - idx[0] + 1 != idx.size:
            warnings.warn(
                f"find_u_star: {{u: dev>eps}} is not a single contiguous block "
                f"(detached island out to u={float(u[idx[-1]]):.1f}s); the sup "
                f"over-reaches (still safe, errs long) -- inspect the LUT tail for "
                f"numerical artifacts.",
                RuntimeWarning,
                stacklevel=2,
            )
    if u_margin > 0.0:
        u_star = min(u_star + u_margin, float(u.max()))
    return u_star, dev


