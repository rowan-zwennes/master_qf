"""Quote rules for the six ablation strategies."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from hjb_lut_builder import (
    LUTHeader,
    build_log_omega,
    read_lut,
    read_lut_sens,
    sens_path_for,
)
from hjb_principal_eigenvector import (
    HJBParams,
    discrete_autonomous_quotes,
    principal_eigvec,
)
from hjb_regime_boundary import find_u_star
from hjb_riccati_solver import FundingParams, RiccatiResult


@dataclass(frozen=True)
class MarketConsts:
    """Derived Regime-I constants, refreshed whenever (sigma, A, k) roll."""

    gamma: float
    sigma: float
    A: float
    k: float

    # cached derived terms (computed once per refresh, read every 100 ms tick)
    c1: float = field(init=False)        # (1/gamma) ln(1 + gamma/k)
    scale: float = field(init=False)     # sqrt(sigma^2 gamma/(2kA) (1+g/k)^{1+k/g})
    inv_gs2: float = field(init=False)   # 1 / (gamma sigma^2)

    def __post_init__(self) -> None:
        g, s, A, k = self.gamma, self.sigma, self.A, self.k
        if min(g, s, A, k) <= 0.0:
            raise ValueError(f"non-positive market parameter: {self}")
        object.__setattr__(self, "c1", (1.0 / g) * math.log1p(g / k))
        object.__setattr__(
            self, "scale",
            math.sqrt(s * s * g / (2.0 * k * A) * (1.0 + g / k) ** (1.0 + k / g)),
        )
        object.__setattr__(self, "inv_gs2", 1.0 / (g * s * s))

    def hjb_params(self, alpha_ml: float, Q: int) -> HJBParams:
        return HJBParams(gamma=self.gamma, sigma=self.sigma, A=self.A,
                         k=self.k, alpha_ml=alpha_ml, Q=Q)


def glt_depths(mc: MarketConsts, q: int, alpha: float) -> tuple[float, float]:
    shift = alpha * mc.inv_gs2
    delta_b = mc.c1 + (-shift + q + 0.5) * mc.scale
    delta_a = mc.c1 + (shift - q + 0.5) * mc.scale
    return delta_b, delta_a


def ml_quote_shift(mc: MarketConsts, alpha: float) -> float:
    """The pure reservation-price shift alpha/(gamma sigma^2) * scale (USDT).
    Subtract from delta_b, add to delta_a."""
    return alpha * mc.inv_gs2 * mc.scale


def exact_drift_horizon(mc: MarketConsts, Q: int, a_probe: float = 1.0e-4) -> float:
    """The exact-f^0 permanent-drift horizon (seconds): the q=0 reservation shift per unit."""
    r1 = RegimeIQuoter.build(mc.hjb_params(a_probe, Q))
    db0, da0 = r1.depths(0)
    return 0.5 * (da0 - db0) / a_probe


class RegimeIQuoter:
    """Exact Regime-I quotes from f^0 (drift baked in), drop-in for glt_depths."""

    def __init__(self, delta_b: np.ndarray, delta_a: np.ndarray, Q: int,
                 log_f0: np.ndarray) -> None:
        self.delta_b = delta_b          # ascending q+Q
        self.delta_a = delta_a
        self.Q = Q
        self.log_f0 = log_f0            # f32, q = Q..-Q, the C++ stream payload

    @classmethod
    def build(cls, p: HJBParams) -> "RegimeIQuoter":
        from hjb_riccati_solver import quotes_from_logomega
        f0 = principal_eigvec(p)                       # q = Q..-Q, positive
        log_f0 = np.log(f0).astype(np.float32).astype(np.float64)
        db, da = quotes_from_logomega(log_f0, p)        # ascending q+Q order
        return cls(np.asarray(db, float), np.asarray(da, float), p.Q,
                   log_f0.astype(np.float32))

    def depths(self, q: int) -> tuple[float, float]:
        i = q + self.Q
        return float(self.delta_b[i]), float(self.delta_a[i])


class ASCorrection:
    """Precomputed per-side AS quote correction, interpolated in alpha per tick."""

    def __init__(self, mc: MarketConsts, c_xi: float, Q: int, *,
                 alpha_max: float, n_grid: int = 41) -> None:
        self.Q = Q
        self.c_xi = c_xi
        self.alpha_grid = np.linspace(-abs(alpha_max), abs(alpha_max), n_grid)
        nq = 2 * Q + 1
        self.db = np.zeros((n_grid, nq))
        self.da = np.zeros((n_grid, nq))
        if c_xi == 0.0:
            return
        for g, al in enumerate(self.alpha_grid):
            xi_b = c_xi * max(al, 0.0)
            xi_a = c_xi * max(-al, 0.0)
            p_as = HJBParams(gamma=mc.gamma, sigma=mc.sigma, A=mc.A, k=mc.k,
                             alpha_ml=al, xi_b=xi_b, xi_a=xi_a, Q=Q)
            p_glt = HJBParams(gamma=mc.gamma, sigma=mc.sigma, A=mc.A, k=mc.k,
                              alpha_ml=al, Q=Q)
            dba, daa = discrete_autonomous_quotes(p_as, principal_eigvec(p_as))
            dbg, dag = discrete_autonomous_quotes(p_glt)
            self.db[g] = np.nan_to_num(dba - dbg, nan=0.0)
            self.da[g] = np.nan_to_num(daa - dag, nan=0.0)

    def correction(self, alpha: float, q: int) -> tuple[float, float]:
        """(Delta_b, Delta_a) signed depth corrections (USDT) to ADD to the quote."""
        i_q = q + self.Q
        db = float(np.interp(alpha, self.alpha_grid, self.db[:, i_q]))
        da = float(np.interp(alpha, self.alpha_grid, self.da[:, i_q]))
        return db, da


def chi_effective_consts(mc: MarketConsts, chi0: float,
                         chi1: float) -> MarketConsts:
    """MarketConsts of the chi-corrected problem in dt-space."""
    if not 0.0 <= chi1 < 1.0:
        raise ValueError(f"chi1 must be in [0, 1): {chi1}")
    k_eff = mc.k / (1.0 - chi1)
    a_eff = mc.A * math.exp(-k_eff * chi0)
    return MarketConsts(mc.gamma, mc.sigma, max(a_eff, 1e-12), k_eff)


def chi_unscale(depth_tilde: float, chi0: float, chi1: float) -> float:
    """Map a dt-space optimal depth back to original delta units."""
    return (depth_tilde + chi0) / (1.0 - chi1)


class RegimeIIQuoter:
    """Pre-computed quote grids from a backward-induction LUT."""

    def __init__(self, delta_b: np.ndarray, delta_a: np.ndarray,
                 du_s: float, u_star_s: float, Q: int,
                 meta: dict | None = None,
                 db_sens: np.ndarray | None = None,
                 da_sens: np.ndarray | None = None) -> None:
        self.delta_b = delta_b          # (n_q, n_u)
        self.delta_a = delta_a
        self.du_s = du_s
        self.u_star_s = u_star_s
        self.Q = Q
        self.n_u = delta_b.shape[1]
        self.meta = meta or {}
        self.db_sens = db_sens
        self.da_sens = da_sens

    @staticmethod
    def _grids(p: HJBParams, fp: FundingParams, u_max: float, du_s: float,
               eps_ticks: float, tick: float):
        """(db_g, da_g [rows q=Q..-Q], u_star, n_u) from one backward induction."""
        from hjb_riccati_solver import quotes_from_logomega
        res = build_log_omega(p, fp, u_max, du_s)
        u_star, _ = find_u_star(res, p, eps_ticks=eps_ticks, tick=tick)
        logw = res.log_omega.astype(np.float32).astype(np.float64).T  # (n_q,n_u)
        n_u = res.u_grid.size
        db_g = np.empty((2 * p.Q + 1, n_u))
        da_g = np.empty((2 * p.Q + 1, n_u))
        for j in range(n_u):
            b, a = quotes_from_logomega(logw[:, j], p)
            db_g[:, j] = b[::-1]                     # rows q = Q..-Q
            da_g[:, j] = a[::-1]
        return db_g, da_g, u_star, n_u, res.method

    @classmethod
    def build(cls, p: HJBParams, fp: FundingParams, *, u_max: float,
              du_s: float, eps_ticks: float = 1.0, tick: float = 0.1,
              q0_ref: float | None = None) -> "RegimeIIQuoter":
        """Backward-induct (Radau reference) and pre-compute the quote grids."""
        db_g, da_g, u_star, _, method = cls._grids(p, fp, u_max, du_s,
                                                   eps_ticks, tick)
        db_sens = da_sens = None
        if q0_ref is not None:
            gs2 = p.gamma * p.sigma * p.sigma          # q0 -> alpha: alpha = q0*gs2
            p_ref = replace(p, alpha_ml=q0_ref * gs2)
            db_r, da_r, _, _, _ = cls._grids(p_ref, fp, u_max, du_s,
                                             eps_ticks, tick)
            db_sens = ((db_r - db_g) / q0_ref).astype(np.float32).astype(np.float64)
            da_sens = ((da_r - da_g) / q0_ref).astype(np.float32).astype(np.float64)
        return cls(db_g, da_g, du_s, u_star, p.Q,
                   meta={"F_t": fp.F_t, "rho": fp.rho, "solver": method,
                         "q0_ref": q0_ref},
                   db_sens=db_sens, da_sens=da_sens)

    @classmethod
    def from_file(cls, path: str | Path, *, gamma: float | None = None,
                  ) -> "RegimeIIQuoter":
        """Load a serialized LUT (hjb_lut_builder binary) and pre-compute the
        quote grids. Used for C++/Python parity checks and cached sweeps."""
        from hjb_riccati_solver import quotes_from_logomega

        hdr, logw = read_lut(path)                  # logw: (n_q, n_u), q=Q..-Q
        Q = hdr.q_max
        p = HJBParams(gamma=gamma if gamma is not None else hdr.gamma,
                      sigma=hdr.sigma, A=hdr.A, k=hdr.k,
                      alpha_ml=hdr.alpha_ml, Q=Q)
        n_u = hdr.n_u
        db = np.empty((hdr.n_q, n_u))
        da = np.empty((hdr.n_q, n_u))
        for j in range(n_u):
            b, a = quotes_from_logomega(logw[:, j], p)
            db[:, j] = b[::-1]                      # back to q = Q..-Q rows
            da[:, j] = a[::-1]
        db_sens = da_sens = None
        spath = sens_path_for(path)
        if spath.exists():
            db_sens, da_sens, _q0ref = read_lut_sens(spath)
            if db_sens.shape != db.shape:
                raise ValueError(f"sens shape {db_sens.shape} != LUT {db.shape}")
        return cls(db, da, hdr.du_ms / 1000.0, hdr.u_star_s, Q,
                   meta={"header": hdr}, db_sens=db_sens, da_sens=da_sens)

    def _u_interp(self, table: np.ndarray, i_q: int, u_s: float) -> float:
        x = u_s / self.du_s
        if x <= 0.0:
            return float(table[i_q, 0])
        if x >= self.n_u - 1:
            return float(table[i_q, -1])
        i0 = int(x)
        a = x - i0
        return float((1.0 - a) * table[i_q, i0] + a * table[i_q, i0 + 1])

    def depths(self, q: int, u_s: float) -> tuple[float, float]:
        """(delta_b, delta_a) at inventory q, backward time u (seconds)."""
        i_q = self.Q - q
        return (self._u_interp(self.delta_b, i_q, u_s),
                self._u_interp(self.delta_a, i_q, u_s))

    def depths_drift(self, q: int, u_s: float, q0: float) -> tuple[float, float]:
        """(delta_b, delta_a) with the linear-response ML-drift correction
        superposed: delta + q0 * (∂delta/∂q0), q0 = alpha/(gamma sigma^2). Falls
        back to the plain LUT quote when no sensitivity was built (db_sens None).
        Exact to first order in q0; the O(q0^2) residual is bounded and clipped
        (quantify_ml_interp_error.py). Both sides use the same u-interpolation as
        depths, so the correction inherits the LUT's smooth-in-u lookup."""
        i_q = self.Q - q
        db = self._u_interp(self.delta_b, i_q, u_s)
        da = self._u_interp(self.delta_a, i_q, u_s)
        if self.db_sens is None:
            return db, da
        return (db + q0 * self._u_interp(self.db_sens, i_q, u_s),
                da + q0 * self._u_interp(self.da_sens, i_q, u_s))


class EmaDrift:
    """RETIRED (see the note above)."""

    def __init__(self, half_life_s: float = 10.0) -> None:
        self.half_life_s = half_life_s
        self.alpha = 0.0
        self._last_price = math.nan
        self._last_ms = -1

    def reset(self) -> None:
        self.alpha = 0.0
        self._last_price = math.nan
        self._last_ms = -1

    def update(self, ts_ms: int, price: float) -> float:
        if self._last_ms >= 0 and ts_ms > self._last_ms:
            dt = (ts_ms - self._last_ms) / 1000.0
            slope = (price - self._last_price) / dt
            w = math.exp(-math.log(2.0) * dt / self.half_life_s)
            self.alpha = w * self.alpha + (1.0 - w) * slope
        self._last_price = price
        self._last_ms = ts_ms
        return self.alpha


@dataclass(frozen=True)
class StrategySpec:
    sid: int
    name: str
    drift: str
    funding: bool       # Regime II LUT inside u*
    naive: bool = False # strategy 1: fixed half-spread, no GLT at all


STRATEGIES: dict[int, StrategySpec] = {
    1: StrategySpec(1, "naive_const_spread", "none", False, naive=True),
    2: StrategySpec(2, "glt_standard", "none", False),
    3: StrategySpec(3, "glt_ar_drift", "ar", False),
    4: StrategySpec(4, "glt_ml_drift", "ml", False),
    5: StrategySpec(5, "glt_funding", "none", True),
    6: StrategySpec(6, "glt_full_hybrid", "ml", True),
}


def quote_depths(
    spec: StrategySpec,
    mc: MarketConsts,
    q: int,
    alpha: float,
    u_s: float,
    r2: RegimeIIQuoter | None,
    *,
    fixed_half_spread: float = 5.0,
    drift_in_lut: bool = False,
    r1: RegimeIQuoter | None = None,
    force_r2: bool = False,
) -> tuple[float, float]:
    """Quote depths (delta_b, delta_a) from the anchor for one strategy."""
    if spec.naive:
        return fixed_half_spread, fixed_half_spread

    if spec.funding and r2 is not None and (force_r2 or u_s <= r2.u_star_s):
        if spec.drift == "ml" and not drift_in_lut and r2.db_sens is not None:
            return r2.depths_drift(q, u_s, alpha * mc.inv_gs2)
        delta_b, delta_a = r2.depths(q, u_s)
        if spec.drift == "ml" and not drift_in_lut:
            shift = ml_quote_shift(mc, alpha)
            delta_b -= shift
            delta_a += shift
        return delta_b, delta_a

    if r1 is not None:
        return r1.depths(q)

    return glt_depths(mc, q, alpha)


