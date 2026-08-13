"""Regime-II drift approximation error."""
from __future__ import annotations

import json
import math

import numpy as np

from hjb_principal_eigenvector import HJBParams
from hjb_riccati_solver import FundingParams
from run_simulation import SimConfig
from sim_quote_engine import MarketConsts, RegimeIIQuoter, ml_quote_shift


def _interp_q(r2: RegimeIIQuoter, qr: float, u: float, side: int) -> float:
    """δ0(qr, u) at real-valued inventory qr by linear blend of integer rows
    (each via r2.depths), extrapolating on the terminal slope past the edge.
    side: 0 = bid (valid q in [-Q, Q-1]), 1 = ask (valid q in [-Q+1, Q])."""
    Q = r2.Q
    lo_min, lo_max = (-Q, Q - 2) if side == 0 else (-Q + 1, Q - 1)
    lo = max(lo_min, min(lo_max, math.floor(qr)))   # both lo and lo+1 valid this side
    hi = lo + 1
    dl = r2.depths(lo, u)[side]
    dh = r2.depths(hi, u)[side]
    return dl + (qr - lo) * (dh - dl)          # blend inside [lo,hi]; extrapolate outside


def run(cfg: SimConfig, F_t: float, q0_grid, u_grid, out_path: str,
        q0_ref: float = 0.1) -> dict:
    g, s, A, k, Q = cfg.gamma_eff, 4.575, 0.2742, 0.0900, cfg.Q
    mc = MarketConsts(g, s, A, k)
    fp = FundingParams(F_t=F_t, rho=cfg.rho_eff, mode=cfg.funding_mode)
    u_max = float(u_grid[-1]) + cfg.lut_du_s

    # β=0 LUT (built once; the correction base) and its u* boundary.
    p0 = HJBParams(gamma=g, sigma=s, A=A, k=k, alpha_ml=0.0, Q=Q)
    r0 = RegimeIIQuoter.build(p0, fp, u_max=u_max, du_s=cfg.lut_du_s,
                              eps_ticks=cfg.eps_ticks, tick=cfg.tick)
    pr = HJBParams(gamma=g, sigma=s, A=A, k=k, alpha_ml=q0_ref * g * s * s, Q=Q)
    rr = RegimeIIQuoter.build(pr, fp, u_max=u_max, du_s=cfg.lut_du_s,
                              eps_ticks=cfg.eps_ticks, tick=cfg.tick)

    rows = []
    for q0 in q0_grid:
        aml = q0 * g * s * s                    # α_ML implied by this q0
        shift = ml_quote_shift(mc, aml)         # legacy Gaussian shift (USDT)
        # exact LUT: same M, drift baked in.
        pe = HJBParams(gamma=g, sigma=s, A=A, k=k, alpha_ml=aml, Q=Q)
        re = RegimeIIQuoter.build(pe, fp, u_max=u_max, du_s=cfg.lut_du_s,
                                  eps_ticks=cfg.eps_ticks, tick=cfg.tick)
        for u in u_grid:
            for q in range(-Q, Q + 1):
                for side, ok in ((0, q < Q), (1, q > -Q)):   # skip the NaN cap side
                    if not ok:
                        continue
                    exact = re.depths(q, u)[side]
                    sgn = -1.0 if side == 0 else 1.0          # bid -shift, ask +shift
                    d0 = r0.depths(q, u)[side]
                    slope = (rr.depths(q, u)[side] - d0) / q0_ref  # exact ∂δ/∂q0
                    e_interp = _interp_q(r0, q - q0, u, side) - exact
                    e_add = (d0 + sgn * shift) - exact
                    e_lin = (d0 + q0 * slope) - exact
                    rows.append((q0, u, q, side, e_interp, e_add, e_lin))

    # cols: 0 ei, 1 ea, 2 el, 3 q0, 4 |q|
    arr = np.array([(r[4], r[5], r[6], r[0], abs(r[2])) for r in rows])
    tick = cfg.tick
    edge = arr[:, 4] >= Q - 1                                     # |q| in {Q-1, Q}

    def stat(col, mask):
        v = np.abs(arr[mask, col]) / tick
        return dict(max=float(v.max()), p99=float(np.percentile(v, 99)),
                    mean=float(v.mean()))

    def method_block(col, mask):
        return {"all": stat(col, mask), "bulk": stat(col, mask & ~edge),
                "edge": stat(col, mask & edge)}

    cols = {"interp": 0, "additive": 1, "linear": 2}
    per_q0 = {}
    for q0 in q0_grid:
        m = arr[:, 3] == q0
        per_q0[f"{q0}"] = {name: method_block(c, m) for name, c in cols.items()}

    allm = np.ones(len(arr), bool)
    res = {
        "params": dict(gamma=g, sigma=s, A=A, k=k, rho=cfg.rho_eff, F_t=F_t,
                       Q=Q, du_s=cfg.lut_du_s, u_star_s=r0.u_star_s,
                       q0_ref=q0_ref, q0_to_alpha=g * s * s),
        "per_q0_ticks": per_q0,
        "interp_ticks": method_block(0, allm),
        "additive_ticks": method_block(1, allm),
        "linear_ticks": method_block(2, allm),
    }
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2)
    return res


if __name__ == "__main__":
    cfg = SimConfig()
    cfg.lut_du_s = 0.5     # quote error is independent of u-grid resolution (u is
    u_grid = np.arange(0.0, 1300.0 + 1e-9, 10.0)
    q0_grid = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 26.0, 50.0, 91.0, 158.0]
    for F_t, tag in ((2.5, "median"), (10.0, "p99"), (40.0, "stress")):
        res = run(cfg, F_t, q0_grid, u_grid, f"reports/ml_interp_error_{tag}.json")
        print(f"[F_t={F_t} {tag}] u*={res['params']['u_star_s']:.0f}s  "
              f"q0_ref={res['params']['q0_ref']}  "
              f"q0=α_ML×{1/res['params']['q0_to_alpha']:.0f}  (max abs err, ticks)")
        print(f"  {'q0':>4} {'α_ML':>9} | {'LIN bulk':>9} {'LIN edge':>9}"
              f" | {'add bulk':>9} {'add edge':>9} | {'interp edge':>11}")
        for q0 in q0_grid:
            d = res["per_q0_ticks"][f"{q0}"]
            aml = q0 * res["params"]["q0_to_alpha"]
            print(f"  {q0:>4} {aml:>9.2e} | {d['linear']['bulk']['max']:>9.3f}"
                  f" {d['linear']['edge']['max']:>9.3f} |"
                  f" {d['additive']['bulk']['max']:>9.3f}"
                  f" {d['additive']['edge']['max']:>9.3f} |"
                  f" {d['interp']['edge']['max']:>11.3f}")
