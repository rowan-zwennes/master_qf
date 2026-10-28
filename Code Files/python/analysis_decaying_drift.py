"""Decaying-drift versus permanent-drift quote comparison."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from hjb_principal_eigenvector import HJBParams, principal_eigvec, q_grid
from hjb_riccati_solver import intrinsic_relaxation_rate, quotes_from_logomega
from run_simulation import SimConfig
from sim_quote_engine import MarketConsts, RegimeIQuoter

SIGMA = 4.575
A_INT = 0.2742
K_INT = 0.0900


def _decaying_logspace_rhs(
    u: float, y: np.ndarray, p: HJBParams, U: float, alpha_0: float, h: float
) -> np.ndarray:
    """d(ln omega)/du for the autonomous market with a DECAYING drift and no funding."""
    q = q_grid(p.Q)
    alpha_u = alpha_0 * math.exp(-(U - u) / h)
    diag = p.alpha * q ** 2 - p.k * alpha_u * q      # D_q(u), ordered Q..-Q
    dy = -diag
    eta = p.eta
    dy[1:] += eta * np.exp(y[:-1] - y[1:])           # lower neighbour (bid, q->q+1)
    dy[:-1] += eta * np.exp(y[1:] - y[:-1])          # upper neighbour (ask, q->q-1)
    return dy


def solve_decaying(p_base: HJBParams, alpha_0: float, h: float,
                   horizon_mult: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """(delta_b, delta_a) at the observation instant for a drift decaying with timescale h."""
    tau_relax = 1.0 / intrinsic_relaxation_rate(p_base)      # ~150 s at calibration
    U = max(horizon_mult * h, 6.0 * tau_relax)
    y0 = np.log(principal_eigvec(HJBParams(gamma=p_base.gamma, sigma=p_base.sigma,
                                           A=p_base.A, k=p_base.k, alpha_ml=0.0,
                                           Q=p_base.Q)))
    y0 -= y0.max()
    sol = solve_ivp(_decaying_logspace_rhs, (0.0, U), y0, method="Radau",
                    args=(p_base, U, alpha_0, h), rtol=1e-9, atol=1e-11,
                    dense_output=False, t_eval=[U])
    if not sol.success:
        raise RuntimeError(f"decaying-drift solve failed: {sol.message}")
    y_now = sol.y[:, -1]
    y_now -= y_now.max()
    return quotes_from_logomega(y_now, p_base)


def exact_f0_horizon(p_base: HJBParams) -> float:
    """The TRUE permanent-drift horizon of the deployed exact-f^0 quote: the q=0
    reservation shift per unit permanent drift, finite-differenced at a small probe.
    This is the Gaussian h_eff corrected for the ~2.16x discrete-vs-Gaussian slope
    (quantify_ml_interp_error.py), measured self-consistently rather than hardcoded.
    Sizing effective_alpha against THIS (not h_eff) lands the deployed quote on the
    finite-horizon optimum alpha_0 * h."""
    Q = p_base.Q
    a_probe = 1.0e-4                                  # small: linear regime
    db, da = _permanent_quotes(p_base, a_probe)
    shift = 0.5 * (da[Q] - db[Q])
    return shift / a_probe


def fixed_effective_alpha(mc: MarketConsts, h_true: float,
                          alpha_0: float, h: float) -> float:
    """The production (--ml-exact-drift-horizon) alpha->effective_alpha map: size the
    target finite-horizon shift clip(alpha_0 h, +-c1) against the EXACT-f^0 drift
    horizon h_true (run_simulation.py drift_horizon = exact_drift_horizon). Baking
    this into exact-f^0 lands the deployed q=0 skew on the finite-horizon target
    alpha_0 h by construction."""
    shift_usd = alpha_0 * h
    shift_usd = max(-mc.c1, min(mc.c1, shift_usd))
    return shift_usd / h_true


def _permanent_quotes(p_base: HJBParams, alpha_eff: float) -> tuple[np.ndarray, np.ndarray]:
    """Deployed permanent-drift exact-f^0 quote at a given effective alpha
    (RegimeIQuoter.build, the same object run_simulation quotes Regime I from)."""
    r1 = RegimeIQuoter.build(HJBParams(gamma=p_base.gamma, sigma=p_base.sigma,
                                       A=p_base.A, k=p_base.k, alpha_ml=alpha_eff,
                                       Q=p_base.Q))
    Q = p_base.Q
    db = np.array([r1.depths(q)[0] for q in range(-Q, Q + 1)])
    da = np.array([r1.depths(q)[1] for q in range(-Q, Q + 1)])
    return db, da


def _max_tick_err(a: np.ndarray, b: np.ndarray, absq: np.ndarray, mask: np.ndarray,
                  tick: float) -> float:
    """max |a-b|/tick over positions selected by mask, ignoring NaN cap sides."""
    d = np.abs(a - b) / tick
    sel = mask & np.isfinite(d)
    return float(d[sel].max()) if sel.any() else 0.0


def run(cfg: SimConfig, h_grid, q0_grid, out_path: str) -> dict:
    g, Q, tick = cfg.gamma_eff, cfg.Q, cfg.tick
    p_base = HJBParams(gamma=g, sigma=SIGMA, A=A_INT, k=K_INT, alpha_ml=0.0, Q=Q)
    mc = MarketConsts(g, SIGMA, A_INT, K_INT)
    h_true = exact_f0_horizon(p_base)               # deployed exact-f^0 drift horizon

    q = np.arange(-Q, Q + 1)
    absq = np.abs(q)
    edge = absq >= Q - 1                             # |q| in {Q-1, Q}
    bulk = ~edge

    def shift0(db, da):
        """q=0 reservation displacement (USDT): half the bid/ask depth asymmetry.
        Positive drift tightens the bid and widens the ask, so this is +ve."""
        return 0.5 * (da[Q] - db[Q])

    per_h = {}
    for h in h_grid:
        rows = {}
        for q0 in q0_grid:
            # operating q0 -> target shift -> raw signal at THIS decay horizon
            alpha_0 = q0 * h_true / (mc.inv_gs2 * h)
            db_r, da_r = solve_decaying(p_base, alpha_0, h)
            a_fix = fixed_effective_alpha(mc, h_true, alpha_0, h)
            db_f, da_f = _permanent_quotes(p_base, a_fix)       # deployed exact-drift

            def block(db_m, da_m):
                aq2 = np.concatenate([absq, absq])
                m_bulk = np.concatenate([bulk, bulk])
                m_edge = np.concatenate([edge, edge])
                return {
                    "bulk_ticks": _max_tick_err(np.concatenate([db_m, da_m]),
                                                np.concatenate([db_r, da_r]),
                                                aq2, m_bulk, tick),
                    "edge_ticks": _max_tick_err(np.concatenate([db_m, da_m]),
                                                np.concatenate([db_r, da_r]),
                                                aq2, m_edge, tick),
                }

            rows[f"{q0}"] = {
                "alpha_0": alpha_0,
                "shift_real_usdt": shift0(db_r, da_r),
                "shift_fixed_usdt": shift0(db_f, da_f),
                "target_alpha0_h_usdt": alpha_0 * h,
                "effective_alpha": a_fix,
                "fixed_vs_real": block(db_f, da_f),
            }
        per_h[f"{h}"] = rows

    res = {
        "params": dict(gamma=g, sigma=SIGMA, A=A_INT, k=K_INT, Q=Q, tick=tick,
                       h_true_s=h_true, c1=mc.c1, scale=mc.scale,
                       inv_gs2=mc.inv_gs2,
                       q0_convention="operating: alpha_0 = q0*h_true/(inv_gs2*h)"),
        "per_h": per_h,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, indent=2))
    return res


def per_q_profile(cfg: SimConfig, h: float, q0: float) -> dict:
    """Per-inventory depth-error profile at ONE (h, q0), for verifying WHERE the deployed."""
    g, Q, tick = cfg.gamma_eff, cfg.Q, cfg.tick
    p_base = HJBParams(gamma=g, sigma=SIGMA, A=A_INT, k=K_INT, alpha_ml=0.0, Q=Q)
    mc = MarketConsts(g, SIGMA, A_INT, K_INT)
    h_true = exact_f0_horizon(p_base)
    alpha_0 = q0 * h_true / (mc.inv_gs2 * h)        # operating-q0 convention

    db_r, da_r = solve_decaying(p_base, alpha_0, h)                       # ground truth
    db_f, da_f = _permanent_quotes(p_base, fixed_effective_alpha(mc, h_true, alpha_0, h))

    def errs(db_m, da_m):
        eb = np.abs(db_m - db_r) / tick
        ea = np.abs(da_m - da_r) / tick
        return eb, ea

    q = np.arange(-Q, Q + 1)
    out = {
        "h": h, "q0": q0, "alpha_0": alpha_0, "h_true_s": h_true, "Q": Q, "tick": tick,
        "q": q.tolist(),
        "db_real": db_r.tolist(), "da_real": da_r.tolist(),
        "db_fixed": db_f.tolist(), "da_fixed": da_f.tolist(),
    }
    eb, ea = errs(db_f, da_f)
    out["err_bid_ticks_fixed"] = eb.tolist()
    out["err_ask_ticks_fixed"] = ea.tolist()
    return out


def _print_profile(prof: dict) -> None:
    Q = prof["Q"]
    print(f"per-q error profile at h = {prof['h']} s, q0 = {prof['q0']}  "
          f"(alpha_0 = {prof['alpha_0']:.4e},  h_true = {prof['h_true_s']:.1f} s,  "
          f"tick = {prof['tick']:g} USDT)")
    print("errors are |delta_method - delta_real| in TICKS; watch the fixed error "
          "grow toward |q| = Q.\n")
    q = prof["q"]
    dbr, dar = prof["db_real"], prof["da_real"]
    dbf, daf = prof["db_fixed"], prof["da_fixed"]
    ebf, eaf = prof["err_bid_ticks_fixed"], prof["err_ask_ticks_fixed"]

    def f(x):
        return "     nan" if (x != x) else f"{x:>8.3f}"

    print(f"  {'q':>3} | {'db_real':>8} {'db_fix':>8} {'fix_b':>7} "
          f"| {'da_real':>8} {'da_fix':>8} {'fix_a':>7}  (err ticks)")
    for i, qi in enumerate(q):
        print(f"  {qi:>3} | {f(dbr[i])} {f(dbf[i])} {f(ebf[i])} "
              f"| {f(dar[i])} {f(daf[i])} {f(eaf[i])}")
    absq = np.abs(np.array(q))
    allf = np.array(ebf + eaf)
    aq2 = np.concatenate([absq, absq])
    fin = np.isfinite(allf)
    bulk = (aq2 <= Q - 2) & fin
    wall = (aq2 >= Q - 1) & fin
    print(f"\n  max fixed error: bulk(|q|<=Q-2) = {allf[bulk].max():.3f} t   "
          f"wall(|q|>=Q-1) = {allf[wall].max():.3f} t")


def _print(res: dict) -> None:
    p = res["params"]
    print(f"h_true (exact-f0 drift horizon) = {p['h_true_s']:.1f} s   "
          f"tick = {p['tick']:g} USDT")
    print(f"q0 = operating drift (shift/(h_true gamma sigma^2)); "
          f"alpha_0 = q0*h_true/(inv_gs2*h);  c1 (clip) = {p['c1']:.2f} USDT")
    print("shift_* = q=0 reservation displacement (USDT); "
          "errors = max|delta_fixed-delta_real|\n")
    for h, rows in res["per_h"].items():
        print(f"=== signal horizon h = {h} s ===")
        print(f"  {'q0':>4} {'shift_real':>10} {'target a0h':>10} {'shift_fix':>10} | "
              f"{'fix bulk':>8} {'fix edge':>8}  (ticks)")
        for q0, d in rows.items():
            fv = d["fixed_vs_real"]
            print(f"  {q0:>4} {d['shift_real_usdt']:>10.4f} "
                  f"{d['target_alpha0_h_usdt']:>10.4f} {d['shift_fixed_usdt']:>10.4f} | "
                  f"{fv['bulk_ticks']:>8.2f} {fv['edge_ticks']:>8.2f}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", action="store_true",
                    help="print the per-q depth-error profile at a single "
                         "(h, q0) instead of the summary sweep")
    ap.add_argument("--profile-h", type=float, default=10.0)
    ap.add_argument("--profile-q0", type=float, default=26.0)
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--h-grid", type=float, nargs="+", default=[1.0, 5.0, 10.0, 30.0])
    ap.add_argument("--q0-grid", type=float, nargs="+", default=[1.0, 26.0, 91.0, 158.0])
    args = ap.parse_args()
    cfg = SimConfig()
    if args.profile:
        prof = per_q_profile(cfg, args.profile_h, args.profile_q0)
        _print_profile(prof)
        out = Path(args.report_dir) / f"decaying_drift_profile_h{args.profile_h:g}_q{args.profile_q0:g}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(prof, indent=2))
        print(f"\n-> {out}")
        return
    res = run(cfg, args.h_grid, args.q0_grid,
              str(Path(args.report_dir) / "decaying_drift.json"))
    _print(res)
    print(f"-> {Path(args.report_dir) / 'decaying_drift.json'}")


if __name__ == "__main__":
    main()
