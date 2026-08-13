"""Numerical check of the frozen reference price S_ref."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

try:
    import pipeline_utils as pu

    _HAVE_PU = True
except Exception:
    _HAVE_PU = False

from calibrate_volatility import MIN_COVERAGE_DEFAULT, _rolling_rv
from hjb_principal_eigenvector import HJBParams
from hjb_riccati_solver import intrinsic_relaxation_rate


BOOK_GRID_MS = 100
DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
TICK_SIZE_USD = 0.1  # BTCUSDT perp tick size
FUNDING_INTERVAL_MS = 8 * 3600 * 1000  # Binance funding settles every 8 h (00/08/16 UTC)


def settlements_in(ts0: int, ts1: int) -> np.ndarray:
    """Funding settlement timestamps (the 8 h UTC grid) within [ts0, ts1]."""
    first = -(-int(ts0) // FUNDING_INTERVAL_MS) * FUNDING_INTERVAL_MS  # ceil to grid
    if first > ts1:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, int(ts1) + 1, FUNDING_INTERVAL_MS, dtype=np.int64)


@dataclass
class SrefConfig:
    dt_grid_s: tuple[float, ...] = (0.1, 0.5, 1.0)
    abs_q: int = 10  # representative |q| (defaults to inventory capacity Q)
    abs_f: float = 3.7e-5
    tick: float = TICK_SIZE_USD


def mid_price_expr() -> pl.Expr:
    return ((pl.col("bid_p_0") + pl.col("ask_p_0")) / 2.0).alias("mid_price")


Segment = tuple[np.ndarray, np.ndarray]  # (ts_ms:int64, mid_price:float64)


def load_real_segments(base: Path, fday: str) -> list[Segment]:
    """Return a list of contiguous valid (timestamp, mid-price) arrays (one per valid segment)."""
    if not _HAVE_PU:
        raise RuntimeError("pipeline_utils unavailable")
    book = pu.load_book_for_funding_day(base, fday)
    if book.is_empty():
        raise RuntimeError(f"no book data for funding day {fday} under {base}")
    book = book.with_columns(mid_price_expr())
    segments: list[Segment] = []
    for _start, _end, seg in pu.iter_valid_segments(book, min_segment_seconds=5.0):
        ts = seg["ts_ms"].to_numpy().astype(np.int64)
        mp = seg["mid_price"].to_numpy().astype(np.float64)
        ok = np.isfinite(mp)
        ts, mp = ts[ok], mp[ok]
        if mp.size > 10:
            segments.append((ts, mp))
    if not segments:
        raise RuntimeError(f"no valid segments >5s for {fday}")
    return segments


def load_preanalysis_segments(base: Path) -> list[tuple[str, list[Segment]]]:
    """Valid mid-price segments for every funding day in the 30-day pre-analysis split."""
    if not _HAVE_PU:
        raise RuntimeError("pipeline_utils unavailable")
    splits = pu.load_splits(base)
    days = splits["splits"]["pre_analysis"]
    out: list[tuple[str, list[Segment]]] = []
    for fday in days:
        try:
            segs = load_real_segments(base, fday)
        except RuntimeError as e:
            print(f"  skip {fday}: {e}")
            continue
        out.append((fday, segs))
    if not out:
        raise RuntimeError("no pre-analysis funding day had usable book data")
    return out


def rolling_sigma(prices: np.ndarray, window_s: float = 60.0) -> np.ndarray:
    """Rolling realised volatility in USD * s^-1/2, aligned to `prices` indices."""
    dt_step_s = BOOK_GRID_MS / 1000.0
    window = max(2, int(round(window_s / dt_step_s)))
    min_obs = max(2, int(math.ceil(MIN_COVERAGE_DEFAULT * window)))
    ds = np.empty(prices.size, dtype=np.float64)
    ds[0] = np.nan
    if prices.size > 1:
        ds[1:] = np.diff(prices)
    sigma, _nobs, _raw = _rolling_rv(ds, window, dt_step_s, min_obs)
    return sigma


def window_errors(
    prices: np.ndarray, win_steps: int, sigma_series: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For non-overlapping dt-wide windows, return eps_price : sup_{t in window} |S_t - S_ref|."""
    n_win = prices.size // win_steps
    if n_win == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    if sigma_series is None:
        sigma_series = rolling_sigma(prices)
    trimmed = prices[: n_win * win_steps].reshape(n_win, win_steps)
    s_ref = trimmed[:, :1]
    eps_price = np.max(np.abs(trimmed - s_ref), axis=1)
    start_idx = np.arange(n_win) * win_steps
    sigma_win = sigma_series[start_idx]
    return eps_price, sigma_win, s_ref[:, 0]


def frozen_epoch_drift(prices: np.ndarray, sample_every: int) -> tuple[np.ndarray, np.ndarray]:
    """Never-refreshed reference: |S_t - S_0| with S_0 the segment start, sampled
    along the segment. Returns (elapsed_seconds, drift_usd)."""
    idx = np.arange(0, prices.size, sample_every)
    elapsed = idx * (BOOK_GRID_MS / 1000.0)
    drift = np.abs(prices[idx] - prices[0])
    return elapsed, drift


def regress_sqrt_dt(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS y = slope * x + intercept; return (slope, intercept, R^2). x = sigma*sqrt(dt)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return float("nan"), float("nan"), float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, r2


@dataclass
class DtResult:
    dt_s: float
    n_windows: int
    rel_p50_bps: float  # HEADLINE metric: 1e4 * eps_price / S_ref (S_ref is where
    rel_p99_bps: float  # the freeze error actually enters: the funding drain F=S_ref*f)
    eps_p99_ticks: float  # raw price move (informational; NOT the approximation error)
    eps_usd_p99: float    # eps_price_p99 * |q| * |f| (informational USD drain error)
    slope: float  # eps_price vs sigma*sqrt(dt)  (functional-form check)
    r2: float


def analyse(segments: list[np.ndarray], cfg: SrefConfig) -> list[DtResult]:
    sigma_per_seg = [rolling_sigma(seg) for seg in segments]
    results: list[DtResult] = []
    for dt_s in cfg.dt_grid_s:
        win_steps = max(2, int(round(dt_s / (BOOK_GRID_MS / 1000.0))) + 1)
        all_eps = []
        all_x = []  # sigma*sqrt(dt)
        all_rel = []  # 1e4 * eps / S_ref (bps)
        for seg, sig_series in zip(segments, sigma_per_seg):
            eps, sig, s_ref = window_errors(seg, win_steps, sig_series)
            if eps.size:
                all_eps.append(eps)
                all_x.append(sig * math.sqrt(dt_s))
                all_rel.append(eps / s_ref * 1.0e4)
        if not all_eps:
            continue
        eps = np.concatenate(all_eps)
        x = np.concatenate(all_x)
        rel = np.concatenate(all_rel)
        slope, _intercept, r2 = regress_sqrt_dt(x, eps)
        eps_p99 = float(np.percentile(eps, 99))
        rel_p50, rel_p99 = np.percentile(rel, [50, 99])
        results.append(
            DtResult(
                dt_s=dt_s,
                n_windows=eps.size,
                rel_p50_bps=rel_p50,
                rel_p99_bps=rel_p99,
                eps_p99_ticks=eps_p99 / cfg.tick,
                eps_usd_p99=eps_p99 * cfg.abs_q * cfg.abs_f,
                slope=slope,
                r2=r2,
            )
        )
    return results


def stress_result(segments: list[np.ndarray], cfg: SrefConfig) -> list[DtResult] | None:
    """Re-run the full dt-grid on the single highest-sigma segment (cascade proxy)."""
    if not segments:
        return None
    # rank segments by overall realised vol
    def seg_sigma(seg: np.ndarray) -> float:
        incr = np.diff(seg)
        return float(np.std(incr)) if incr.size else 0.0

    worst = max(segments, key=seg_sigma)
    return analyse([worst], cfg)


HEADLINE_DT_S = BOOK_GRID_MS / 1000.0  # 0.1 s
PASS_BAR_BPS = 10.0  # pre-stated: p99 relative freeze error must stay below this


def print_report(results: list[DtResult], stress: list[DtResult], cfg: SrefConfig) -> None:
    print("\n=== S_ref affine-approximation error (Justification III) ===")
    print(f"HEADLINE metric: relative freeze error 1e4*eps/S_ref (bps) at dt={HEADLINE_DT_S:g}s "
          f"(the 100 ms requote horizon). eps_usd uses |q|={cfg.abs_q}, |f|={cfg.abs_f:.2e}.")
    hdr = (f"{'dt(s)':>6} {'n_win':>9} {'relp50(bps)':>11} {'relp99(bps)':>11} "
           f"{'p99(USD)':>10} {'eps_p99(tk)':>11} {'slope':>7} {'R^2':>6}")
    print(hdr)

    def _row(r: DtResult) -> str:
        return (f"{r.dt_s:>6.2f} {r.n_windows:>9d} {r.rel_p50_bps:>11.3f} {r.rel_p99_bps:>11.3f} "
                f"{r.eps_usd_p99:>10.2e} {r.eps_p99_ticks:>11.1f} {r.slope:>7.3f} {r.r2:>6.3f}")

    for r in results:
        print(_row(r))
    if stress:
        print("\n-- stress (highest-sigma segment) --")
        print(hdr)
        for r in stress:
            print(_row(r))
    # headline verdict, on the 100 ms requote horizon only
    head = [r for r in results if abs(r.dt_s - HEADLINE_DT_S) < 1e-9]
    if head:
        r = head[0]
        verdict = "PASS" if r.rel_p99_bps < PASS_BAR_BPS else "FAIL"
        print(
            f"\nHeadline (dt={HEADLINE_DT_S:g}s requote horizon): p99 relative freeze error "
            f"= {r.rel_p99_bps:.3f} bps (bar {PASS_BAR_BPS:g} bps -> {verdict}); "
            f"sqrt-dt scaling slope={r.slope:.3f}, R^2={r.r2:.3f}."
        )


def per_day_spread(
    per_day: list[tuple[str, list[np.ndarray]]], cfg: SrefConfig, dt_s: float = HEADLINE_DT_S
) -> list[dict]:
    """Per-day p99 relative freeze error (bps) at the 100 ms requote horizon, so
    the pooled headline can be backed by 'it holds on EVERY pre-analysis day'."""
    rows: list[dict] = []
    for fday, segs in per_day:
        res = analyse([px for _ts, px in segs], cfg)
        r = next((rr for rr in res if abs(rr.dt_s - dt_s) < 1e-9), None)
        if r is None:
            continue
        rows.append({"fday": fday, "n_windows": r.n_windows,
                     "p99_bps": r.rel_p99_bps, "slope": r.slope, "r2": r.r2})
    return rows


def print_day_spread(rows: list[dict], cfg: SrefConfig, dt_s: float = HEADLINE_DT_S) -> None:
    print(f"\n=== Per-day robustness (dt={dt_s:g}s p99 relative error, "
          f"{len(rows)} pre-analysis days) ===")
    if not rows:
        print("  (no per-day results)")
        return
    p99s = np.array([r["p99_bps"] for r in rows])
    worst = max(rows, key=lambda r: r["p99_bps"])
    n_pass = int(np.sum(p99s < PASS_BAR_BPS))
    print(f"  p99 (bps): min={p99s.min():.3f}  median={np.median(p99s):.3f}  "
          f"max={p99s.max():.3f}  (worst day {worst['fday']})")
    print(f"  pre-stated rule (p99 < {PASS_BAR_BPS:g} bps every day): {n_pass}/{len(rows)} "
          f"days pass -> {'PASS' if n_pass == len(rows) else 'FAIL'}")


def spectral_gap_report(p: HJBParams) -> dict:
    """Compute one-over-g, the spectral-gap relaxation time of the autonomous M."""
    g = intrinsic_relaxation_rate(p)
    tau_g = 1.0 / g if g > 0 else float("inf")
    delta_t_quote_s = BOOK_GRID_MS / 1000.0
    practical_ratio = math.sqrt(tau_g / delta_t_quote_s) if tau_g < float("inf") \
        else float("inf")
    return {
        "spectral_gap_g_per_s": g,
        "tau_g_s": tau_g,
        "delta_t_quote_s": delta_t_quote_s,
        "practical_jump_over_drain_ratio": practical_ratio,
    }


def print_spectral_gap(rep: dict, p: HJBParams) -> None:
    print("\n=== Spectral-gap practical-influence window (justification III closing) ===")
    print(
        f"HJB params used: gamma={p.gamma:.2e}, sigma={p.sigma:.3g} USD/sqrt(s), "
        f"A={p.A:.3g}, k={p.k:.3g}, alpha_ml={p.alpha_ml:.2e}, Q={p.Q}"
    )
    print(
        f"  g = lambda_1 - lambda_0 = {rep['spectral_gap_g_per_s']:.4e} 1/s"
    )
    print(
        f"  1/g (practical influence window) = {rep['tau_g_s']:.2f} s"
    )
    print(
        f"  per-quote practical ratio sqrt(1/g / Delta_t) at Delta_t="
        f"{rep['delta_t_quote_s']:.3f}s: {rep['practical_jump_over_drain_ratio']:.2f}"
    )
    print(
        "  -> 1/g and the practical ratio "
        "(rejected-Sref-at-settlement paragraph) before compile."
    )


def build_boundary_errors(segments: list[Segment], tau_g_s: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """One (e_cont, e_disc) pair of bps error arrays per funding boundary layer [T - tau_g, T]."""
    tau_ms = int(round(tau_g_s * 1000))
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    for ts, px in segments:
        if ts.size < 3:
            continue
        for T in settlements_in(int(ts[0]), int(ts[-1])):
            hiT = int(np.searchsorted(ts, T, side="right")) - 1
            if hiT < 1 or (T - int(ts[hiT])) > 1000:  # settlement missing / in a gap
                continue
            lo = int(np.searchsorted(ts, T - tau_ms, side="left"))
            if hiT - lo < 10:
                continue
            idx = np.arange(lo, hiT)
            S = px[idx]
            e_cont = (px[idx + 1] - S) / S * 1.0e4
            e_disc = (S - px[hiT]) / S * 1.0e4
            layers.append((e_cont, e_disc))
    return layers


def _pooled_lag1(layers: list[tuple[np.ndarray, np.ndarray]], which: int) -> float:
    """Lag-1 autocorrelation pooled across layers (demeaned per layer, so it never
    crosses a settlement boundary)."""
    num = den = 0.0
    for pair in layers:
        a = pair[which]
        if a.size > 1:
            a = a - a.mean()
            num += float(np.dot(a[:-1], a[1:]))
            den += float(np.dot(a, a))
    return num / den if den > 0 else float("nan")


def error_comparison(layers: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Median + p99 magnitude (bps) and lag-1 correlation of the continuous and
    discrete error arrays, all from the two paired arrays of build_boundary_errors."""
    if not layers:
        return {"n_layers": 0, "n_requotes": 0}
    cont = np.abs(np.concatenate([p[0] for p in layers]))
    disc = np.abs(np.concatenate([p[1] for p in layers]))
    c50, c99 = (float(x) for x in np.percentile(cont, [50, 99]))
    d50, d99 = (float(x) for x in np.percentile(disc, [50, 99]))
    return {
        "n_layers": len(layers),
        "n_requotes": int(cont.size),
        "cont_p50_bps": c50, "cont_p99_bps": c99, "cont_lag1": _pooled_lag1(layers, 0),
        "disc_p50_bps": d50, "disc_p99_bps": d99, "disc_lag1": _pooled_lag1(layers, 1),
        "ratio_p50": (d50 / c50) if c50 > 1e-9 else None,
        "ratio_p99": (d99 / c99) if c99 > 1e-9 else None,
    }


def print_error_comparison(ec: dict) -> None:
    print("\n=== Continuous vs discrete S_ref error (boundary-layer requotes, both "
          "refreshed every 100 ms) ===")
    if ec.get("n_requotes", 0) == 0:
        print("  (no funding boundary layers in range)")
        return
    print(f"  {ec['n_layers']} settlements, {ec['n_requotes']} requotes")
    print(f"  continuous (next 100 ms move): p50={ec['cont_p50_bps']:.3f}  "
          f"p99={ec['cont_p99_bps']:.3f} bps  lag-1 corr={ec['cont_lag1']:+.3f}")
    print(f"  discrete  (gap to S_Tfund)   : p50={ec['disc_p50_bps']:.3f}  "
          f"p99={ec['disc_p99_bps']:.3f} bps  lag-1 corr={ec['disc_lag1']:+.3f}")
    r50 = "n/a (continuous p50=0)" if ec["ratio_p50"] is None else f"{ec['ratio_p50']:.1f}x"
    r99 = "n/a" if ec["ratio_p99"] is None else f"{ec['ratio_p99']:.1f}x"
    print(f"  ratio discrete/continuous: p50 {r50}  p99 {r99}")


def crosscheck_production_rho(g: float, outdir: Path) -> None:
    """Warn if the recomputed spectral gap diverges from describe_rho.json's
    production spectral_gap_queue_aware. This is the single source of truth for
    rho; the two must agree or we quote two different boundary-layer
    times (the exact bug the sigma=8.3/Q=100 defaults used to cause)."""
    import json

    path = outdir / "describe_rho.json"
    if not path.exists():
        return
    try:
        alt = json.loads(path.read_text()).get("alternatives", {})
        g_prod = float(alt["spectral_gap_queue_aware"])
    except Exception:
        return
    rel = abs(g - g_prod) / g_prod if g_prod else float("inf")
    if rel > 0.05:
        print(
            f"\nWARN: spectral gap g={g:.4e} differs from describe_rho "
            f"spectral_gap_queue_aware={g_prod:.4e} by {100*rel:.1f}%; the HJB "
            f"params here no longer match the production rho calibration."
        )
    else:
        print(
            f"\n[cross-check] g={g:.4e} matches describe_rho production "
            f"spectral_gap_queue_aware={g_prod:.4e} (rel diff {100*rel:.2f}%)."
        )


def to_latex(results: list[DtResult], cfg: SrefConfig) -> str:
    lines = [
        r"% Auto-generated by validate_sref_approx.py",
        r"% S_ref affine-approximation error (Justification III).",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$\Delta t$ (s) & $n_{\text{win}}$ & $\varepsilon^{\text{rel}}_{p50}$ (bps) & "
        r"$\varepsilon^{\text{rel}}_{p99}$ (bps) & slope ($\sigma\sqrt{\Delta t}$) & $R^2$ \\",
        r"\midrule",
    ]
    for r in results:
        lines.append(
            f"{r.dt_s:.2f} & {r.n_windows} & {r.rel_p50_bps:.3f} & "
            f"{r.rel_p99_bps:.3f} & {r.slope:.3f} & {r.r2:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def make_figure(segments: list[np.ndarray], results: list[DtResult], cfg: SrefConfig, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) sqrt-dt scaling scatter at dt=1s on the first segment
    dt_s = 1.0
    win_steps = max(2, int(round(dt_s / (BOOK_GRID_MS / 1000.0))) + 1)  # closed-window sample count (see analyse)
    seg = max(segments, key=lambda s: s.size)
    eps, sig, _s_ref = window_errors(seg, win_steps)
    x = sig * math.sqrt(dt_s)
    ax0.scatter(x, eps, s=6, alpha=0.3)
    if x.size > 2:
        slope, intercept, r2 = regress_sqrt_dt(x, eps)
        xs = np.linspace(0, np.nanpercentile(x, 99), 50)
        ax0.plot(xs, slope * xs + intercept, "C3", lw=2, label=f"slope={slope:.2f}, $R^2$={r2:.2f}")
        ax0.legend()
    ax0.set_xlabel(r"$\sigma_{\rm win}\sqrt{\Delta t}$ (USD)")
    ax0.set_ylabel(r"$\varepsilon_{\rm price}$ (USD)")
    ax0.set_title(r"$\varepsilon$ vs $\sigma\sqrt{\Delta t}$ ($\Delta t$=1s)")
    ax0.grid(True, alpha=0.3)

    elapsed, drift = frozen_epoch_drift(seg, sample_every=win_steps)
    ax1.plot(elapsed, drift / cfg.tick, label="frozen for whole epoch", color="C1")
    n_win = seg.size // win_steps
    win_eps, _sig, _s_ref = window_errors(seg, win_steps)
    win_times = np.arange(n_win) * win_steps * (BOOK_GRID_MS / 1000.0)
    ax1.plot(win_times[: win_eps.size], win_eps / cfg.tick, label="refreshed each $\\Delta t$", color="C2", alpha=0.8)
    ax1.axhline(1.0, color="k", ls="--", lw=0.8, label="1 tick")
    ax1.set_xlabel("elapsed time (s)")
    ax1.set_ylabel(r"reference error (ticks)")
    ax1.set_title("Non-accumulation: refresh vs frozen")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preanalysis", action="store_true",
                    help="sweep all pre-analysis funding days")
    ap.add_argument("--base", type=str, default=DEFAULT_BASE, help="dataset base dir (real mode)")
    ap.add_argument("--fday", type=str, default=None, help="single funding day YYYY-MM-DD (spot check)")
    ap.add_argument("--abs-q", type=int, default=10, help="representative |q| for USD conversion")
    ap.add_argument("--abs-f", type=float, default=SrefConfig.abs_f,
                    help="representative |settlement funding rate|; scales "
                         "eps_usd only")
    ap.add_argument("--tick", type=float, default=TICK_SIZE_USD)
    ap.add_argument("--outdir", type=str, default="reports")
    ap.add_argument("--hjb-gamma", type=float, default=2.0e-5)
    ap.add_argument("--hjb-sigma", type=float, default=4.5748,
                    help="production 60 s realised vol sigma_rep, the HJB feed")
    ap.add_argument("--hjb-A", type=float, default=0.274,
                    help="AS base intensity per side (queue-aware grid-30 calibration)")
    ap.add_argument("--hjb-k", type=float, default=0.090,
                    help="AS decay (queue-aware grid-30 calibration)")
    ap.add_argument("--hjb-alpha-ml", type=float, default=0.0)
    ap.add_argument("--hjb-Q", type=int, default=10,
                    help="production inventory cap")
    args = ap.parse_args()

    cfg = SrefConfig(abs_q=args.abs_q, abs_f=args.abs_f, tick=args.tick)
    hjb_p = HJBParams(
        gamma=args.hjb_gamma,
        sigma=args.hjb_sigma,
        A=args.hjb_A,
        k=args.hjb_k,
        alpha_ml=args.hjb_alpha_ml,
        Q=args.hjb_Q,
    )

    per_day: list[tuple[str, list[Segment]]] | None = None
    if args.preanalysis:
        print(f"pre-analysis sweep (30 days), base={args.base}")
        per_day = load_preanalysis_segments(Path(args.base))
        segments = [s for _, segs in per_day for s in segs]
        print(f"  {len(per_day)} pre-analysis day(s) with data.")
    else:
        if not args.fday:
            ap.error("requires --fday YYYY-MM-DD or --preanalysis")
        print(f"base={args.base}, fday={args.fday}")
        segments = load_real_segments(Path(args.base), args.fday)

    px_segments = [px for _ts, px in segments]  # price-only view for the headline
    print(f"Loaded {len(segments)} valid segment(s); total samples = "
          f"{sum(px.size for px in px_segments)}.")
    results = analyse(px_segments, cfg)
    stress = stress_result(px_segments, cfg)
    print_report(results, stress, cfg)

    day_rows = per_day_spread(per_day, cfg) if per_day is not None else []
    if per_day is not None:
        print_day_spread(day_rows, cfg)

    sg = spectral_gap_report(hjb_p)
    print_spectral_gap(sg, hjb_p)
    crosscheck_production_rho(sg["spectral_gap_g_per_s"], Path(args.outdir))

    layers = build_boundary_errors(segments, sg["tau_g_s"])
    ec = error_comparison(layers)
    print_error_comparison(ec)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "sref_approx_table.tex").write_text(to_latex(results, cfg) + "\n")
    if per_day is not None:
        import json as _json
        p99s = [r["p99_bps"] for r in day_rows]
        (outdir / "sref_preanalysis_per_day.json").write_text(_json.dumps({
            "dt_s": HEADLINE_DT_S,
            "metric": "p99 relative freeze error 1e4*eps/S_ref (bps)",
            "n_days": len(day_rows),
            "rule": f"p99 < {PASS_BAR_BPS:g} bps on every pre-analysis day",
            "n_pass": int(sum(p < PASS_BAR_BPS for p in p99s)),
            "p99_min": min(p99s) if p99s else None,
            "p99_median": float(np.median(p99s)) if p99s else None,
            "p99_max": max(p99s) if p99s else None,
            "days": day_rows,
        }, indent=2) + "\n")
    import json
    (outdir / "sref_spectral_gap.json").write_text(json.dumps({
        **sg,
        "error_comparison": ec,
        "hjb_params": {
            "gamma": hjb_p.gamma, "sigma": hjb_p.sigma, "A": hjb_p.A,
            "k": hjb_p.k, "alpha_ml": hjb_p.alpha_ml, "Q": hjb_p.Q,
        },
    }, indent=2) + "\n")
    make_figure(px_segments, results, cfg, outdir / "sref_approx.png")
    extra = (f", {outdir/'sref_preanalysis_per_day.json'}"
             if per_day is not None else "")
    print(
        f"\nWrote {outdir/'sref_approx_table.tex'}, {outdir/'sref_spectral_gap.json'}, "
        f"{outdir/'sref_approx.png'}{extra}."
    )


if __name__ == "__main__":
    main()
