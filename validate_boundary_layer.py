"""Validation of the pre-settlement boundary-layer form."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

try:
    import pipeline_utils as pu

    _HAVE_PU = True
except Exception:
    _HAVE_PU = False

try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan

    _HAVE_SM = True
except Exception:
    _HAVE_SM = False

from scipy.optimize import curve_fit

DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"


@dataclass
class BLConfig:
    tau_max_s: int = 1800
    bin_s: int = 1
    drop_stale: bool = True


def _date_range(start: str, end: str) -> list[str]:
    d0 = pu.parse_date(start)
    d1 = pu.parse_date(end)
    out = []
    d = d0
    while d <= d1:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def load_markprice_span(base: Path, start: str, end: str) -> pl.DataFrame:
    """Concatenate markprice_ffill_*.parquet over [start-1day, end] (the extra
    leading day covers 00:00 UTC settlement windows). Returns ts_ms, MarkPrice,
    IndexPrice, is_stale, sorted by ts_ms."""
    if not _HAVE_PU:
        raise RuntimeError("pipeline_utils unavailable")
    lead = (pu.parse_date(start) - timedelta(days=1)).strftime("%Y-%m-%d")
    dates = [lead] + _date_range(start, end)
    frames = []
    for d in dates:
        for h in range(24):
            p = base / d / f"markprice_ffill_{h:02d}h.parquet"
            if p.exists():
                frames.append(
                    pl.read_parquet(p).select(["ts_ms", "MarkPrice", "IndexPrice", "is_stale"])
                )
    if not frames:
        raise RuntimeError(f"no markprice_ffill files under {base} for {start}..{end}")
    return pl.concat(frames, how="vertical_relaxed").sort("ts_ms").unique("ts_ms", keep="first")


def settlement_windows(
    df: pl.DataFrame, start: str, end: str, cfg: BLConfig
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """For each funding settlement in [start, end], return
    (settlement_id, tau_seconds, delta) for the pre-settlement window, where
    delta = |MarkPrice - IndexPrice| / IndexPrice. Drops stale / non-positive."""
    sett_ids: list[tuple[int, np.ndarray, np.ndarray]] = []
    sid = 0
    for d in _date_range(start, end):
        for t_fund in pu.funding_settlement_times(d):
            lo = t_fund - cfg.tau_max_s * 1000
            w = df.filter((pl.col("ts_ms") >= lo) & (pl.col("ts_ms") < t_fund))
            if cfg.drop_stale and "is_stale" in w.columns:
                w = w.filter(~pl.col("is_stale").fill_null(True))
            w = w.filter(pl.col("IndexPrice") > 0)
            if w.height < 30:
                continue
            ts = w["ts_ms"].to_numpy()
            mark = w["MarkPrice"].to_numpy()
            index = w["IndexPrice"].to_numpy()
            delta = np.abs(mark - index) / index
            tau = (t_fund - ts) / 1000.0
            ok = np.isfinite(delta) & (delta > 0)
            if ok.sum() < 30:
                continue
            sett_ids.append((sid, tau[ok], delta[ok]))
            sid += 1
    return sett_ids


def bin_curve(tau: np.ndarray, delta: np.ndarray, cfg: BLConfig) -> tuple[np.ndarray, np.ndarray]:
    """Mean delta per tau bin. Returns (bin_centres, mean_delta)."""
    edges = np.arange(0, cfg.tau_max_s + cfg.bin_s, cfg.bin_s)
    idx = np.clip(np.digitize(tau, edges) - 1, 0, edges.size - 2)
    sums = np.bincount(idx, weights=delta, minlength=edges.size - 1)
    cnts = np.bincount(idx, minlength=edges.size - 1)
    centres = edges[:-1] + cfg.bin_s / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnts > 0, sums / cnts, np.nan)
    ok = cnts > 0
    return centres[ok], mean[ok]


def pooled_exp_fit(windows, cfg: BLConfig) -> dict:
    """Pooled log-linear fit ln(delta) = c_k - rho*tau with settlement fixed
    effects. Returns rho, HAC SE, within-R^2, per-settlement R^2 list, and the
    pooled residuals/tau for the heteroscedasticity test."""
    taus, lns, dummies, sids = [], [], [], []
    per_sett_r2 = []
    for sid, tau, delta in windows:
        cen, mean = bin_curve(tau, delta, cfg)
        ln = np.log(mean)
        # per-settlement OLS for R^2
        if cen.size >= 5:
            A = np.vstack([cen, np.ones_like(cen)]).T
            coef, *_ = np.linalg.lstsq(A, ln, rcond=None)
            yhat = A @ coef
            ss_res = np.sum((ln - yhat) ** 2)
            ss_tot = np.sum((ln - ln.mean()) ** 2)
            per_sett_r2.append(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)
        taus.append(cen)
        lns.append(ln)
        sids.append(np.full(cen.size, sid))
    tau_all = np.concatenate(taus)
    ln_all = np.concatenate(lns)
    sid_all = np.concatenate(sids)

    result: dict = {"per_sett_r2": np.array(per_sett_r2)}

    if _HAVE_SM:
        uniq = np.unique(sid_all)
        dmat = np.column_stack(
            [tau_all] + [(sid_all == s).astype(float) for s in uniq[1:]]
        )
        dmat = sm.add_constant(dmat)
        model = sm.OLS(ln_all, dmat)
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": 30})
        rho = -float(res.params[1]) 
        rho_se = float(res.bse[1])
        # within-R^2: regress out FE then correlate
        result.update(rho=rho, rho_se=rho_se, r2=float(res.rsquared))
        result["resid"] = np.asarray(res.resid)
        result["tau_all"] = tau_all
    else:
        ln_dm = ln_all.copy()
        tau_dm = tau_all.copy()
        for s in np.unique(sid_all):
            m = sid_all == s
            ln_dm[m] -= ln_all[m].mean()
            tau_dm[m] -= tau_all[m].mean()
        slope = np.sum(tau_dm * ln_dm) / np.sum(tau_dm ** 2)
        rho = -slope
        resid = ln_dm - slope * tau_dm
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum(ln_dm ** 2)
        result.update(
            rho=rho,
            rho_se=float("nan"),
            r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            resid=resid,
            tau_all=tau_all,
        )
    return result


def aic_model_comparison(windows, cfg: BLConfig) -> dict:
    """Fit exp / power / double-exp to the across-settlement MEAN curve g_bar(tau);
    return AIC for each. Lower AIC = better."""
    grid = np.arange(cfg.bin_s / 2.0, cfg.tau_max_s, cfg.bin_s)
    acc = np.zeros(grid.size)
    cnt = np.zeros(grid.size)
    for _sid, tau, delta in windows:
        cen, mean = bin_curve(tau, delta, cfg)
        j = np.clip(((cen - cfg.bin_s / 2.0) / cfg.bin_s).round().astype(int), 0, grid.size - 1)
        acc[j] += mean
        cnt[j] += 1
    ok = cnt > 0
    tau = grid[ok]
    g = acc[ok] / cnt[ok]
    g = np.clip(g, 1e-300, None)
    n = tau.size

    def aic(rss: float, k: int) -> float:
        return n * math.log(rss / n) + 2 * k

    out: dict = {}
    # exp: ln g = c - rho tau
    A = np.vstack([tau, np.ones_like(tau)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(g), rcond=None)
    rss_exp = float(np.sum((np.log(g) - A @ coef) ** 2))
    out["exp"] = {"aic": aic(rss_exp, 2), "rho": -float(coef[0])}

    # power: ln g = c - alpha ln tau  (drop tau<=0)
    pos = tau > 0
    Ap = np.vstack([np.log(tau[pos]), np.ones(pos.sum())]).T
    cp, *_ = np.linalg.lstsq(Ap, np.log(g[pos]), rcond=None)
    rss_pow = float(np.sum((np.log(g[pos]) - Ap @ cp) ** 2))
    out["power"] = {"aic": aic(rss_pow, 2), "alpha": -float(cp[0])}

    # double-exp: g = a1 e^{-r1 tau} + a2 e^{-r2 tau}  (fit in linear space)
    def dexp(t, a1, r1, a2, r2):
        return a1 * np.exp(-r1 * t) + a2 * np.exp(-r2 * t)

    try:
        g0 = g[0]
        p0 = [g0 * 0.6, 1 / 300.0, g0 * 0.4, 1 / 1200.0]
        popt, _ = curve_fit(dexp, tau, g, p0=p0, maxfev=20000)
        rss_de = float(np.sum((g - dexp(tau, *popt)) ** 2))
        g_exp = np.exp(A @ coef)
        rss_exp_lin = float(np.sum((g - g_exp) ** 2))
        g_pow = np.full_like(g, np.nan)
        g_pow[pos] = np.exp(Ap @ cp)
        rss_pow_lin = float(np.nansum((g[pos] - g_pow[pos]) ** 2))
        out["exp"]["aic_linspace"] = aic(rss_exp_lin, 2)
        out["power"]["aic_linspace"] = aic(rss_pow_lin, 2)
        out["double_exp"] = {"aic_linspace": aic(rss_de, 4), "params": popt.tolist()}
    except Exception as e:  # pragma: no cover
        out["double_exp"] = {"error": str(e)}
    return out


def bp_heteroscedasticity(fit: dict) -> float | None:
    """Breusch-Pagan p-value of residuals on tau. Low p => heteroscedastic =>
    the pooled OLS is biased near tau->0 and a weighted estimator is warranted."""
    if not _HAVE_SM:
        return None
    resid = fit.get("resid")
    tau = fit.get("tau_all")
    if resid is None or tau is None:
        return None
    exog = sm.add_constant(tau)
    try:
        lm, lm_p, f, f_p = het_breuschpagan(resid, exog)
        return float(lm_p)
    except Exception:
        return None


def fit_intensity_signal(*_args, **_kwargs):
    raise NotImplementedError(
        
    )


def fit_markout_signal(*_args, **_kwargs):
    raise NotImplementedError(
        
    )


def print_report(fit: dict, aic: dict, bp_p: float | None, n_sett: int) -> None:
    print("\n=== Boundary-layer decay validation, g1 = |Delta_t| ===")
    print(f"settlements used: {n_sett}")
    print(f"pooled rho_hat   = {fit['rho']:.6e} s^-1  (SE={fit['rho_se']:.2e})  -> 1/rho = {1.0/fit['rho']:.1f} s")
    print(f"pooled within-R^2= {fit['r2']:.4f}")
    r2s = fit["per_sett_r2"]
    r2s = r2s[np.isfinite(r2s)]
    if r2s.size:
        print(f"per-settlement R^2: median={np.median(r2s):.3f}  min={r2s.min():.3f}  max={r2s.max():.3f}")
    print("\nAIC model comparison on g_bar(tau):")
    for name in ("exp", "power", "double_exp"):
        m = aic.get(name, {})
        a = m.get("aic_linspace", m.get("aic"))
        extra = ""
        if name == "exp" and "rho" in m:
            extra = f"  (rho={m['rho']:.3e})"
        if name == "power" and "alpha" in m:
            extra = f"  (alpha={m['alpha']:.3f})"
        print(f"  {name:>11}: AIC={a:.2f}{extra}" if a is not None else f"  {name:>11}: n/a")
    # verdict
    a_exp = aic.get("exp", {}).get("aic_linspace")
    a_pow = aic.get("power", {}).get("aic_linspace")
    if a_exp is not None and a_pow is not None:
        if a_pow + 2 < a_exp:
            print("\n  power law fits better than exponential")
        else:
            print("\n  exponential competitive with the power law")
    if bp_p is not None:
        flag = " (heteroscedastic: consider WLS near tau->0)" if bp_p < 0.05 else ""
        print(f"Breusch-Pagan p-value on residuals: {bp_p:.3e}{flag}")


def make_figure(windows, fit: dict, cfg: BLConfig, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    for sid, tau, delta in windows[:40]:
        cen, mean = bin_curve(tau, delta, cfg)
        ax0.plot(cen, np.log(mean), color="C0", alpha=0.15, lw=0.8)
    # overlay fitted exponential slope (use mean intercept for placement)
    cen_grid = np.linspace(0, cfg.tau_max_s, 100)
    all_ln0 = []
    for _sid, tau, delta in windows:
        cen, mean = bin_curve(tau, delta, cfg)
        if cen.size:
            all_ln0.append(np.log(mean[0]))
    c0 = float(np.mean(all_ln0)) if all_ln0 else 0.0
    ax0.plot(cen_grid, c0 - fit["rho"] * cen_grid, "C3", lw=2, label=f"fit: rho={fit['rho']:.2e}")
    ax0.set_xlabel(r"$\tau = T_{fund} - t$ (s)")
    ax0.set_ylabel(r"$\ln |\Delta_t|$")
    ax0.set_title("Boundary-layer decay across settlements")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # residual Q-Q
    resid = fit.get("resid")
    if resid is not None and resid.size:
        from scipy import stats

        stats.probplot(resid, dist="norm", plot=ax1)
        ax1.set_title("Residual Q-Q (Gaussian)")
        ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def to_latex(fit: dict, aic: dict, n_sett: int) -> str:
    a_exp = aic.get("exp", {}).get("aic_linspace")
    a_pow = aic.get("power", {}).get("aic_linspace")
    a_de = aic.get("double_exp", {}).get("aic_linspace")
    lines = [
        r"% Auto-generated by validate_boundary_layer.py (signal g1 = |Delta_t|).",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        f"Settlements & {n_sett} \\\\",
        f"$\\hat\\rho_{{|\\Delta|}}$ (s$^{{-1}}$) & {fit['rho']:.4e} \\\\",
        f"SE($\\hat\\rho$) & {fit['rho_se']:.2e} \\\\",
        f"within-$R^2$ & {fit['r2']:.4f} \\\\",
        f"AIC exp & {a_exp:.2f} \\\\" if a_exp is not None else "AIC exp & -- \\\\",
        f"AIC power & {a_pow:.2f} \\\\" if a_pow is not None else "AIC power & -- \\\\",
        f"AIC double-exp & {a_de:.2f} \\\\" if a_de is not None else "AIC double-exp & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=str, default=DEFAULT_BASE)
    ap.add_argument("--start", type=str, default=None, help="first training calendar date YYYY-MM-DD")
    ap.add_argument("--end", type=str, default=None, help="last training calendar date YYYY-MM-DD")
    ap.add_argument("--tau-max", type=int, default=1800)
    ap.add_argument("--bin-s", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="reports")
    args = ap.parse_args()

    cfg = BLConfig(tau_max_s=args.tau_max, bin_s=args.bin_s)

    if not _HAVE_SM:
        print("WARN: statsmodels unavailable; using the numpy OLS fallback (no HAC SE / BP test).")

    if not (args.start and args.end):
        ap.error("requires --start and --end")
    print(f"base={args.base}, {args.start}..{args.end}")
    df = load_markprice_span(Path(args.base), args.start, args.end)
    windows = settlement_windows(df, args.start, args.end, cfg)

    print(f"settlements with usable pre-window: {len(windows)}")
    if len(windows) < 3:
        raise SystemExit("too few settlements to fit; widen the date range.")

    fit = pooled_exp_fit(windows, cfg)
    aic = aic_model_comparison(windows, cfg)
    bp_p = bp_heteroscedasticity(fit)
    print_report(fit, aic, bp_p, len(windows))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "boundary_layer_table.tex").write_text(to_latex(fit, aic, len(windows)) + "\n")
    make_figure(windows, fit, cfg, outdir / "boundary_layer.png")
    print(f"\nWrote {outdir/'boundary_layer_table.tex'} and {outdir/'boundary_layer.png'}.")


if __name__ == "__main__":
    main()
