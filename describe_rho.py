"""Selection of the boundary-layer rate rho on the pre-analysis days."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from datetime import timedelta
from pathlib import Path

import numpy as np

try:
    import polars as pl
    _HAVE_PL = True
except Exception:  # pragma: no cover
    _HAVE_PL = False

try:
    import pipeline_utils as pu
    _HAVE_PU = True
except Exception:  # pragma: no cover
    _HAVE_PU = False

from validate_boundary_layer import BLConfig, bin_curve, _date_range


DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
DEFAULT_OUT_DIR = Path("reports")
DEFAULT_OUT_STEM = "describe_rho"
THEORETICAL_ANCHOR_RHO = 1.0 / 180.0  # 3-minute e-folding fallback
MIN_R2_FOR_FIT = 0.20


SIGNAL_DESCRIPTIONS = {
    "S_f":    ("|FundingRate(t)|",
               "peaked",  # signal-shape class (peaked-at-zero vs convergence)
               "current calibrate_rho primary"),
    "S_pid":  ("|MarkPrice - IndexPrice| / IndexPrice",
               "peaked",
               "original primary"),
    "S_conv": ("|FundingRate(t) - FundingRate(T_fund)|",
               "convergence",
               "convergence-to-settlement of the funding-rate forecast"),
}


def _settle_value(values: np.ndarray, tau: np.ndarray) -> float:
    """Best estimate of the value AT T_fund: the tick with the smallest tau."""
    if values.size == 0:
        return float("nan")
    return float(values[int(np.argmin(tau))])


def settlement_signals(funding: "pl.DataFrame",
                       markprice: "pl.DataFrame",
                       fdays: list[str], cfg: BLConfig) -> list[dict]:
    """For each funding day in `fdays`, extract the three settlements that
    fall within that 04:00-04:00 window (08:00, 16:00, and 00:00 of the next
    day). This avoids using data before Jan 19 00:00 for the first day and
    ensures exactly 3*len(fdays) settlements are analyzed."""
    recs: list[dict] = []
    sid = 0
    for d in fdays:
        d_dt = pu.parse_date(d)
        s1 = int(d_dt.replace(hour=8).timestamp() * 1000)
        s2 = int(d_dt.replace(hour=16).timestamp() * 1000)
        s3 = int((d_dt + timedelta(days=1)).replace(hour=0).timestamp() * 1000)

        for t_fund in [s1, s2, s3]:
            lo = t_fund - cfg.tau_max_s * 1000

            # funding-rate signals
            wf = funding.filter((pl.col("ts_ms") >= lo) & (pl.col("ts_ms") < t_fund))
            if cfg.drop_stale and "is_stale" in wf.columns:
                wf = wf.filter(~pl.col("is_stale").fill_null(True))
            wf = wf.filter(pl.col("FundingRate").is_not_null())

            # markprice signals
            wm = markprice.filter((pl.col("ts_ms") >= lo) & (pl.col("ts_ms") < t_fund))
            if cfg.drop_stale and "is_stale" in wm.columns:
                wm = wm.filter(~pl.col("is_stale").fill_null(True))
            wm = wm.filter(pl.col("IndexPrice") > 0)

            if wf.height < 30 or wm.height < 30:
                continue

            # funding side
            ts_f = wf["ts_ms"].to_numpy()
            f_signed = wf["FundingRate"].to_numpy().astype(float)
            tau_f = (t_fund - ts_f) / 1000.0
            f_T = _settle_value(f_signed, tau_f)
            S_f = np.abs(f_signed)
            S_conv = np.abs(f_signed - f_T)
            ok_f = np.isfinite(tau_f) & np.isfinite(f_signed)
            if ok_f.sum() < 30:
                continue
            tau_f = tau_f[ok_f]; S_f = S_f[ok_f]; S_conv = S_conv[ok_f]

            # markprice side
            ts_m = wm["ts_ms"].to_numpy()
            mark = wm["MarkPrice"].to_numpy()
            idx = wm["IndexPrice"].to_numpy()
            tau_m = (t_fund - ts_m) / 1000.0
            S_pid = np.abs(mark - idx) / idx
            ok_m = np.isfinite(tau_m) & np.isfinite(S_pid) & (S_pid > 0)
            if ok_m.sum() < 30:
                continue
            tau_m = tau_m[ok_m]; S_pid = S_pid[ok_m]

            recs.append({
                "settlement_id": sid, "t_fund_ms": int(t_fund), "fday": d,
                "f_signed_at_settlement": f_T,
                # funding-side arrays share tau_f
                "tau_f": tau_f, "S_f": S_f, "S_conv": S_conv,
                # markprice-side arrays share tau_m (different cadence/coverage)
                "tau_m": tau_m, "S_pid": S_pid,
            })
            sid += 1
    return recs


@dataclass
class SignalDescription:
    name: str
    description: str
    shape: str        # "peaked" | "convergence"
    n_settlements: int
    n_pairs_pooled: int
    headline_tau_s: float          # e-folding (peaked) or half-rise (convergence)
    headline_rho: float            # 1/tau_e or ln(2)/tau_h
    loglinear_rho: float
    loglinear_r2: float
    s_at_tau0: float               # binned mean at the smallest tau bin
    s_at_taumax: float             # binned mean at the largest tau bin
    decay_detected: bool
    nls_rho: float = float("nan")
    nls_r2: float = float("nan")


def _pool_into_bins(recs: list[dict], tau_key: str, sig_key: str,
                     cfg: BLConfig) -> tuple[np.ndarray, np.ndarray]:
    """Pool all (tau, signal) pairs across settlements then bin once.
    Equivalent to a fixed-effects pooling under the assumption that
    settlements have similar shapes up to a multiplicative level; the
    geometric measurement is level-invariant so the level mixing is fine."""
    all_tau: list[np.ndarray] = []
    all_sig: list[np.ndarray] = []
    for r in recs:
        all_tau.append(r[tau_key])
        all_sig.append(r[sig_key])
    if not all_tau:
        return np.empty(0), np.empty(0)
    tau = np.concatenate(all_tau)
    sig = np.concatenate(all_sig)
    return bin_curve(tau, sig, cfg)


def _geometric_e_folding(centres: np.ndarray, mean: np.ndarray) -> float:
    """For a peaked-at-zero signal: the smallest tau at which the smoothed
    binned mean crosses peak/e, where peak = mean at the smallest tau bin.
    Returns NaN if the signal never crosses (i.e. no e-folding within the
    observed window)."""
    if centres.size < 5 or mean.size != centres.size:
        return float("nan")
    order = np.argsort(centres)
    cen = centres[order]
    m = mean[order]
    peak = float(m[0])
    if not math.isfinite(peak) or peak <= 0:
        return float("nan")
    target = peak / math.e
    below = np.where(m <= target)[0]
    if below.size == 0:
        return float("nan")
    j = int(below[0])
    if j == 0:
        return float(cen[0])
    # linear interpolation between j-1 and j on the original scale
    x0, x1 = cen[j - 1], cen[j]
    y0, y1 = m[j - 1], m[j]
    if y1 == y0:
        return float(x1)
    frac = (target - y0) / (y1 - y0)
    return float(x0 + frac * (x1 - x0))


def _convergence_half_rise(centres: np.ndarray, mean: np.ndarray) -> float:
    """For a convergence signal: the smallest tau at which the binned mean
    crosses S_max/2, where S_max = mean at the largest tau bin. Returns
    NaN if the signal does not reach half its asymptote within the window."""
    if centres.size < 5 or mean.size != centres.size:
        return float("nan")
    order = np.argsort(centres)
    cen = centres[order]
    m = mean[order]
    s_max = float(m[-1])
    if not math.isfinite(s_max) or s_max <= 0:
        return float("nan")
    target = s_max / 2.0
    above = np.where(m >= target)[0]
    if above.size == 0:
        return float("nan")
    j = int(above[0])
    if j == 0:
        return float(cen[0])
    x0, x1 = cen[j - 1], cen[j]
    y0, y1 = m[j - 1], m[j]
    if y1 == y0:
        return float(x1)
    frac = (target - y0) / (y1 - y0)
    return float(x0 + frac * (x1 - x0))


def _loglinear_slope(centres: np.ndarray, mean: np.ndarray) -> tuple[float, float]:
    """OLS of ln(mean) on tau. Returns (-slope, R^2). NaN if not enough
    positive bins."""
    ok = np.isfinite(centres) & np.isfinite(mean) & (mean > 0)
    if ok.sum() < 5:
        return float("nan"), float("nan")
    x = centres[ok]
    y = np.log(mean[ok])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return -float(coef[0]), r2


def _nls_exponential(centres: np.ndarray, mean: np.ndarray,
                     shape: str) -> tuple[float, float]:
    """Level-space NLS of the exponential template on the binned profile.
    Returns (rho, R^2 in level space); NaN on failure. See SignalDescription."""
    ok = np.isfinite(centres) & np.isfinite(mean)
    if ok.sum() < 8:
        return float("nan"), float("nan")
    x = centres[ok].astype(float)
    y = mean[ok].astype(float)
    try:
        from scipy.optimize import curve_fit
        span = max(float(x.max() - x.min()), 1.0)
        if shape == "peaked":
            def f(t, a, rho, c):
                return a * np.exp(-rho * t) + c
            p0 = (max(y[0] - y[-1], 1e-12), 3.0 / span, max(y[-1], 0.0))
            bounds = ([0.0, 1e-7, 0.0], [np.inf, 1.0, np.inf])
        else:
            def f(t, a, rho):
                return a * (1.0 - np.exp(-rho * t))
            p0 = (max(y[-1], 1e-12), 3.0 / span)
            bounds = ([0.0, 1e-7], [np.inf, 1.0])
        popt, _ = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=20000)
        yhat = f(x, *popt)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return float(popt[1]), r2
    except Exception:
        return float("nan"), float("nan")


def model_intrinsic_rho(gamma: float, sigma: float, A_side: float, k: float,
                        Q: int = 10) -> float:
    """The HJB system's OWN boundary-layer rate: the spectral gap of the autonomous M."""
    try:
        from hjb_principal_eigenvector import HJBParams
        from hjb_riccati_solver import intrinsic_relaxation_rate
        p = HJBParams(gamma=gamma, sigma=sigma, A=A_side, k=k, alpha_ml=0.0, Q=Q)
        return intrinsic_relaxation_rate(p)
    except Exception:
        return float("nan")


def describe_signal(recs: list[dict], name: str, tau_key: str, sig_key: str,
                    cfg: BLConfig) -> SignalDescription:
    desc, shape, _ = SIGNAL_DESCRIPTIONS[name]
    centres, mean = _pool_into_bins(recs, tau_key, sig_key, cfg)
    n_pairs = int(sum(r[tau_key].size for r in recs))

    s_at_0 = float(mean[0]) if centres.size else float("nan")
    s_at_max = float(mean[-1]) if centres.size else float("nan")
    if shape == "peaked":
        head_tau = _geometric_e_folding(centres, mean)
    else:
        head_tau = _convergence_half_rise(centres, mean)
    head_rho = (
        (1.0 / head_tau) if (shape == "peaked" and math.isfinite(head_tau) and head_tau > 0)
        else (math.log(2.0) / head_tau) if (math.isfinite(head_tau) and head_tau > 0)
        else float("nan")
    )
    rho_fit, r2 = _loglinear_slope(centres, mean)
    rho_nls, r2_nls = _nls_exponential(centres, mean, shape)

    decay = math.isfinite(head_tau) and head_tau > 0

    return SignalDescription(
        name=name, description=desc, shape=shape,
        n_settlements=len(recs), n_pairs_pooled=n_pairs,
        headline_tau_s=float(head_tau),
        headline_rho=float(head_rho),
        loglinear_rho=float(rho_fit),
        loglinear_r2=float(r2),
        s_at_tau0=s_at_0, s_at_taumax=s_at_max,
        decay_detected=bool(decay),
        nls_rho=float(rho_nls),
        nls_r2=float(r2_nls),
    )


def load_ffill_span(base: Path, start: str, end: str):
    """Concatenate markprice_ffill_*.parquet over the funding days [start, end].
    Returns ts_ms, FundingRate, MarkPrice, IndexPrice, is_stale, sorted and
    de-duplicated."""
    start_ms = pu.funding_day_bounds(start)[0]
    end_ms = pu.funding_day_bounds(end)[1]

    tail = (pu.parse_date(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    dates = _date_range(start, tail)
    frames = []
    cols = ["ts_ms", "FundingRate", "MarkPrice", "IndexPrice", "is_stale"]
    for d in dates:
        for h in range(24):
            p = base / d / f"markprice_ffill_{h:02d}h.parquet"
            if p.exists():
                frames.append(pl.read_parquet(p).select(cols))
    if not frames:
        raise RuntimeError(f"no markprice_ffill files under {base} for {start}..{end}")
    return (pl.concat(frames, how="vertical_relaxed")
            .sort("ts_ms")
            .unique("ts_ms", keep="first")
            .filter((pl.col("ts_ms") >= start_ms) & (pl.col("ts_ms") < end_ms)))

def _verdict(descs: dict[str, SignalDescription],
             alternatives: dict | None = None) -> dict:
    """Pick the headline rho."""
    alternatives = alternatives or {}
    for key in ("spectral_gap_queue_aware", "spectral_gap_trade_through"):
        val = alternatives.get(key)
        if val is None:
            continue
        try:
            rho_val = float(val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(rho_val) or rho_val <= 0:
            continue
        return {
            "chosen_signal": key,
            "chosen_rho": rho_val,
            "source": "spectral_gap_of_M",
            "explanation": (
                f"rho = spectral gap of the autonomous M evaluated at the "
                f"{key.replace('spectral_gap_', '')} (A,k) calibration: "
                f"rho = {rho_val:.5g} s^-1 (boundary layer 1/rho = "
                f"{1.0/rho_val:.1f} s). Under drain_normalized this is the "
                f"model-consistent pick, the drain anticipation rate equals "
                f"the system's own quote relaxation rate."
            ),
        }
    # Legacy fallback (cross-check only, flagged DEMOTED)
    order = ["S_conv", "S_pid", "S_f"]
    chosen = None
    for k in order:
        d = descs.get(k)
        if d is None:
            continue
        if d.decay_detected and math.isfinite(d.loglinear_r2) and d.loglinear_r2 >= MIN_R2_FOR_FIT:
            chosen = d
            break
    if chosen is not None:
        return {
            "chosen_signal": chosen.name,
            "chosen_rho": chosen.headline_rho,
            "source": ("convergence_half_rise_DEMOTED"
                       if chosen.shape != "peaked"
                       else "geometric_e_folding_DEMOTED"),
            "explanation": (
                f"WARNING: spectral_gap alternatives unavailable; falling "
                f"back to the legacy {chosen.name} pick (DEMOTED 2026-06-18, "
                f"measures forecast convergence not quote relaxation). "
                f"tau = {chosen.headline_tau_s:.1f} s -> rho = "
                f"{chosen.headline_rho:.5f} s^-1. To recover the production "
                f"spectral-gap pick: populate "
                f"reports/intensity_queue_aware/queue_aware_summary.json "
                f"(queue-aware, production) and/or "
                f"reports/intensity_calib/intensity_calib_fold00.json "
                f"(trade-through, sanity fallback), then rerun describe_rho."
            ),
        }
    return {
        "chosen_signal": None,
        "chosen_rho": THEORETICAL_ANCHOR_RHO,
        "source": "theoretical_anchor",
        "explanation": (
            f"No spectral gap available and no pre-settlement signal "
            f"showed a clean decay. Falling back to a 3-minute theoretical "
            f"anchor: rho = 1/180 = {THEORETICAL_ANCHOR_RHO:.5f} s^-1."
        ),
    }


def _print_report(descs: dict[str, SignalDescription], verdict: dict) -> None:
    print("\n=== describe_rho on the pre-analysis 30 days ===")
    hdr = (f"{'signal':>8} {'shape':>11} {'n_sett':>6} "
           f"{'tau_head(s)':>11} {'rho_head':>11} {'rho_fit':>11} {'R2':>6} "
           f"{'rho_nls':>11} {'R2_nls':>6} "
           f"{'S(0)':>10} {'S(max)':>10} {'decay':>6}")
    print(hdr)
    for d in descs.values():
        print(f"{d.name:>8} {d.shape:>11} {d.n_settlements:>6} "
              f"{d.headline_tau_s:>11.1f} {d.headline_rho:>11.5f} "
              f"{d.loglinear_rho:>11.5f} {d.loglinear_r2:>6.3f} "
              f"{d.nls_rho:>11.5f} {d.nls_r2:>6.3f} "
              f"{d.s_at_tau0:>10.3e} {d.s_at_taumax:>10.3e} "
              f"{'YES' if d.decay_detected else 'NO':>6}")
    print()
    print(f"Headline: chosen_signal={verdict['chosen_signal']}  "
          f"rho={verdict['chosen_rho']:.5f} s^-1  source={verdict['source']}")
    print(f"  {verdict['explanation']}")


def _tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&")
            .replace("%", r"\%").replace("#", r"\#").replace("_", r"\_"))


def _write_latex(descs: dict[str, SignalDescription], verdict: dict,
                 out_path: Path) -> None:
    lines = [
        r"% Auto-generated by describe_rho.py",
        r"% rho as a HYPERPARAMETER from pre-analysis 30-day description.",
        r"\begin{tabular}{lllrrrrl}",
        r"\toprule",
        r"signal & shape & $n_{\text{sett}}$ & $\tau_{\text{head}}$ (s) & "
        r"$\rho_{\text{head}}$ (s$^{-1}$) & $\rho_{\text{fit}}$ (s$^{-1}$) & "
        r"$R^2$ & decay \\",
        r"\midrule",
    ]
    for d in descs.values():
        lines.append(
            f"{_tex_escape(d.name)} & {_tex_escape(d.shape)} & "
            f"{d.n_settlements} & "
            f"{d.headline_tau_s:.1f} & {d.headline_rho:.5f} & "
            f"{d.loglinear_rho:.5f} & {d.loglinear_r2:.3f} & "
            f"{'yes' if d.decay_detected else 'no'} \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{8}{l}{Headline: "
        f"$\\rho = {verdict['chosen_rho']:.5f}$ s$^{{-1}}$ "
        f"(source: {_tex_escape(verdict['source'])}).}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    out_path.write_text("\n".join(lines))


def _make_figure(recs: list[dict],
                 descs: dict[str, SignalDescription],
                 cfg: BLConfig, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key in zip(axes, ["S_f", "S_pid", "S_conv"]):
        tau_key = "tau_m" if key == "S_pid" else "tau_f"
        centres, mean = _pool_into_bins(recs, tau_key, key, cfg)
        d = descs[key]
        if centres.size == 0:
            ax.set_title(f"{key}: no data")
            continue
        ax.plot(centres, mean, color="C0", lw=1.0, alpha=0.85,
                 label=f"binned mean ({d.n_pairs_pooled} pairs)")
        # log-linear fit overlay
        if math.isfinite(d.loglinear_rho) and centres.size >= 5:
            ok = mean > 0
            if ok.sum() >= 5:
                x = centres[ok]
                y = np.log(mean[ok])
                A = np.vstack([x, np.ones_like(x)]).T
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                ax.plot(x, np.exp(A @ coef), color="C3", lw=1.2, alpha=0.8,
                         label=fr"log-linear fit ($\rho={d.loglinear_rho:.4f}$, $R^2={d.loglinear_r2:.2f}$)")
        # headline timescale marker
        if math.isfinite(d.headline_tau_s) and d.headline_tau_s > 0:
            ax.axvline(d.headline_tau_s, color="C2", ls="--", lw=1.0,
                        label=fr"$\tau_{{\text{{head}}}}={d.headline_tau_s:.0f}$ s")
        ax.set_xlabel(r"$\tau = T_{\rm fund} - t$ (s)")
        ax.set_ylabel(key)
        ax.set_title(f"{key}: {d.description}")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def run_real(base: Path, fdays: list[str], out_dir: Path, cfg: BLConfig) -> dict:
    if not (_HAVE_PL and _HAVE_PU):
        raise SystemExit("polars/pipeline_utils unavailable")
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = sorted({d for d in fdays})
    if not dates:
        raise SystemExit("no fdays")
    start, end = dates[0], dates[-1]
    print(f"  loading markprice + funding over {start}..{end} "
          f"({len(dates)} pre-analysis fdays)")

    # Single-pass load of all required columns from the ffill Parquets
    df = load_ffill_span(base, start, end)
    recs = settlement_signals(df, df, dates, cfg)
    print(f"  extracted {len(recs)} settlements from pre-analysis span")

    descs = {
        "S_f":    describe_signal(recs, "S_f",    "tau_f", "S_f",    cfg),
        "S_pid":  describe_signal(recs, "S_pid",  "tau_m", "S_pid",  cfg),
        "S_conv": describe_signal(recs, "S_conv", "tau_f", "S_conv", cfg),
    }

    alternatives: dict = {}
    sigma_rep = float("nan")
    sigma_path = Path("reports/validate_sigma.json")
    if sigma_path.exists():
        sj = json.loads(sigma_path.read_text()).get("summary", {})
        for entry in sj.get("window_sensitivity", []):
            if entry.get("window_s") == 60:
                sigma_rep = float(entry["sigma_median"])
                break
    if not math.isfinite(sigma_rep):
        sigma_rep = 2.2
        alternatives["sigma_rep_fallback_used"] = True
    alternatives["sigma_rep_usdt_per_sqrt_s"] = sigma_rep
    alternatives["sigma_rep_source"] = (
        "reports/validate_sigma.json summary.window_sensitivity[window_s=60].sigma_median"
        if sigma_rep != 2.2 else "HARDCODED FALLBACK 2.2, validate_sigma.json missing"
    )
    gamma_rep = 2e-5
    alternatives["gamma_rep"] = gamma_rep

    def _finite_positive(x) -> bool:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return False
        return math.isfinite(xf) and xf > 0.0

    qa_path = Path("reports/intensity_queue_aware/queue_aware_summary.json")
    if qa_path.exists():
        qa = json.loads(qa_path.read_text()).get("pooled_fit", {})
        if _finite_positive(qa.get("A")) and _finite_positive(qa.get("k")):
            alternatives["spectral_gap_queue_aware"] = model_intrinsic_rho(
                gamma_rep, sigma_rep, float(qa["A"]), float(qa["k"]))
            alternatives["queue_aware_inputs"] = {
                "A_per_side": float(qa["A"]), "k": float(qa["k"]),
                "source": "reports/intensity_queue_aware/queue_aware_summary.json",
            }

    tt_path = Path("reports/intensity_calib/intensity_calib_fold00.json")
    if tt_path.exists():
        tt = json.loads(tt_path.read_text())
        A_tt_per_side = tt.get("A_per_side")
        k_tt = tt.get("k_touch") if _finite_positive(tt.get("k_touch")) else tt.get("k")
        if _finite_positive(A_tt_per_side) and _finite_positive(k_tt):
            alternatives["spectral_gap_trade_through"] = model_intrinsic_rho(
                gamma_rep, sigma_rep, float(A_tt_per_side), float(k_tt))
            alternatives["trade_through_inputs"] = {
                "A_per_side": float(A_tt_per_side), "k_touch": float(k_tt),
                "source": "reports/intensity_calib/intensity_calib_fold00.json",
            }

    if any(k.startswith("spectral_gap_") for k in alternatives):
        alternatives["note"] = (
            "spectral gap (1/s) of the autonomous M = the model's own "
            "quote relaxation rate; under drain_normalized this IS the "
            "production rho (see _verdict preference order). All inputs "
            "computed on the 30 pre-analysis days."
        )

    verdict = _verdict(descs, alternatives)

    summary = {
        "fdays": list(dates), "n_fdays": len(dates),
        "tau_max_s": cfg.tau_max_s, "bin_s": cfg.bin_s,
        "min_r2_for_fit": MIN_R2_FOR_FIT,
        "theoretical_anchor_rho": THEORETICAL_ANCHOR_RHO,
        "signals": {k: asdict(v) for k, v in descs.items()},
        "verdict": verdict,
        "alternatives": alternatives,
    }
    (out_dir / f"{DEFAULT_OUT_STEM}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    _write_latex(descs, verdict, out_dir / "tab_describe_rho.tex")
    _print_report(descs, verdict)
    if recs:
        _make_figure(recs, descs, cfg, out_dir / f"{DEFAULT_OUT_STEM}.png")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tau-max-s", type=int, default=1800)
    ap.add_argument("--bin-s", type=int, default=1)
    ap.add_argument("--fdays", nargs="+", default=None,
                     help="explicit funding-day list "
                          "(default: splits.json -> pre_analysis)")
    args = ap.parse_args()


    if not (_HAVE_PL and _HAVE_PU):
        raise SystemExit("polars/pipeline_utils unavailable")

    base = Path(args.base)
    out_dir = Path(args.out_dir)
    if args.fdays is not None:
        fdays = list(args.fdays)
    else:
        fdays = list(pu.load_splits(base)["splits"]["pre_analysis"])
    cfg = BLConfig(tau_max_s=args.tau_max_s, bin_s=args.bin_s, drop_stale=True)

    print(f"  describe_rho: tau_max={cfg.tau_max_s}s bin={cfg.bin_s}s  "
          f"fdays={len(fdays)} (pre-analysis only)")
    run_real(base, fdays, out_dir, cfg)


if __name__ == "__main__":
    main()
