"""Validation of the realised-volatility estimator choices."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
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

try:
    from calibrate_volatility import estimate_volatility, VolConfig
    _HAVE_VOL = True
except Exception:  # pragma: no cover
    _HAVE_VOL = False


DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
REPORTS_DIR = Path("reports")
PRICE_COL = "mid_price"

# Q3 windows to compare (seconds); 60 is the production default.
WINDOWS_S = [30, 60, 120, 300]
HALFLIVES_S = [30, 60, 300, 900]
# Q1/Q2 verdict thresholds
BIPOWER_RATIO_THRESHOLD = 0.85   # median bipower/sigma must exceed this
SPARSE_RATIO_LO, SPARSE_RATIO_HI = 0.90, 1.10
LOGRET_RATIO_LO, LOGRET_RATIO_HI = 0.98, 1.02


@dataclass
class RatioStats:
    median: float
    p5: float
    p95: float
    n_obs: int


@dataclass
class WindowStats:
    window_s: int
    n_days: int             # days that contributed a finite ratio
    sigma_median: float     # median across days of the daily median sigma_w
    ratio_median: float     # median across days of daily median(sigma_w / sigma_60)
    ratio_p5: float         # 5th pct across days of that daily ratio
    ratio_p95: float        # 95th pct across days
    ratio_worst: float      # daily ratio furthest from 1.0 (worst-day robustness)
    worst_day: str          # the funding day achieving ratio_worst


@dataclass
class HalflifeStats:
    halflife_s: int
    n_days: int
    sigma_median: float     # median across days of daily median sigma_ewma(hl)
    ratio_median: float     # median across days of daily median(sigma_ewma(hl) / sigma_ewma(60))
    ratio_p5: float
    ratio_p95: float
    ratio_worst: float
    worst_day: str


@dataclass
class VolValidationResult:
    n_days: int
    n_rows_total: int
    frozen_pct_overall: float
    bipower_ratio: RatioStats
    sparse_ratio: RatioStats
    ewma_ratio: RatioStats
    window_sensitivity: list[WindowStats]
    halflife_sensitivity: list[HalflifeStats]
    logret_consistency: RatioStats  # (sigma_logret * S) / sigma on the sample day
    sparse_bipower_ratio: RatioStats  # 5 s-grid bipower/RV pooled across days (jump test)
    zero_increment_frac: float        # mean 1 s zero-increment fraction across days
    verdicts: dict[str, str]


def ratio_stats(num: np.ndarray, denom: np.ndarray) -> RatioStats:
    """Element-wise ratio num/denom, both finite and denom != 0."""
    mask = np.isfinite(num) & np.isfinite(denom) & (denom > 0)
    r = num[mask] / denom[mask]
    if r.size == 0:
        return RatioStats(math.nan, math.nan, math.nan, 0)
    return RatioStats(
        median=float(np.median(r)),
        p5=float(np.percentile(r, 5)),
        p95=float(np.percentile(r, 95)),
        n_obs=int(r.size),
    )


DaySeries = tuple[str, np.ndarray, np.ndarray, np.ndarray]


def _day_ratio_by_key(
    sig_by_key: dict, baseline_key
) -> dict:
    """Given {key: sigma_array} for one day, return {key: (ratio_vs_baseline,
    median_sigma)} where the ratio is the day-median of sigma_key / sigma_base on
    rows where both are finite and positive."""
    baseline = sig_by_key[baseline_key]
    mask_b = np.isfinite(baseline) & (baseline > 0)
    out: dict = {}
    for key, sig in sig_by_key.items():
        active = np.isfinite(sig) & (sig > 0)
        med_sig = float(np.median(sig[active])) if active.any() else math.nan
        comb = mask_b & active
        ratio = float(np.median(sig[comb] / baseline[comb])) if comb.any() else math.nan
        out[key] = (ratio, med_sig)
    return out


def _day_window_ratios(
    price: np.ndarray, ts_ms: np.ndarray, valid: np.ndarray,
    windows_s: list[int], grid_dt_s: float = 1.0,
) -> dict:
    """One day: {w: (sigma_w/sigma_60 day-median, median sigma_w)}. Only the
    'sigma' column is computed per window (estimators={'sigma'}), so the four
    unused estimators are not paid for on every window and every day."""
    sig: dict = {}
    for w in windows_s:
        cfg = VolConfig(window_s=w, grid_dt_s=grid_dt_s, sparse_dt_s=5, min_coverage=0.5)
        sig[w] = estimate_volatility(price, ts_ms, valid, cfg, estimators={"sigma"})["sigma"]
    return _day_ratio_by_key(sig, 60 if 60 in windows_s else windows_s[0])


def _day_halflife_ratios(
    price: np.ndarray, ts_ms: np.ndarray, valid: np.ndarray,
    halflives_s: list[int], grid_dt_s: float = 1.0,
) -> dict:
    """One day: {hl: (sigma_ewma(hl)/sigma_ewma(60) day-median, median sigma_ewma)}.
    Only the 'sigma_ewma' column is computed per half-life."""
    sig: dict = {}
    for hl in halflives_s:
        cfg = VolConfig(window_s=60, grid_dt_s=grid_dt_s, sparse_dt_s=5,
                        halflife_s=hl, min_coverage=0.5)
        sig[hl] = estimate_volatility(
            price, ts_ms, valid, cfg, estimators={"sigma_ewma"})["sigma_ewma"]
    return _day_ratio_by_key(sig, 60 if 60 in halflives_s else halflives_s[0])


def _aggregate_across_days(per_day: dict, key) -> tuple:
    """Collapse per-day (ratio, median_sigma) for one key into across-day
    (n_days, sigma_median, ratio_median, ratio_p5, ratio_p95, ratio_worst,
    worst_day). worst = the daily ratio furthest from 1.0."""
    ratios: list[tuple[str, float]] = []
    med_sigs: list[float] = []
    for day, dd in per_day.items():
        r, ms = dd.get(key, (math.nan, math.nan))
        if not math.isnan(r):
            ratios.append((day, r))
        if not math.isnan(ms):
            med_sigs.append(ms)
    if ratios:
        rvals = np.array([r for _, r in ratios])
        ratio_median = float(np.median(rvals))
        ratio_p5 = float(np.percentile(rvals, 5))
        ratio_p95 = float(np.percentile(rvals, 95))
        worst_day, ratio_worst = max(ratios, key=lambda dr: abs(dr[1] - 1.0))
    else:
        ratio_median = ratio_p5 = ratio_p95 = ratio_worst = math.nan
        worst_day = ""
    sigma_median = float(np.median(med_sigs)) if med_sigs else math.nan
    return (len(ratios), sigma_median, ratio_median, ratio_p5, ratio_p95,
            ratio_worst, worst_day)


def window_sensitivity_stats(
    days: list[DaySeries], windows_s: list[int], grid_dt_s: float = 1.0,
) -> list[WindowStats]:
    """Pool the per-day window ratios across all `days` (label, price, ts, valid)
    into one WindowStats per window. The 60 s row is 1.0 by construction."""
    per_day = {lbl: _day_window_ratios(p, t, v, windows_s, grid_dt_s)
               for (lbl, p, t, v) in days}
    out: list[WindowStats] = []
    for w in windows_s:
        n, sm, rm, p5, p95, wr, wd = _aggregate_across_days(per_day, w)
        out.append(WindowStats(window_s=w, n_days=n, sigma_median=sm, ratio_median=rm,
                               ratio_p5=p5, ratio_p95=p95, ratio_worst=wr, worst_day=wd))
    return out


def halflife_sensitivity_stats(
    days: list[DaySeries], halflives_s: list[int], grid_dt_s: float = 1.0,
) -> list[HalflifeStats]:
    """EWMA analogue of window_sensitivity_stats, pooled across `days`. The
    half-life is the same memory axis as the flat window, so this confirms
    sigma_ewma inherits the 'not knife-edge' property the window sweep establishes."""
    per_day = {lbl: _day_halflife_ratios(p, t, v, halflives_s, grid_dt_s)
               for (lbl, p, t, v) in days}
    out: list[HalflifeStats] = []
    for hl in halflives_s:
        n, sm, rm, p5, p95, wr, wd = _aggregate_across_days(per_day, hl)
        out.append(HalflifeStats(halflife_s=hl, n_days=n, sigma_median=sm, ratio_median=rm,
                                 ratio_p5=p5, ratio_p95=p95, ratio_worst=wr, worst_day=wd))
    return out


def logret_consistency_ratio(
    sigma: np.ndarray,
    sigma_logret: np.ndarray,
    price: np.ndarray,
    is_frozen: "np.ndarray | None" = None,
) -> RatioStats:
    """Arithmetic<->log identity cross-check: (sigma_logret * S) / sigma, from PRE-COMPUTED."""
    sigma = np.asarray(sigma, dtype=float)
    sigma_logret = np.asarray(sigma_logret, dtype=float)
    price = np.asarray(price, dtype=float)
    live = (np.ones(sigma.size, dtype=bool) if is_frozen is None
            else ~np.asarray(is_frozen, dtype=bool))
    num = np.where(live, sigma_logret * price, np.nan)
    den = np.where(live, sigma, np.nan)
    return ratio_stats(num, den)


def _stats_of(values: list[float]) -> RatioStats:
    """median / p5 / p95 / count of a list of per-day scalars (already ratios,
    so there is no num/denom to form). NaNs are dropped; n_obs is the day count."""
    arr = np.array([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return RatioStats(math.nan, math.nan, math.nan, 0)
    return RatioStats(float(np.median(arr)), float(np.percentile(arr, 5)),
                      float(np.percentile(arr, 95)), int(arr.size))


def _day_jump_diagnostics(
    price: np.ndarray, ts_ms: np.ndarray, valid: np.ndarray, grid_dt_s: float = 1.0,
) -> tuple[float, float]:
    """One day: (sparse bipower/RV day-median, 1 s zero-increment fraction).

    The sparse ratio uses the coarse-grid estimators (intermittency-free), so it
    isolates GENUINE jump variation from the 1 s tick-discreteness artifact; the
    zero fraction is the share of valid, unit-step 1 s mid increments that are
    exactly zero, i.e. the mechanism behind any 1 s bipower deflation."""
    cfg = VolConfig(window_s=60, grid_dt_s=grid_dt_s, sparse_dt_s=5, min_coverage=0.5)
    res = estimate_volatility(price, ts_ms, valid, cfg,
                              estimators={"sigma_sparse", "sigma_bipower_sparse"})
    bps, sps = res["sigma_bipower_sparse"], res["sigma_sparse"]
    m = np.isfinite(bps) & np.isfinite(sps) & (sps > 0)
    sparse_ratio = float(np.median(bps[m] / sps[m])) if m.any() else math.nan

    dp = np.diff(np.asarray(price, dtype=float))
    dt = np.diff(np.asarray(ts_ms, dtype=np.int64))
    valid = np.asarray(valid, dtype=bool)
    ok = ((dt == int(round(grid_dt_s * 1000))) & valid[1:] & valid[:-1]
          & np.isfinite(dp))
    zero_frac = float((ok & (dp == 0)).sum() / ok.sum()) if ok.sum() else math.nan
    return sparse_ratio, zero_frac


def _verdicts(res: VolValidationResult) -> dict[str, str]:
    v = {}

    bp = res.bipower_ratio.median            # 1 s, pooled over all seconds
    sbp = res.sparse_bipower_ratio.median    # 5 s grid, pooled across days
    zf = res.zero_increment_frac
    thr = BIPOWER_RATIO_THRESHOLD
    if math.isnan(bp):
        v["jump_contamination"] = "INCONCLUSIVE (no data)"
    elif math.isnan(sbp):
        v["jump_contamination"] = (
            f"INFO: 1 s bipower/sigma = {bp:.3f}; sparse (5 s) disambiguation unavailable"
        )
    elif sbp >= thr:
        v["jump_contamination"] = (
            f"DISCRETENESS: 1 s bipower/sigma = {bp:.3f} (< {thr}), 5 s bipower/RV = "
            f"{sbp:.3f} (>= {thr}), 1 s zero-increment fraction {zf:.0%}"
        )
    else:
        v["jump_contamination"] = (
            f"JUMPS: 5 s bipower/RV = {sbp:.3f} (< {thr}), 1 s ratio {bp:.3f}, "
            f"zero-increment fraction {zf:.0%}"
        )

    sp = res.sparse_ratio.median
    if math.isnan(sp):
        v["noise_bias"] = "INCONCLUSIVE (no data)"
    elif SPARSE_RATIO_LO <= sp <= SPARSE_RATIO_HI:
        v["noise_bias"] = (
            f"PASS: median sparse/sigma = {sp:.3f} within "
            f"[{SPARSE_RATIO_LO}, {SPARSE_RATIO_HI}]"
        )
    else:
        v["noise_bias"] = (
            f"WARN: median sparse/sigma = {sp:.3f} outside "
            f"[{SPARSE_RATIO_LO}, {SPARSE_RATIO_HI}]"
        )

    ew = res.ewma_ratio.median
    if math.isnan(ew):
        v["ewma_level"] = "INCONCLUSIVE (sigma_ewma column absent; rerun calibrate_volatility)"
    else:
        spread = res.ewma_ratio.p95 - res.ewma_ratio.p5
        v["ewma_level"] = (
            f"INFO: median ewma/sigma = {ew:.3f}, spread p5={res.ewma_ratio.p5:.2f}, "
            f"p95={res.ewma_ratio.p95:.2f}, range {spread:.2f}"
        )

    w_n = next((ws.n_days for ws in res.window_sensitivity if ws.window_s == 60), 0)
    ratios = [ws.ratio_worst for ws in res.window_sensitivity
              if not math.isnan(ws.ratio_worst) and ws.window_s != 60]
    if ratios:
        max_dev = max(abs(r - 1.0) for r in ratios)
        if max_dev < 0.15:
            v["window_sensitivity"] = (
                f"PASS: worst-day deviation from the 60 s baseline = {max_dev:.1%} "
                f"< 15% over {w_n} days"
            )
        else:
            v["window_sensitivity"] = (
                f"NOTE: worst-day deviation from the 60 s baseline = {max_dev:.1%} "
                f"over {w_n} days"
            )
    else:
        v["window_sensitivity"] = "INCONCLUSIVE (window recomputation unavailable)"

    hl_n = next((hs.n_days for hs in res.halflife_sensitivity if hs.halflife_s == 60), 0)
    hl_ratios = [hs.ratio_worst for hs in res.halflife_sensitivity
                 if not math.isnan(hs.ratio_worst) and hs.halflife_s != 60]
    if hl_ratios:
        max_dev = max(abs(r - 1.0) for r in hl_ratios)
        if max_dev < 0.15:
            v["halflife_sensitivity"] = (
                f"PASS: worst-day deviation from the 60 s EWMA half-life = "
                f"{max_dev:.1%} < 15% over {hl_n} days"
            )
        else:
            v["halflife_sensitivity"] = (
                f"NOTE: worst-day deviation from the 60 s EWMA half-life = "
                f"{max_dev:.1%} over {hl_n} days"
            )
    else:
        v["halflife_sensitivity"] = "INCONCLUSIVE (EWMA recomputation unavailable)"

    lr = res.logret_consistency.median
    if math.isnan(lr):
        v["logret_consistency"] = "INCONCLUSIVE (sample day unavailable for log cross-check)"
    elif LOGRET_RATIO_LO <= lr <= LOGRET_RATIO_HI:
        v["logret_consistency"] = (
            f"PASS: median (sigma_logret * S) / sigma = {lr:.4f} within "
            f"[{LOGRET_RATIO_LO}, {LOGRET_RATIO_HI}]"
        )
    else:
        v["logret_consistency"] = (
            f"WARN: median (sigma_logret * S) / sigma = {lr:.4f} outside "
            f"[{LOGRET_RATIO_LO}, {LOGRET_RATIO_HI}]"
        )

    return v


def _load_vol_parquets(vol_dir: Path, fdays: list[str]) -> "pl.DataFrame | None":
    """Concatenate pre-computed volatility parquets for the given funding days."""
    want = ["ts_ms", "sigma", "sigma_bipower", "sigma_sparse", "sigma_ewma",
            "sigma_logret", "is_frozen"]
    frames = []
    for fday in fdays:
        p = vol_dir / f"volatility_{fday}.parquet"
        if p.exists():
            df = pl.read_parquet(p)
            for opt in ("sigma_ewma", "sigma_logret"):
                if opt not in df.columns:
                    df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(opt))
            frames.append(df.select(want))
    if not frames:
        return None
    return pl.concat(frames, how="vertical_relaxed").sort("ts_ms")


def _load_day_features(base: Path, fday: str) -> "pl.DataFrame | None":
    """Load mid_price + valid for one funding day (all hours) for the sensitivity sweep."""
    frames = []
    for d, h in pu.funding_day_paths(base, fday):
        p = base / d / f"features_{h:02d}h.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p).select(["ts_ms", "valid", PRICE_COL]))
    if not frames:
        return None
    return pl.concat(frames, how="vertical_relaxed").sort("ts_ms").unique("ts_ms", keep="first")


def run_real(base: Path, fdays: list[str], out_dir: Path) -> VolValidationResult:
    """Run all three analyses on pre-analysis funding days."""
    vol_dir = base / "volatility"

    # Q1 / Q2 / frozen: read pre-computed parquets
    df = _load_vol_parquets(vol_dir, fdays)
    if df is None:
        raise FileNotFoundError(
            f"No volatility parquets found in {vol_dir} for the given funding days. "
            "Run calibrate_volatility.py first."
        )
    unfrozen = df.filter(~pl.col("is_frozen"))

    sigma_arr = unfrozen["sigma"].to_numpy()
    bp_arr    = unfrozen["sigma_bipower"].to_numpy()
    sp_arr    = unfrozen["sigma_sparse"].to_numpy()
    ew_arr    = unfrozen["sigma_ewma"].to_numpy()

    bipower_ratio = ratio_stats(bp_arr, sigma_arr)
    sparse_ratio  = ratio_stats(sp_arr,  sigma_arr)
    ewma_ratio    = ratio_stats(ew_arr,  sigma_arr)
    frozen_pct    = float(df["is_frozen"].mean() * 100) if df.height else math.nan

    print(f"  Q1 bipower/sigma: median={bipower_ratio.median:.3f} "
          f"p5={bipower_ratio.p5:.3f} p95={bipower_ratio.p95:.3f} "
          f"n={bipower_ratio.n_obs:,}")
    print(f"  Q2 sparse/sigma:  median={sparse_ratio.median:.3f} "
          f"p5={sparse_ratio.p5:.3f} p95={sparse_ratio.p95:.3f} "
          f"n={sparse_ratio.n_obs:,}")
    print(f"  Q2b ewma/sigma:   median={ewma_ratio.median:.3f} "
          f"p5={ewma_ratio.p5:.3f} p95={ewma_ratio.p95:.3f} "
          f"n={ewma_ratio.n_obs:,}")
    print(f"  Frozen: {frozen_pct:.2f}%  of all seconds across {len(fdays)} pre-analysis days")

    win_stats: list[WindowStats] = []
    hl_stats: list[HalflifeStats] = []
    logret_stats = RatioStats(math.nan, math.nan, math.nan, 0)
    sparse_bipower_ratio = RatioStats(math.nan, math.nan, math.nan, 0)
    zero_increment_frac = math.nan
    if _HAVE_VOL:
        sweep_days: list[DaySeries] = []
        sparse_ratios: list[float] = []   # Q1b: per-day 5 s bipower/RV
        zero_fracs: list[float] = []      # Q1b: per-day 1 s zero-increment fraction
        for fd in fdays:
            if not (base / "volatility" / f"volatility_{fd}.parquet").exists():
                continue
            df_feat = _load_day_features(base, fd)
            if df_feat is None or df_feat.height <= 300:
                continue
            price = df_feat[PRICE_COL].to_numpy()
            ts_ms = df_feat["ts_ms"].to_numpy()
            valid = df_feat["valid"].to_numpy()
            sweep_days.append((fd, price, ts_ms, valid))
            # Q1b: jump-vs-discreteness disambiguation on this day.
            sr, zfr = _day_jump_diagnostics(price, ts_ms, valid)
            sparse_ratios.append(sr)
            zero_fracs.append(zfr)
            if logret_stats.n_obs == 0:
                joined = df_feat.join(
                    df.select(["ts_ms", "sigma", "sigma_logret", "is_frozen"]),
                    on="ts_ms", how="inner",
                )
                if joined.height and joined["sigma_logret"].null_count() < joined.height:
                    logret_stats = logret_consistency_ratio(
                        joined["sigma"].to_numpy(),
                        joined["sigma_logret"].to_numpy(),
                        joined[PRICE_COL].to_numpy(),
                        joined["is_frozen"].to_numpy(),
                    )
                    print(f"  Q4 arithmetic<->log on {fd}: "
                          f"(sigma_logret*S)/sigma median={logret_stats.median:.4f} "
                          f"p5={logret_stats.p5:.4f} p95={logret_stats.p95:.4f} "
                          f"n={logret_stats.n_obs:,}")

        if sweep_days:
            sparse_bipower_ratio = _stats_of(sparse_ratios)
            valid_zf = [z for z in zero_fracs if not math.isnan(z)]
            zero_increment_frac = float(np.mean(valid_zf)) if valid_zf else math.nan
            print(f"  Q1b jumps-vs-discreteness across {len(sweep_days)} days: "
                  f"sparse(5s) bipower/RV median={sparse_bipower_ratio.median:.3f} "
                  f"[p5={sparse_bipower_ratio.p5:.3f} p95={sparse_bipower_ratio.p95:.3f}]; "
                  f"1s zero-increment frac={zero_increment_frac:.1%}")
            win_stats = window_sensitivity_stats(sweep_days, WINDOWS_S)
            print(f"  Q3 window sensitivity across {len(sweep_days)} pre-analysis days "
                  f"(ratio sigma_w/sigma_60):")
            for ws in win_stats:
                print(f"    w={ws.window_s:>4}s  median={ws.ratio_median:.3f}  "
                      f"[p5={ws.ratio_p5:.3f} p95={ws.ratio_p95:.3f}]  "
                      f"worst={ws.ratio_worst:.3f} ({ws.worst_day})")
            hl_stats = halflife_sensitivity_stats(sweep_days, HALFLIVES_S)
            print(f"  Q3b EWMA half-life sensitivity across {len(sweep_days)} days "
                  f"(ratio sigma_ewma(hl)/sigma_ewma(60)):")
            for hs in hl_stats:
                print(f"    hl={hs.halflife_s:>4}s  median={hs.ratio_median:.3f}  "
                      f"[p5={hs.ratio_p5:.3f} p95={hs.ratio_p95:.3f}]  "
                      f"worst={hs.ratio_worst:.3f} ({hs.worst_day})")
        if logret_stats.n_obs == 0:
            print("  Q4 arithmetic<->log: sigma_logret absent (older parquet); INCONCLUSIVE")

    result = VolValidationResult(
        n_days=len(fdays),
        n_rows_total=df.height,
        frozen_pct_overall=frozen_pct,
        bipower_ratio=bipower_ratio,
        sparse_ratio=sparse_ratio,
        ewma_ratio=ewma_ratio,
        window_sensitivity=win_stats,
        halflife_sensitivity=hl_stats,
        logret_consistency=logret_stats,
        sparse_bipower_ratio=sparse_bipower_ratio,
        zero_increment_frac=zero_increment_frac,
        verdicts={},
    )
    result.verdicts = _verdicts(result)
    for label, text in result.verdicts.items():
        print(f"  [{label}] {text}")

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = {"summary": asdict(result)}
    (out_dir / "validate_sigma.json").write_text(json.dumps(out_json, indent=2))
    _write_latex(result, out_dir / "tab_sigma_validation.tex")
    print(f"\n  written: {out_dir}/validate_sigma.json  {out_dir}/tab_sigma_validation.tex")
    return result


def _write_latex(res: VolValidationResult, path: Path) -> None:
    bp = res.bipower_ratio
    sp = res.sparse_ratio
    ew = res.ewma_ratio
    lr = res.logret_consistency
    sbp = res.sparse_bipower_ratio

    def _cell(x: float) -> str:
        # logret cross-check is single-day only; render "--" when unavailable.
        return "--" if math.isnan(x) else f"{x:.4f}"

    zf_cell = ("--" if math.isnan(res.zero_increment_frac)
               else rf"{res.zero_increment_frac * 100:.1f}\%")
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Analysis & Metric & Median & P5 & P95 & Worst \\",
        r"\midrule",
        rf"Jump diagnostic (1\,s) & $\sigma_{{BP}}/\sigma$ & "
        rf"{bp.median:.3f} & {bp.p5:.3f} & {bp.p95:.3f} & -- \\",
        rf"Jumps vs discreteness (5\,s) & $\sigma_{{BP,sp}}/\sigma_{{sp}}$ & "
        rf"{_cell(sbp.median)} & {_cell(sbp.p5)} & {_cell(sbp.p95)} & -- \\",
        rf"1\,s zero-increment frac & $\Pr[\Delta S_{{1\,\mathrm{{s}}}} = 0]$ & "
        rf"{zf_cell} & -- & -- & -- \\",
        rf"Noise bias & $\sigma_{{sparse}}/\sigma$ & "
        rf"{sp.median:.3f} & {sp.p5:.3f} & {sp.p95:.3f} & -- \\",
        rf"EWMA responsiveness & $\sigma_{{EWMA}}/\sigma$ & "
        rf"{ew.median:.3f} & {ew.p5:.3f} & {ew.p95:.3f} & -- \\",
        rf"Arithmetic vs log & $\sigma_{{\log}}\!\cdot\!S/\sigma$ & "
        rf"{_cell(lr.median)} & {_cell(lr.p5)} & {_cell(lr.p95)} & -- \\",
        r"\midrule",
    ]
    if res.window_sensitivity:
        nd = next((ws.n_days for ws in res.window_sensitivity if ws.window_s == 60),
                  res.window_sensitivity[0].n_days)
        lines.append(
            rf"\multicolumn{{6}}{{l}}{{\textit{{Window sensitivity across {nd} days: "
            rf"ratio $\sigma_w/\sigma_{{60}}$ (median, P5, P95, worst day)}}}} \\")
        for ws in res.window_sensitivity:
            marker = r"\,$\leftarrow$" if ws.window_s == 60 else ""
            lines.append(
                rf"Window & $w={ws.window_s}$ s & "
                rf"{ws.ratio_median:.3f} & {ws.ratio_p5:.3f} & {ws.ratio_p95:.3f}"
                rf" & {ws.ratio_worst:.3f}{marker} \\"
            )
    if res.halflife_sensitivity:
        nd = next((hs.n_days for hs in res.halflife_sensitivity if hs.halflife_s == 60),
                  res.halflife_sensitivity[0].n_days)
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{6}}{{l}}{{\textit{{EWMA half-life sensitivity across {nd} days: "
            rf"ratio $\sigma_{{EWMA}}(h)/\sigma_{{EWMA}}(60)$ (median, P5, P95, worst day)}}}} \\")
        for hs in res.halflife_sensitivity:
            marker = r"\,$\leftarrow$" if hs.halflife_s == 60 else ""
            lines.append(
                rf"Half-life & $h={hs.halflife_s}$ s & "
                rf"{hs.ratio_median:.3f} & {hs.ratio_p5:.3f} & {hs.ratio_p95:.3f}"
                rf" & {hs.ratio_worst:.3f}{marker} \\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Realised-volatility estimator validation on {res.n_days} pre-analysis"
        rf" funding days. Frozen fraction: {res.frozen_pct_overall:.2f}\%."
        rf" Cells marked -- are not defined for that diagnostic: the pooled"
        rf" estimator-ratio rows carry no per-day worst, and the zero-increment"
        rf" fraction is a single pooled share; only the per-day window and"
        rf" half-life sweeps have a worst day.}}",
        r"\label{tab:sigma_validation}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description="Realised-volatility estimator validation.")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--fdays", nargs="+", default=None,
                   help="explicit funding-day list (default: pre_analysis from splits.json)")
    p.add_argument("--split", choices=["pre_analysis", "sim", "all"], default="pre_analysis",
                   help="which split to use when --fdays not given (default: pre_analysis)")
    p.add_argument("--out-dir", default=str(REPORTS_DIR))
    args = p.parse_args()


    if not (_HAVE_PL and _HAVE_PU and _HAVE_VOL):
        raise SystemExit("polars / pipeline_utils / calibrate_volatility unavailable")

    base = Path(args.base)
    out_dir = Path(args.out_dir)

    if args.fdays is not None:
        fdays = list(args.fdays)
    else:
        splits = pu.load_splits(base)["splits"]
        if args.split == "all":
            fdays = list(splits["pre_analysis"]) + list(splits["sim"])
        else:
            fdays = list(splits[args.split])

    print(f"validate_sigma | {len(fdays)} funding days ({args.split}) | base={base}")
    run_real(base, fdays, out_dir)


if __name__ == "__main__":
    main()
