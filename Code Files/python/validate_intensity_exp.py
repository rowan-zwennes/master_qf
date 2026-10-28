"""Validation of the exponential intensity form and its symmetry."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import polars as pl
    import pipeline_utils as pu  # used via calibrate_intensity AND the QA pre-analysis guard
    from calibrate_intensity import (
        MAX_BOOK_STALENESS_MS,
        # queue-aware production estimator, reused verbatim under --queue-aware
        load_for_funding_day, extract_queue_fill_records, queue_aware_fit,
        QA_HORIZON_S, QA_REPOST_S, QA_HEADLINE_H, QA_DEPTH_GRID, QA_MIN_FILLS_BIN,
    )
    _HAVE_DATA = True
except Exception:  # pragma: no cover
    _HAVE_DATA = False
    MAX_BOOK_STALENESS_MS = 2000
    QA_HORIZON_S, QA_REPOST_S, QA_HEADLINE_H = 120.0, 60.0, 2.5
    QA_DEPTH_GRID = tuple(0.5 * i for i in range(1, 61))
    QA_MIN_FILLS_BIN = 10


def _date_range(start: str, end: str) -> list[str]:
    from datetime import datetime, timedelta, timezone
    d0 = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    d1 = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    out, d = [], d0
    while d <= d1:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _load_day(base: "Path", date: str, stem: str, cols: list[str]) -> "pl.DataFrame | None":
    frames = []
    for h in range(24):
        p = base / date / f"{stem}_{h:02d}h.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p).select(cols))
    if not frames:
        return None
    ts_col = "ts_ms" if "ts_ms" in cols else cols[0]
    return pl.concat(frames, how="vertical_relaxed").sort(ts_col)

try:
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
TICK_USDT = 0.1  # BTCUSDT perp tick size (reference rescale only)

MIN_WEEK_SAMPLE_FRAC = 0.5


@dataclass
class ReachSample:
    """Penetration past the touch, split by the resting side that gets filled."""
    reach_ask: np.ndarray
    reach_bid: np.ndarray
    n_ask_all: int
    n_bid_all: int
    exposure_s: float


def extract_reach_by_side(trades: "pl.DataFrame", book: "pl.DataFrame") -> ReachSample:
    empty = ReachSample(np.empty(0), np.empty(0), 0, 0, 0.0)
    if trades.is_empty() or book.is_empty():
        return empty
    b = (book.select([
            pl.col("last_event_time"),
            pl.col("ts_ms").alias("book_grid_ms"),
            pl.col("bid_p_0"), pl.col("ask_p_0"),
         ])
         .filter(
            (pl.col("bid_p_0") > 0)
            & (pl.col("ask_p_0") > pl.col("bid_p_0"))
            & pl.col("last_event_time").is_not_null()
         )
         .sort("last_event_time"))
    t = (trades.select([
            pl.col("EventTime").cast(pl.Int64).alias("ts_ms"),
            pl.col("Price").cast(pl.Float64),
            pl.col("MakerWasBuyer").cast(pl.Boolean),
         ])
         .drop_nulls()
         .sort("ts_ms"))
    if b.is_empty() or t.is_empty():
        return empty
    j = t.join_asof(b, left_on="ts_ms", right_on="last_event_time",
                    strategy="backward")
    j = j.with_columns(
        (pl.col("ts_ms") - pl.col("last_event_time")).alias("book_age_ms")
    )
    j = j.filter(
        pl.col("bid_p_0").is_not_null()
        & (pl.col("book_age_ms") <= MAX_BOOK_STALENESS_MS)
    )
    j = j.with_columns([
        pl.max_horizontal(pl.col("bid_p_0") - pl.col("Price"), pl.lit(0.0)).alias("reach_bid"),
        pl.max_horizontal(pl.col("Price") - pl.col("ask_p_0"), pl.lit(0.0)).alias("reach_ask"),
    ])
    sell_mo = j.filter(pl.col("MakerWasBuyer"))
    buy_mo = j.filter(~pl.col("MakerWasBuyer"))
    reach_bid = sell_mo["reach_bid"].to_numpy()
    reach_ask = buy_mo["reach_ask"].to_numpy()
    reach_bid = reach_bid[np.isfinite(reach_bid) & (reach_bid > 0)]
    reach_ask = reach_ask[np.isfinite(reach_ask) & (reach_ask > 0)]
    span_ms = float(t["ts_ms"].max() - t["ts_ms"].min())
    return ReachSample(
        reach_ask=reach_ask,
        reach_bid=reach_bid,
        n_ask_all=int(buy_mo.height),
        n_bid_all=int(sell_mo.height),
        exposure_s=max(span_ms / 1000.0, 1.0),
    )


def empirical_survival(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted support and the empirical survival S(x_i) = P(X >= x_i) using the
    plotting positions 1 - (i - 0.5)/n, so S in (0, 1) (no log(0))."""
    xs = np.sort(x)
    n = xs.size
    surv = 1.0 - (np.arange(n) + 0.5) / n
    return xs, surv


def exp_mle_k(reach_pos: np.ndarray) -> float:
    """k = 1 / mean of the positive penetration (exponential MLE)."""
    r = reach_pos[np.isfinite(reach_pos) & (reach_pos > 0)]
    return 1.0 / float(r.mean()) if r.size >= 5 and r.mean() > 0 else float("nan")


def log_survival_fit(reach_pos: np.ndarray) -> tuple[float, float]:
    """OLS of ln S(delta) on delta. Returns (k = -slope, R^2). Exponential <=> the
    log-survival is a straight line, so R^2 near 1 supports the form."""
    if reach_pos.size < 8:
        return float("nan"), float("nan")
    xs, surv = empirical_survival(reach_pos)
    ok = surv > 0
    x, y = xs[ok], np.log(surv[ok])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return -float(coef[0]), r2


def weibull_shape(reach_pos: np.ndarray) -> tuple[float, float, float]:
    """Weibull shape b from  ln(-ln S(delta)) = b ln delta + c. Returns
    (b, SE(b), R^2). The exponential is Weibull with b = 1, so |b - 1| within a
    few SE is the cleanest 'it really is exponential' check; b > 1 means the hazard
    rises with depth (sub-exponential tail), b < 1 means a heavier tail."""
    if reach_pos.size < 8:
        return float("nan"), float("nan"), float("nan")
    xs, surv = empirical_survival(reach_pos)
    ok = (surv > 0) & (surv < 1) & (xs > 0)
    lx = np.log(xs[ok])
    ly = np.log(-np.log(surv[ok]))
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    yhat = A @ coef
    resid = ly - yhat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n = lx.size
    sxx = float(np.sum((lx - lx.mean()) ** 2))
    se_b = math.sqrt(ss_res / (n - 2) / sxx) if (n > 2 and sxx > 0) else float("nan")
    return float(coef[0]), se_b, r2


def aic_forms(reach_pos: np.ndarray) -> dict:
    """AIC of exponential / power-law / Weibull fit to the log-survival curve
    (lower is better). All three are fit in the SAME space (ln S as the response)
    so the AICs are comparable.
       exp:     ln S = c - k delta
       power:   ln S = c - a ln delta          (Pareto-type tail)
       weibull: ln S = c - (delta/lam)^b  -> fit via ln(-ln S) = b ln delta + c'
    """
    out: dict = {}
    if reach_pos.size < 8:
        return out
    xs, surv = empirical_survival(reach_pos)
    ok = (surv > 0) & (xs > 0)
    x, lnS = xs[ok], np.log(surv[ok])
    n = x.size

    def aic(rss: float, kpar: int) -> float:
        return n * math.log(rss / n) + 2 * kpar

    # exponential
    A = np.vstack([x, np.ones_like(x)]).T
    ce, *_ = np.linalg.lstsq(A, lnS, rcond=None)
    rss_exp = float(np.sum((lnS - A @ ce) ** 2))
    out["exp"] = {"aic": aic(rss_exp, 2), "k": -float(ce[0])}

    # power law: ln S = c - a ln x
    Ap = np.vstack([np.log(x), np.ones_like(x)]).T
    cp, *_ = np.linalg.lstsq(Ap, lnS, rcond=None)
    rss_pow = float(np.sum((lnS - Ap @ cp) ** 2))
    out["power"] = {"aic": aic(rss_pow, 2), "alpha": -float(cp[0])}

    sl = (surv[ok] > 0) & (surv[ok] < 1)
    if sl.sum() >= 4:
        lx = np.log(x[sl])
        ly = np.log(-np.log(surv[ok][sl]))
        Aw = np.vstack([lx, np.ones_like(lx)]).T
        cw, *_ = np.linalg.lstsq(Aw, ly, rcond=None)
        b = float(cw[0])
        lnlam = -float(cw[1]) / b if b != 0 else float("nan")
        lnS_hat = -np.exp(b * (np.log(x) - lnlam))  # = -(x/lam)^b
        rss_wb = float(np.sum((lnS - lnS_hat) ** 2))
        out["weibull"] = {"aic": aic(rss_wb, 2), "b": b}
    return out


def anderson_expon(reach_pos: np.ndarray) -> dict | None:
    """Anderson-Darling goodness-of-fit for the exponential (scale fit by MLE).
    reject5 = True means the exponential is rejected at the 5% level."""
    if not _HAVE_SCIPY or reach_pos.size < 8:
        return None
    try:
        res = _sps.anderson(reach_pos, dist="expon")
    except Exception:
        return None
    sig = np.asarray(res.significance_level, dtype=float)  # e.g. [15,10,5,2.5,1]
    crit = np.asarray(res.critical_values, dtype=float)
    j = int(np.argmin(np.abs(sig - 5.0)))
    return {"stat": float(res.statistic), "crit5": float(crit[j]),
            "reject5": bool(res.statistic > crit[j])}


def _exp_loglik(reach_pos: np.ndarray, k: float) -> float:
    """Log-likelihood of Exp(k) on positive reaches: N ln k - k sum(d)."""
    r = reach_pos[np.isfinite(reach_pos) & (reach_pos > 0)]
    if r.size == 0 or not (k > 0):
        return float("nan")
    return r.size * math.log(k) - k * float(r.sum())


def decay_symmetry_lrt(reach_bid: np.ndarray, reach_ask: np.ndarray) -> dict:
    """Likelihood-ratio test of H0: k^b = k^a against k^b != k^a, under the
    exponential reach model. LRT = 2(ll_sep - ll_pooled) ~ chi2(1)."""
    rb = reach_bid[np.isfinite(reach_bid) & (reach_bid > 0)]
    ra = reach_ask[np.isfinite(reach_ask) & (reach_ask > 0)]
    if rb.size < 5 or ra.size < 5:
        return {}
    k_bid = 1.0 / float(rb.mean())
    k_ask = 1.0 / float(ra.mean())
    pooled = np.concatenate([rb, ra])
    k_pool = 1.0 / float(pooled.mean())
    ll_sep = _exp_loglik(rb, k_bid) + _exp_loglik(ra, k_ask)
    ll_pool = _exp_loglik(rb, k_pool) + _exp_loglik(ra, k_pool)
    lrt = 2.0 * (ll_sep - ll_pool)
    p = float(_sps.chi2.sf(lrt, 1)) if _HAVE_SCIPY else float("nan")
    eff = abs(k_bid - k_ask) / k_pool if k_pool > 0 else float("nan")
    return {"k_bid": k_bid, "k_ask": k_ask, "k_pool": k_pool,
            "lrt": float(lrt), "p": p, "effect_size": float(eff),
            "n_bid": int(rb.size), "n_ask": int(ra.size)}


def arrival_symmetry(n_bid_all: int, n_ask_all: int) -> dict:
    """Binomial test of H0: equal bid/ask market-order arrival rates (A^b = A^a).
    Under H0 the bid-side count is Binomial(n_total, 0.5)."""
    n = n_bid_all + n_ask_all
    if n == 0:
        return {}
    ratio = n_bid_all / n_ask_all if n_ask_all > 0 else float("inf")
    if _HAVE_SCIPY:
        try:
            p = float(_sps.binomtest(n_bid_all, n, 0.5).pvalue)
        except AttributeError:  # very old scipy
            p = float(_sps.binom_test(n_bid_all, n, 0.5))
    else:
        p = float("nan")
    return {"n_bid": n_bid_all, "n_ask": n_ask_all, "ratio_bid_ask": float(ratio), "p": p}


def stability_by_week(per_day_reach: list[tuple[str, np.ndarray]],
                      min_week_frac: float = MIN_WEEK_SAMPLE_FRAC) -> dict:
    """Group daily positive-reach arrays into ISO weeks, fit k per week, and report the mean."""
    from datetime import datetime, timezone
    buckets: dict[str, list[np.ndarray]] = {}
    for day, reach in per_day_reach:
        if reach.size == 0:
            continue
        wk = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).strftime("%G-W%V")
        buckets.setdefault(wk, []).append(reach)
    if not buckets:
        return {}
    week_n = {wk: int(np.concatenate(arrs).size) for wk, arrs in buckets.items()}
    floor = min_week_frac * float(np.median(list(week_n.values())))
    weeks, ks, dropped = [], [], []
    for wk in sorted(buckets):
        if week_n[wk] < floor:
            dropped.append({"week": wk, "n": week_n[wk]})
            continue
        k = exp_mle_k(np.concatenate(buckets[wk]))
        if math.isfinite(k):
            weeks.append(wk)
            ks.append(k)
    ks = np.array(ks)
    if ks.size == 0:
        return {}
    cv = float(ks.std() / ks.mean()) if ks.mean() > 0 else float("nan")
    drift = float((ks.max() - ks.min()) / ks.mean()) if ks.mean() > 0 else float("nan")
    return {"weeks": weeks, "k_by_week": ks.tolist(), "k_mean": float(ks.mean()),
            "k_cv": cv, "k_max_drift": drift,
            "weeks_dropped_sparse": dropped, "min_week_n": floor}


def load_window(base: Path, start: str, end: str) -> tuple[ReachSample, list[tuple[str, np.ndarray]]]:
    """Pool side-aware reaches over [start, end]; also keep per-day pooled reach
    (both sides) for the weekly stability test."""
    if not _HAVE_DATA:
        raise RuntimeError("polars/pipeline_utils/calibrate_intensity unavailable")
    all_ask, all_bid = [], []
    n_ask_all = n_bid_all = 0
    exposure = 0.0
    per_day: list[tuple[str, np.ndarray]] = []
    for d in _date_range(start, end):
        trades = _load_day(base, d, "trades", ["EventTime", "Price", "MakerWasBuyer"])
        book = _load_day(base, d, "book20",
                         ["ts_ms", "last_event_time", "bid_p_0", "ask_p_0"])
        if trades is None or book is None:
            continue
        rs = extract_reach_by_side(trades, book)
        if rs.reach_ask.size or rs.reach_bid.size:
            all_ask.append(rs.reach_ask)
            all_bid.append(rs.reach_bid)
            n_ask_all += rs.n_ask_all
            n_bid_all += rs.n_bid_all
            exposure += rs.exposure_s
            per_day.append((d, np.concatenate([rs.reach_ask, rs.reach_bid])))
    pooled = ReachSample(
        reach_ask=np.concatenate(all_ask) if all_ask else np.empty(0),
        reach_bid=np.concatenate(all_bid) if all_bid else np.empty(0),
        n_ask_all=n_ask_all, n_bid_all=n_bid_all, exposure_s=exposure,
    )
    return pooled, per_day


def run_tests(pooled: ReachSample, per_day: list[tuple[str, np.ndarray]]) -> dict:
    reach_all = np.concatenate([pooled.reach_ask, pooled.reach_bid]) if (
        pooled.reach_ask.size or pooled.reach_bid.size) else np.empty(0)
    res: dict = {
        "n_reach_pos": int(reach_all.size),
        "k_mle": exp_mle_k(reach_all),
    }
    k_ls, r2_ls = log_survival_fit(reach_all)
    res["k_logsurv"], res["r2_logsurv"] = k_ls, r2_ls
    b, b_se, b_r2 = weibull_shape(reach_all)
    res["weibull_b"], res["weibull_b_se"], res["weibull_r2"] = b, b_se, b_r2
    res["aic"] = aic_forms(reach_all)
    res["anderson"] = anderson_expon(reach_all)
    res["symmetry_k"] = decay_symmetry_lrt(pooled.reach_bid, pooled.reach_ask)
    res["symmetry_A"] = arrival_symmetry(pooled.n_bid_all, pooled.n_ask_all)
    res["stability"] = stability_by_week(per_day)
    return res


def print_report(res: dict) -> None:
    print("\n=== Avellaneda-Stoikov intensity validation lambda(delta)=A exp(-k delta) ===")
    print(f"positive-reach observations: {res['n_reach_pos']}")
    print(f"k (MLE 1/mean)        = {res['k_mle']:.5f} USDT^-1   (char depth 1/k = {1.0/res['k_mle']:.3f} USDT)")
    print(f"k (log-survival slope)= {res['k_logsurv']:.5f}   R^2 = {res['r2_logsurv']:.4f}")

    b, b_se = res["weibull_b"], res["weibull_b_se"]
    z = abs(b - 1.0) / b_se if (b_se and math.isfinite(b_se) and b_se > 0) else float("nan")
    eco_exp = abs(b - 1.0) < 0.1
    if eco_exp:
        verdict = "exponential (|b-1| < 0.1)"
    else:
        verdict = "DIFFERS from exponential"
    print(f"Weibull shape b       = {b:.4f} +/- {b_se:.4f}  (|b-1|/SE={z:.2f}, {verdict})")

    print("\nAIC on the survival curve (lower is better):")
    aic = res["aic"]
    for name in ("exp", "power", "weibull"):
        m = aic.get(name)
        if m:
            extra = {"exp": f"k={m.get('k', float('nan')):.4f}",
                     "power": f"alpha={m.get('alpha', float('nan')):.3f}",
                     "weibull": f"b={m.get('b', float('nan')):.3f}"}[name]
            print(f"  {name:>8}: AIC={m['aic']:.2f}   ({extra})")
    a_exp = aic.get("exp", {}).get("aic")
    a_alt = [aic.get(n, {}).get("aic") for n in ("power", "weibull")]
    a_alt = [a for a in a_alt if a is not None]
    if a_exp is not None and a_alt:
        best_alt = min(a_alt)
        if best_alt + 2.0 < a_exp:
            print("  WARNING: a non-exponential form fits better (AIC gap > 2)")
        else:
            print("  exponential competitive with the alternatives")

    ad = res["anderson"]
    if ad:
        flag = "REJECT exponential at 5%" if ad["reject5"] else "do not reject exponential"
        print(f"Anderson-Darling: stat={ad['stat']:.3f} vs crit5={ad['crit5']:.3f}  -> {flag}")

    sk = res["symmetry_k"]
    if sk:
        print("\nSymmetry k^b = k^a (decay):")
        print(f"  k_bid={sk['k_bid']:.5f} (n={sk['n_bid']})  k_ask={sk['k_ask']:.5f} (n={sk['n_ask']})  "
              f"|k_b-k_a|/k = {sk['effect_size']:.3%}")
        flag = "REJECT symmetry" if (math.isfinite(sk["p"]) and sk["p"] < 0.05) else "symmetry supported"
        print(f"  LRT={sk['lrt']:.2f}  p={sk['p']:.3e}  -> {flag}")
    sA = res["symmetry_A"]
    if sA:
        print("Symmetry A^b = A^a (arrival):")
        flag = "REJECT" if (math.isfinite(sA["p"]) and sA["p"] < 0.05) else "supported"
        print(f"  n_bid={sA['n_bid']}  n_ask={sA['n_ask']}  ratio={sA['ratio_bid_ask']:.3f}  "
              f"binom p={sA['p']:.3e}  -> {flag}")

    st = res["stability"]
    if st:
        print(f"\nStability of k across {len(st['weeks'])} weeks: mean={st['k_mean']:.5f}  "
              f"CV={st['k_cv']:.3%}  max drift={st['k_max_drift']:.3%}")
        if st.get("weeks_dropped_sparse"):
            print(f"  ({len(st['weeks_dropped_sparse'])} under-sampled week(s) dropped "
                  f"below {st['min_week_n']:.0f} reaches: "
                  f"{[d['week'] for d in st['weeks_dropped_sparse']]})")
        if st["k_cv"] > 0.25:
            print("  week-to-week CV > 25%")


def to_latex(res: dict) -> str:
    sk = res.get("symmetry_k", {})
    ad = res.get("anderson", {}) or {}
    aic = res.get("aic", {})
    lines = [
        r"% Auto-generated by validate_intensity_exp.py",
        r"% Avellaneda-Stoikov intensity validation.",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        f"Positive-reach $n$ & {res['n_reach_pos']} \\\\",
        f"$\\hat k$ (MLE, USDT$^{{-1}}$) & {res['k_mle']:.4f} \\\\",
        f"log-survival $R^2$ & {res['r2_logsurv']:.4f} \\\\",
        f"Weibull shape $b$ & {res['weibull_b']:.3f} $\\pm$ {res['weibull_b_se']:.3f} \\\\",
        f"AIC exp & {aic.get('exp', {}).get('aic', float('nan')):.2f} \\\\",
        f"AIC power & {aic.get('power', {}).get('aic', float('nan')):.2f} \\\\",
        f"AIC Weibull & {aic.get('weibull', {}).get('aic', float('nan')):.2f} \\\\",
        f"AD stat (crit$_{{5\\%}}$) & {ad.get('stat', float('nan')):.3f} ({ad.get('crit5', float('nan')):.3f}) \\\\",
        f"$|k^b-k^a|/k$ & {sk.get('effect_size', float('nan')) * 100:.3f}\\% \\\\",
        f"symmetry LRT $p$ & {sk.get('p', float('nan')):.3e} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def make_figure(pooled: ReachSample, res: dict, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reach_all = np.concatenate([pooled.reach_ask, pooled.reach_bid]) if (
        pooled.reach_ask.size or pooled.reach_bid.size) else np.empty(0)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) log-survival with exponential fit + bid/ask split (symmetry visual)
    for arr, lab, c in ((pooled.reach_bid, "bid (sell MO)", "C0"),
                        (pooled.reach_ask, "ask (buy MO)", "C1")):
        if arr.size >= 8:
            xs, sv = empirical_survival(arr)
            ok = sv > 0
            ax0.plot(xs[ok], np.log(sv[ok]), color=c, lw=1.0, alpha=0.7, label=lab)
    if reach_all.size >= 8:
        k = res["k_logsurv"]
        xs, sv = empirical_survival(reach_all)
        ok = sv > 0
        x = xs[ok]
        c0 = float(np.log(sv[ok][0])) if x.size else 0.0
        ax0.plot(x, c0 - k * (x - x[0]), "k--", lw=1.5, label=f"exp fit k={k:.3f}")
    ax0.set_xlabel(r"penetration past touch $\delta$ (USDT)")
    ax0.set_ylabel(r"$\ln S(\delta)$")
    ax0.set_title("Log-survival: exponential = straight line")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    if reach_all.size >= 8:
        xs, sv = empirical_survival(reach_all)
        ok = (sv > 0) & (sv < 1) & (xs > 0)
        lx, ly = np.log(xs[ok]), np.log(-np.log(sv[ok]))
        ax1.scatter(lx, ly, s=5, alpha=0.3)
        b = res["weibull_b"]
        # overlay the fitted slope and a shape=1 reference
        xx = np.linspace(lx.min(), lx.max(), 50)
        cfit = ly.mean() - b * lx.mean()
        ax1.plot(xx, b * xx + cfit, "C3", lw=2, label=f"fit b={b:.2f}")
        c1 = ly.mean() - 1.0 * lx.mean()
        ax1.plot(xx, 1.0 * xx + c1, "k--", lw=1, label="b=1 (exponential)")
        ax1.set_xlabel(r"$\ln \delta$")
        ax1.set_ylabel(r"$\ln(-\ln S(\delta))$")
        ax1.set_title("Weibull shape: slope = b")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def qa_form_tests(per_depth: list[dict], min_fills_bin: int = QA_MIN_FILLS_BIN) -> dict:
    """Functional-form falsification on the fill-rate ladder lambda(delta)."""
    bins = [r for r in per_depth
            if (r.get("fills") or 0) >= min_fills_bin and (r.get("lam") or 0.0) > 0.0
            and r.get("depth_mean") is not None and r["depth_mean"] > 0]
    out: dict = {"n_bins": len(bins)}
    if len(bins) < 4:
        return out
    x = np.array([r["depth_mean"] for r in bins], dtype=float)
    lam = np.array([r["lam"] for r in bins], dtype=float)
    w = np.array([r["fills"] for r in bins], dtype=float)
    y = np.log(lam)
    sw = np.sqrt(w)
    n = x.size

    def wls(X: np.ndarray) -> tuple[np.ndarray, float]:
        beta, *_ = np.linalg.lstsq(sw[:, None] * X, sw * y, rcond=None)
        return beta, float(np.sum(w * (y - X @ beta) ** 2))

    def aic(wrss: float, kpar: int) -> float:
        return n * math.log(wrss / n) + 2 * kpar

    Xe = np.vstack([x, np.ones_like(x)]).T               # exp
    be, rss_e = wls(Xe)
    Xp = np.vstack([np.log(x), np.ones_like(x)]).T        # power
    bp, rss_p = wls(Xp)
    Xq = np.vstack([x ** 2, x, np.ones_like(x)]).T        # quadratic
    bq, rss_q = wls(Xq)

    W = np.diag(w)
    cov = np.linalg.inv(Xq.T @ W @ Xq) * (rss_q / max(n - 3, 1))
    q = float(bq[0])
    q_se = float(math.sqrt(cov[0, 0])) if cov[0, 0] > 0 else float("nan")
    t_q = q / q_se if (math.isfinite(q_se) and q_se > 0) else float("nan")
    p_q = (float(2.0 * _sps.t.sf(abs(t_q), max(n - 3, 1)))
           if (_HAVE_SCIPY and math.isfinite(t_q)) else float("nan"))

    ybar = float(np.average(y, weights=w))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    r2_exp = 1.0 - rss_e / ss_tot if ss_tot > 0 else float("nan")
    r2_quad = 1.0 - rss_q / ss_tot if ss_tot > 0 else float("nan")
    k_exp = -float(be[0])
    span = float(x.max() - x.min())
    slope_change_rel = (abs(2.0 * q * span) / abs(k_exp)
                        if abs(k_exp) > 0 else float("nan"))
    return {
        "n_bins": n,
        "k_exp": k_exp, "r2_exp": r2_exp,
        "aic_exp": aic(rss_e, 2), "aic_power": aic(rss_p, 2),
        "alpha_power": -float(bp[0]),
        "aic_power_minus_exp": aic(rss_p, 2) - aic(rss_e, 2),
        "r2_quad": r2_quad, "aic_quad": aic(rss_q, 3),
        "aic_quad_minus_exp": aic(rss_q, 3) - aic(rss_e, 2),
        "curv_q": q, "curv_q_se": q_se, "curv_t": t_q, "curv_p": p_q,
        "curv_slope_change_rel": slope_change_rel,
    }


def _qa_nfills(df: "pl.DataFrame", horizon_s: float) -> int:
    """Re-censored fill count (filled and time_to_event_s <= horizon), matching the
    re-censoring queue_aware_fit applies internally."""
    return int(df.filter(pl.col("filled")
                         & (pl.col("time_to_event_s") <= horizon_s)).height)


def qa_symmetry(records: "pl.DataFrame", horizon_s: float) -> dict:
    rb = records.filter(pl.col("side") == 1)
    ra = records.filter(pl.col("side") == -1)
    fb = queue_aware_fit(rb, horizon_s=horizon_s)
    fa = queue_aware_fit(ra, horizon_s=horizon_s)
    fp = queue_aware_fit(records, horizon_s=horizon_s)
    if not all(math.isfinite(v) for v in
               (fb["k"], fa["k"], fb["A"], fa["A"], fp["k"], fp["A"])):
        return {}
    out: dict = {"k_bid": fb["k"], "k_ask": fa["k"], "k_pool": fp["k"],
                 "A_bid": fb["A"], "A_ask": fa["A"], "A_pool": fp["A"],
                 "n_bins_bid": fb["n_bins_fit"], "n_bins_ask": fa["n_bins_fit"]}
    se_k = math.hypot(fb["k_se"], fa["k_se"])
    z_k = (fb["k"] - fa["k"]) / se_k if se_k > 0 else float("nan")
    out["k_z"] = float(z_k)
    out["k_p"] = (float(2.0 * _sps.norm.sf(abs(z_k)))
                  if (_HAVE_SCIPY and math.isfinite(z_k)) else float("nan"))
    out["k_effect_size"] = (abs(fb["k"] - fa["k"]) / fp["k"]
                            if fp["k"] > 0 else float("nan"))
    se_lnA = math.hypot(fb["A_se"] / fb["A"], fa["A_se"] / fa["A"])
    z_A = ((math.log(fb["A"]) - math.log(fa["A"])) / se_lnA
           if se_lnA > 0 else float("nan"))
    out["A_z"] = float(z_A)
    out["A_p"] = (float(2.0 * _sps.norm.sf(abs(z_A)))
                  if (_HAVE_SCIPY and math.isfinite(z_A)) else float("nan"))
    out["A_effect_size"] = (abs(fb["A"] - fa["A"]) / fp["A"]
                            if fp["A"] > 0 else float("nan"))
    n_b, n_a = _qa_nfills(rb, horizon_s), _qa_nfills(ra, horizon_s)
    out["n_fill_bid"], out["n_fill_ask"] = n_b, n_a
    if (n_b + n_a) > 0 and _HAVE_SCIPY:
        try:
            out["fill_count_p"] = float(_sps.binomtest(n_b, n_b + n_a, 0.5).pvalue)
        except AttributeError:  # very old scipy
            out["fill_count_p"] = float(_sps.binom_test(n_b, n_b + n_a, 0.5))
    else:
        out["fill_count_p"] = float("nan")
    return out


def qa_stability(records: "pl.DataFrame", horizon_s: float,
                 min_week_frac: float = MIN_WEEK_SAMPLE_FRAC) -> dict:
    """Refit k per ISO-week of the pre-analysis window and report drift (same cadence-flag."""
    r = records.with_columns(
        pl.col("post_ts").cast(pl.Datetime("ms")).dt.strftime("%G-W%V").alias("week"))
    weeks_all = sorted(r["week"].unique().to_list())
    per_week = [(wk, r.filter(pl.col("week") == wk)) for wk in weeks_all]
    per_week = [(wk, sub, _qa_nfills(sub, horizon_s)) for wk, sub in per_week]
    if not per_week:
        return {}
    floor = min_week_frac * float(np.median([n for _, _, n in per_week]))
    weeks, ks, dropped = [], [], []
    for wk, sub, n_fill in per_week:
        if n_fill < floor:
            dropped.append({"week": wk, "n_fills": n_fill})
            continue
        fit = queue_aware_fit(sub, horizon_s=horizon_s)
        if math.isfinite(fit["k"]) and fit["n_bins_fit"] >= 3:
            weeks.append(wk)
            ks.append(fit["k"])
    arr = np.array(ks)
    if arr.size == 0:
        return {}
    cv = float(arr.std() / arr.mean()) if arr.mean() > 0 else float("nan")
    drift = float((arr.max() - arr.min()) / arr.mean()) if arr.mean() > 0 else float("nan")
    return {"weeks": weeks, "k_by_week": arr.tolist(), "k_mean": float(arr.mean()),
            "k_cv": cv, "k_max_drift": drift,
            "weeks_dropped_sparse": dropped, "min_week_fills": floor}


def run_tests_qa(records: "pl.DataFrame", horizon_s: float = QA_HEADLINE_H) -> dict:
    pooled = queue_aware_fit(records, horizon_s=horizon_s)
    return {
        "horizon_s": horizon_s,
        "n_orders": int(records.height),
        "n_fills": _qa_nfills(records, horizon_s),
        "A": pooled["A"], "k": pooled["k"], "r2": pooled["r2"],
        "A_se": pooled["A_se"], "k_se": pooled["k_se"],
        "n_bins_fit": pooled["n_bins_fit"],
        "per_depth": pooled["per_depth"],
        "form": qa_form_tests(pooled["per_depth"]),
        "symmetry": qa_symmetry(records, horizon_s),
        "stability": qa_stability(records, horizon_s),
    }


def load_window_qa(base: Path, fdays: list[str] | None, horizon_s: float,
                   repost_s: float) -> tuple["pl.DataFrame", list[dict]]:
    if not _HAVE_DATA:
        raise RuntimeError("polars/calibrate_intensity unavailable")
    from data_gap_handler import load_pause_intervals, merge_intervals

    pre = list(pu.load_splits(base)["splits"]["pre_analysis"])
    if fdays is None:
        fdays = pre
    bad = [d for d in fdays if d not in pre]
    if bad:
        raise SystemExit(f"queue-aware validation is pre-analysis only; "
                         f"refusing OOS days: {bad}")
    book_cols = (["ts_ms", "valid"]
                 + [f"{s}_{w}_{i}" for s in ("bid", "ask")
                    for w in ("p", "q") for i in range(20)])
    frames, per_day = [], []
    for fday in fdays:
        trades = load_for_funding_day(
            base, fday, "trades",
            ["EventTime", "id", "Price", "Quantity", "MakerWasBuyer"])
        book = load_for_funding_day(base, fday, "book20", book_cols)
        if trades.is_empty() or book.is_empty():
            per_day.append({"fday": fday, "status": "no_data"})
            continue
        pauses = merge_intervals(load_pause_intervals(base, fday))
        rec = extract_queue_fill_records(
            trades, book, horizon_s=horizon_s, repost_s=repost_s,
            pause_intervals=pauses)
        frames.append(rec)
        per_day.append({"fday": fday, "status": "ok", "n_orders": int(rec.height),
                        "n_fills": int(rec["filled"].sum())})
    records = (pl.concat(frames) if frames
               else pl.DataFrame(schema=extract_queue_fill_records(
                   pl.DataFrame(), pl.DataFrame()).schema))
    return records, per_day


def print_report_qa(res: dict) -> None:
    print("\n=== Queue-aware intensity validation lambda(delta)=A exp(-k delta) "
          "(production fill-rate ladder) ===")
    print(f"resting orders: {res['n_orders']}   fills (re-censored H={res['horizon_s']}s): "
          f"{res['n_fills']}   depth bins in fit: {res['n_bins_fit']}")
    print(f"A_qa = {res['A']:.4f} /s/side (+/- {res['A_se']:.4f})   "
          f"k_qa = {res['k']:.5f} USDT^-1 (+/- {res['k_se']:.5f})   "
          f"char depth 1/k = {1.0 / res['k']:.3f} USDT")

    f = res["form"]
    if f and f.get("n_bins", 0) >= 4:
        print("\nFunctional form (ln lambda vs delta on the rate ladder):")
        print(f"  weighted R^2: exp={f['r2_exp']:.4f}  quad={f['r2_quad']:.4f}   k_exp={f['k_exp']:.5f}")
        print(f"  AIC exp={f['aic_exp']:.2f}   AIC power={f['aic_power']:.2f} "
              f"(power-exp={f['aic_power_minus_exp']:+.2f}, alpha={f['alpha_power']:.3f})   "
              f"AIC quad-exp={f['aic_quad_minus_exp']:+.2f}")
        if f["aic_power_minus_exp"] < -2.0:
            print("  WARNING: a power law fits the ladder better (AIC gap > 2)")
        else:
            print("  exponential beats / matches the power law on the ladder")
        if f["aic_quad_minus_exp"] < -2.0:
            print(f"  quadratic curvature diagnostic wins on AIC by "
                  f"{-f['aic_quad_minus_exp']:.0f}")
        sig = math.isfinite(f["curv_p"]) and f["curv_p"] < 0.05
        eco = math.isfinite(f["curv_slope_change_rel"]) and f["curv_slope_change_rel"] > 0.25
        verdict = ("curvature MATERIAL" if (sig and eco)
                   else "no material curvature" if not eco else "curvature small")
        print(f"  quadratic curvature q={f['curv_q']:.3e} (t={f['curv_t']:.2f}, "
              f"p={f['curv_p']:.2e}); slope change over span={f['curv_slope_change_rel']:.1%}  "
              f"-> {verdict}")

    sy = res["symmetry"]
    if sy:
        print("\nSymmetry k^b = k^a (decay):")
        print(f"  k_bid={sy['k_bid']:.5f}  k_ask={sy['k_ask']:.5f}  "
              f"|k_b-k_a|/k = {sy['k_effect_size']:.3%}   z={sy['k_z']:.2f}  p={sy['k_p']:.3e}  "
              f"-> {'REJECT symmetry' if (math.isfinite(sy['k_p']) and sy['k_p'] < 0.05) else 'symmetry supported'}")
        print("Symmetry A^b = A^a (arrival/scale):")
        print(f"  A_bid={sy['A_bid']:.4f}  A_ask={sy['A_ask']:.4f}  "
              f"|A_b-A_a|/A = {sy['A_effect_size']:.3%}   z={sy['A_z']:.2f}  p={sy['A_p']:.3e}  "
              f"-> {'REJECT' if (math.isfinite(sy['A_p']) and sy['A_p'] < 0.05) else 'supported'}")
        print(f"  fill-count cross-check: n_bid={sy['n_fill_bid']}  n_ask={sy['n_fill_ask']}  "
              f"binom p={sy['fill_count_p']:.3e}")

    st = res["stability"]
    if st:
        print(f"\nStability of k across {len(st['weeks'])} weeks: mean={st['k_mean']:.5f}  "
              f"CV={st['k_cv']:.3%}  max drift={st['k_max_drift']:.3%}")
        if st.get("weeks_dropped_sparse"):
            print(f"  ({len(st['weeks_dropped_sparse'])} under-sampled week(s) dropped "
                  f"below {st['min_week_fills']:.0f} fills: "
                  f"{[d['week'] for d in st['weeks_dropped_sparse']]})")
        if st["k_cv"] > 0.25:
            print("  *** week-to-week CV > 25%: calibration cadence may be too slow "
                  ".")


def _sci(x: float, digits: int) -> str:
    """LaTeX scientific notation, e.g. 3.98e-04 -> $3.98\\times10^{-4}$."""
    if not math.isfinite(x):
        return "--"
    mant, exp = f"{x:.{digits}e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def to_latex_qa(res: dict) -> str:
    f = res.get("form", {}) or {}
    sy = res.get("symmetry", {}) or {}
    st = res.get("stability", {}) or {}
    lines = [
        r"% Auto-generated by validate_intensity_exp.py --queue-aware",
        r"% Queue-aware intensity validation on the production fill-rate ladder.",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        r"\multicolumn{2}{@{}l}{\textit{Sample}} \\",
        f"Resting virtual orders & {res['n_orders']:,} \\\\",
        f"of which filled ($H = {res['horizon_s']}$\\,s) & {res['n_fills']:,} \\\\",
        r"\addlinespace",
        r"\multicolumn{2}{@{}l}{\textit{Fitted intensity}} \\",
        f"$\\hat{{A}}$ (per second per side) & {res['A']:.3f} \\\\",
        f"$\\hat{{k}}$ (USDT$^{{-1}}$) & {res['k']:.4f} \\\\",
        r"\addlinespace",
        r"\multicolumn{2}{@{}l}{\textit{Functional form}} \\",
        f"log-rate $R^2$, exponential & {f.get('r2_exp', float('nan')):.3f} \\\\",
        f"log-rate $R^2$, with quadratic term & {f.get('r2_quad', float('nan')):.3f} \\\\",
        f"$\\Delta$AIC, power law $-$ exponential & ${f.get('aic_power_minus_exp', float('nan')):+.0f}$ \\\\",
        f"$\\Delta$AIC, quadratic $-$ exponential & ${f.get('aic_quad_minus_exp', float('nan')):+.0f}$ \\\\",
        f"quadratic coefficient ($p$) & {_sci(f.get('curv_q', float('nan')), 2)} ({_sci(f.get('curv_p', float('nan')), 1)}) \\\\",
        r"\addlinespace",
        r"\multicolumn{2}{@{}l}{\textit{Side symmetry}} \\",
        f"$|k^b-k^a|/k$ & {sy.get('k_effect_size', float('nan')) * 100:.1f}\\% \\\\",
        f"$|A^b-A^a|/A$ & {sy.get('A_effect_size', float('nan')) * 100:.1f}\\% \\\\",
        f"Wald $p$, $k^b = k^a$ & {_sci(sy.get('k_p', float('nan')), 1)} \\\\",
        f"weekly variation in $k$ (CV) & {st.get('k_cv', float('nan')) * 100:.1f}\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def make_figure_qa(res: dict, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pd = [r for r in res["per_depth"]
          if (r.get("fills") or 0) >= QA_MIN_FILLS_BIN and (r.get("lam") or 0.0) > 0.0]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    if pd:
        x = np.array([r["depth_mean"] for r in pd], float)
        lam = np.array([r["lam"] for r in pd], float)
        ax0.scatter(x, np.log(lam), s=14, alpha=0.7, label="ladder")
        k, A = res["k"], res["A"]
        xx = np.linspace(x.min(), x.max(), 50)
        ax0.plot(xx, math.log(A) - k * xx, "k--", lw=1.5,
                 label=f"exp fit k={k:.3f}")
        ax0.set_xlabel(r"posted depth $\delta$ (USDT)")
        ax0.set_ylabel(r"$\ln \lambda(\delta)$  ($\ln$ fills/s)")
        ax0.set_title("Fill-rate ladder: exponential = straight line")
        ax0.legend(fontsize=8)
        ax0.grid(True, alpha=0.3)
        # residuals of the exponential fit (curvature is a systematic bow here)
        ax1.scatter(x, np.log(lam) - (math.log(A) - k * x), s=14, alpha=0.7)
        ax1.axhline(0.0, color="k", lw=1)
        ax1.set_xlabel(r"posted depth $\delta$ (USDT)")
        ax1.set_ylabel(r"$\ln \lambda - $ exp fit (residual)")
        ax1.set_title("Exp-fit residuals (bow = curvature)")
        ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=str, default=DEFAULT_BASE)
    ap.add_argument("--start", type=str, default=None, help="first training date YYYY-MM-DD")
    ap.add_argument("--end", type=str, default=None, help="last training date YYYY-MM-DD")
    ap.add_argument("--outdir", type=str, default="reports")
    ap.add_argument("--queue-aware", action="store_true",
                    help="validate the production fill-rate ladder instead "
                         "of trade-through")
    ap.add_argument("--fdays", nargs="+", default=None,
                    help="queue-aware: funding-day subset (default = all pre-analysis)")
    ap.add_argument("--qa-horizon-s", type=float, default=QA_HORIZON_S,
                    help="queue-aware record extraction horizon (re-censored to headline)")
    ap.add_argument("--qa-repost-s", type=float, default=QA_REPOST_S,
                    help="queue-aware cohort repost cadence")
    args = ap.parse_args()

    if not _HAVE_SCIPY:
        print("WARN: scipy unavailable; Anderson-Darling and the p-values are skipped")


    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.queue_aware:
        print(f"queue-aware | base={args.base} horizon={args.qa_horizon_s}s "
              f"repost={args.qa_repost_s}s H={QA_HEADLINE_H}s")
        records, per_day = load_window_qa(Path(args.base), args.fdays,
                                          args.qa_horizon_s, args.qa_repost_s)
        n_ok = sum(1 for d in per_day if d.get("status") == "ok")
        print(f"pooled queue-aware records: {records.height} over {n_ok} funding days "
              f"(fills {int(records['filled'].sum()) if records.height else 0}).")
        if records.height < 200:
            raise SystemExit("too few queue-aware records to validate; widen --fdays.")
        res = run_tests_qa(records, QA_HEADLINE_H)
        print_report_qa(res)
        (outdir / "intensity_exp_qa_table.tex").write_text(to_latex_qa(res) + "\n")
        make_figure_qa(res, outdir / "intensity_exp_qa.png")
        print(f"\nWrote {outdir/'intensity_exp_qa_table.tex'} and "
              f"{outdir/'intensity_exp_qa.png'}.")
        return

    if not (args.start and args.end):
        ap.error("requires --start and --end")
    print(f"Mode: real data, base={args.base}, {args.start}..{args.end}.")
    pooled, per_day = load_window(Path(args.base), args.start, args.end)
    n_pos = pooled.reach_ask.size + pooled.reach_bid.size
    print(f"pooled positive-reach observations: {n_pos} "
          f"(ask {pooled.reach_ask.size}, bid {pooled.reach_bid.size}); "
          f"total MOs ask {pooled.n_ask_all} / bid {pooled.n_bid_all}")
    if n_pos < 50:
        raise SystemExit("too few penetrating market orders to validate; widen the date range.")

    res = run_tests(pooled, per_day)
    print_report(res)

    (outdir / "intensity_exp_table.tex").write_text(to_latex(res) + "\n")
    make_figure(pooled, res, outdir / "intensity_exp.png")
    print(f"\nWrote {outdir/'intensity_exp_table.tex'} and {outdir/'intensity_exp.png'}.")


if __name__ == "__main__":
    main()
