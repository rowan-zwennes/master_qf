"""Calibration of the Avellaneda-Stoikov fill intensity."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import polars as pl
    import pipeline_utils as pu
    _HAVE_DATA = True
except Exception:  # pragma: no cover
    _HAVE_DATA = False


DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
DEFAULT_OUT = "reports/intensity_calib"
WINDOW_DAYS = 30
TICK_USDT = 0.1                # BTCUSDT perp tick size (reference rescale only)
MAX_BOOK_STALENESS_MS = 200    # drop a trade if the matched book is older than this


@dataclass
class IntensityFit:
    A: float
    k: float
    A_se: float
    k_se: float
    n_orders: int
    exposure_s: float
    seed_A: float
    seed_k: float
    seed_r2: float
    k_touch: float = float("nan")
    penetrate_frac: float = float("nan")

    def to_dict(self) -> dict:
        A_touch = (self.A * self.penetrate_frac) if (
            math.isfinite(self.A) and math.isfinite(self.penetrate_frac)) else float("nan")
        return {
            "A": _f(self.A), "k": _f(self.k),
            "A_se": _f(self.A_se), "k_se": _f(self.k_se),
            "n_orders": self.n_orders, "exposure_s": self.exposure_s,
            "char_depth_usdt": _f(1.0 / self.k) if self.k else None,
            "k_touch": _f(self.k_touch),
            "char_depth_touch_usdt": _f(1.0 / self.k_touch) if self.k_touch else None,
            "penetrate_frac": _f(self.penetrate_frac),
            "A_touch": _f(A_touch),
            "A_per_side": _f(self.A / 2.0) if math.isfinite(self.A) else None,
            "A_touch_per_side": _f(A_touch / 2.0),
            "seed": {"A": _f(self.seed_A), "k": _f(self.seed_k), "r2": _f(self.seed_r2)},
            "method": "trade_through",
            "k_note": "k=from-mid (half-spread-floor biased); k_touch=floor-free decay; seed.k=survival slope",
            "convention_note": ("A, A_touch are POOLED (buy+sell); GLT formulas use per-side. "
                                "Under the symmetry assumption: A_per_side = A/2, k_per_side = k (k is a "
                                "shape parameter, NOT halved)."),
        }


def loglinear_seed(depths: np.ndarray, exposure_s: float,
                   n_grid: int = 25) -> tuple[float, float, float]:
    """Initial estimate from ln(survival intensity). Returns (A, k, r2).

    lambda_hat(delta) = #{d_i >= delta} / T, evaluated on a grid spanning
    [0, q95]; OLS of ln lambda_hat on delta gives slope -k and intercept ln A.
    """
    if depths.size < 5:
        return float("nan"), float("nan"), float("nan")
    hi = float(np.quantile(depths, 0.95))
    if not (hi > 0):
        return float("nan"), float("nan"), float("nan")
    grid = np.linspace(0.0, hi, n_grid)
    # survival count at each grid point (orders reaching at least delta)
    sd = np.sort(depths)
    counts = depths.size - np.searchsorted(sd, grid, side="left")
    lam = counts / exposure_s
    ok = lam > 0
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    g, ln = grid[ok], np.log(lam[ok])
    Amat = np.vstack([g, np.ones_like(g)]).T
    coef, *_ = np.linalg.lstsq(Amat, ln, rcond=None)
    yhat = Amat @ coef
    ss_res = float(np.sum((ln - yhat) ** 2))
    ss_tot = float(np.sum((ln - ln.mean()) ** 2))
    k = -float(coef[0])
    A = math.exp(float(coef[1]))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return A, k, r2


def mle_intensity(depths: np.ndarray, exposure_s: float) -> IntensityFit:
    """Closed-form exponential/Poisson MLE for (A, k) with Fisher SEs."""
    depths = np.asarray(depths, dtype=float)
    depths = depths[np.isfinite(depths) & (depths > 0)]
    n = depths.size
    if n < 5 or exposure_s <= 0:
        return IntensityFit(*( [float("nan")] * 4 ), n, exposure_s,
                            float("nan"), float("nan"), float("nan"))
    mean_d = float(depths.mean())
    A = n / exposure_s
    k = 1.0 / mean_d
    A_se = math.sqrt(n) / exposure_s
    k_se = k / math.sqrt(n)
    sA, sk, sr2 = loglinear_seed(depths, exposure_s)
    return IntensityFit(A, k, A_se, k_se, n, exposure_s, sA, sk, sr2)


def _f(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def load_for_funding_day(base: Path, fday: str, stem: str, cols: list[str]) -> pl.DataFrame:
    """Load and stitch parquets for a single funding day [04:00, 04:00)."""
    start_ms, end_ms = pu.funding_day_bounds(fday)
    frames = []
    for d, h in pu.funding_day_paths(base, fday):
        p = base / d / f"{stem}_{h:02d}h.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p).select(cols))
    if not frames:
        return pl.DataFrame()
    ts_col = "ts_ms" if "ts_ms" in cols else ("EventTime" if "EventTime" in cols else cols[0])
    return (
        pl.concat(frames, how="vertical_relaxed")
        .sort(ts_col)
        .filter((pl.col(ts_col) >= start_ms) & (pl.col(ts_col) < end_ms))
    )


def joined_market_orders(trades: "pl.DataFrame",
                         book: "pl.DataFrame") -> "pl.DataFrame":
    """Return the per-trade as-of-joined frame against the prevailing book, with the depth."""
    empty_schema = {
        "ts_ms": pl.Int64, "book_age_ms": pl.Int64,
        "mid": pl.Float64, "bid_p_0": pl.Float64, "ask_p_0": pl.Float64,
        "Price": pl.Float64, "MakerWasBuyer": pl.Boolean, "Quantity": pl.Float64,
        "depth_mid": pl.Float64, "reach_touch": pl.Float64,
    }
    if trades.is_empty() or book.is_empty():
        return pl.DataFrame(schema=empty_schema)
    b = (book.select([
            pl.col("last_event_time"),
            pl.col("ts_ms").alias("book_grid_ms"),  # kept for diagnostics
            ((pl.col("bid_p_0") + pl.col("ask_p_0")) / 2.0).alias("mid"),
            pl.col("bid_p_0"), pl.col("ask_p_0"),
         ])
         .filter(
            (pl.col("bid_p_0") > 0)
            & (pl.col("ask_p_0") > pl.col("bid_p_0"))
            & pl.col("last_event_time").is_not_null()
         )
         .sort("last_event_time"))
    trade_cols = [
        pl.col("EventTime").cast(pl.Int64).alias("ts_ms"),
        pl.col("Price").cast(pl.Float64),
        pl.col("MakerWasBuyer").cast(pl.Boolean),
    ]
    if "Quantity" in trades.columns:
        trade_cols.append(pl.col("Quantity").cast(pl.Float64))
    t = (trades.select(trade_cols)
         .drop_nulls()
         .sort("ts_ms"))
    if "Quantity" not in t.columns:
        t = t.with_columns(pl.lit(float("nan")).alias("Quantity"))
    if b.is_empty() or t.is_empty():
        return pl.DataFrame(schema=empty_schema)
    j = t.join_asof(b, left_on="ts_ms", right_on="last_event_time",
                    strategy="backward")
    j = j.with_columns(
        (pl.col("ts_ms") - pl.col("last_event_time")).alias("book_age_ms")
    )
    j = j.filter(
        pl.col("mid").is_not_null()
        & (pl.col("book_age_ms") <= MAX_BOOK_STALENESS_MS)
    )
    reach = (pl.when(pl.col("MakerWasBuyer"))
               .then(pl.col("bid_p_0") - pl.col("Price"))
               .otherwise(pl.col("Price") - pl.col("ask_p_0")))
    j = j.with_columns([
        (pl.col("Price") - pl.col("mid")).abs().alias("depth_mid"),
        pl.max_horizontal(reach, pl.lit(0.0)).alias("reach_touch"),
    ])
    return j.select(list(empty_schema.keys()))


def extract_market_order_depths(trades: "pl.DataFrame", book: "pl.DataFrame"
                                ) -> tuple[np.ndarray, np.ndarray, float]:
    """Thin wrapper over joined_market_orders: return the numpy arrays that
    mle_intensity / decay_from_touch consume, plus the trade-time exposure
    span. depth_mid is filtered to positive finite values (drops the
    occasional |Price-mid|=0 trade that adds nothing to the from-mid MLE);
    reach_touch keeps zeros (non-penetrating MOs are still observations of
    'reached zero past the touch') and only drops non-finite entries."""
    j = joined_market_orders(trades, book)
    if j.is_empty():
        return np.empty(0), np.empty(0), 0.0
    depth_mid = j["depth_mid"].to_numpy()
    reach_touch = j["reach_touch"].to_numpy()
    m = np.isfinite(depth_mid) & (depth_mid > 0)
    depth_mid = depth_mid[m]
    reach_touch = reach_touch[np.isfinite(reach_touch)]
    span_ms = float(j["ts_ms"].max() - j["ts_ms"].min())
    return depth_mid, reach_touch, max(span_ms / 1000.0, 1.0)


def decay_from_touch(reach_touch: np.ndarray) -> tuple[float, float]:
    """Unbiased decay rate k from penetration past the touch. For penetrating
    orders (reach_touch > 0) the depth-beyond-touch is ~Exp(k); k = 1/mean of
    the positive reaches (translation-invariant, so floor-free). Also returns the
    penetration fraction P(reach_touch > 0). Returns (k_touch, penetrate_frac)."""
    reach_touch = np.asarray(reach_touch, dtype=float)
    reach_touch = reach_touch[np.isfinite(reach_touch)]
    if reach_touch.size == 0:
        return float("nan"), float("nan")
    pos = reach_touch[reach_touch > 0]
    pen_frac = pos.size / reach_touch.size
    if pos.size < 5:
        return float("nan"), pen_frac
    return 1.0 / float(pos.mean()), pen_frac


QA_DEPTH_GRID = tuple(0.5 * i for i in range(1, 61))  
QA_HORIZON_S = 120.0       
QA_REPOST_S = 60.0         
QA_MIN_FILLS_BIN = 10      
QA_HEADLINE_H = 2.5        

try:  
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid  
    from numba import njit as _njit
    _njit(cache=True)(lambda: 0)()   
except Exception:  # pragma: no cover
    def _njit(*a, **k):
        def deco(f):
            return f
        return deco if not (len(a) == 1 and callable(a[0])) else a[0]


@_njit(cache=True)
def _qa_scan(post_idx, post_price, post_qa, t_end_ms,
             tts, tpx, tqty, is_bid, eps, qty_eps):
    """First-fill scan for virtual last-in-queue orders on ONE side.

    Mechanics = fill_core_reference: a print strictly through our price fills
    us whole (sweep); at-level prints ratchet the queue ahead down, and the
    first print that overflows the remaining queue is our (partial) fill.
    Returns (filled, t_event_ms, mechanism 0=censored/1=sweep/2=at_level)."""
    n = post_idx.size
    filled = np.zeros(n, np.bool_)
    t_event = np.empty(n, np.int64)
    mech = np.zeros(n, np.int8)
    for i in range(n):
        j = post_idx[i]
        p = post_price[i]
        qa = post_qa[i]
        te = t_end_ms[i]
        t_event[i] = te
        while j < tts.size and tts[j] <= te:
            px = tpx[j]
            if (is_bid and px < p - eps) or ((not is_bid) and px > p + eps):
                filled[i] = True
                t_event[i] = tts[j]
                mech[i] = 1
                break
            if abs(px - p) <= eps:
                qa -= tqty[j]
                if qa < -qty_eps:
                    filled[i] = True
                    t_event[i] = tts[j]
                    mech[i] = 2
                    break
            j += 1
    return filled, t_event, mech


def _qa_queue_ahead(price: np.ndarray, lvl_p: np.ndarray, lvl_q: np.ndarray,
                    side: int, tick: float) -> np.ndarray:
    """Vectorised last-in-queue entry assumption (run_simulation._queue_ahead):
    exact level match -> displayed size; between levels / inside the touch ->
    empty level -> 0; deeper than the visible book -> deepest level's size as
    a conservative proxy. price: (n,); lvl_p/lvl_q: (n, L)."""
    match = np.abs(lvl_p - price[:, None]) < (tick * 0.5)
    has = match.any(axis=1)
    first = match.argmax(axis=1)
    qa = np.where(has, lvl_q[np.arange(price.size), first], 0.0)
    deepest_p = lvl_p[:, -1]
    beyond = (~has) & (deepest_p > 0) & (
        (price < deepest_p) if side > 0 else (price > deepest_p))
    return np.where(beyond, lvl_q[:, -1], qa)


def extract_queue_fill_records(
    trades: "pl.DataFrame",
    book: "pl.DataFrame",
    *,
    depth_grid: tuple = QA_DEPTH_GRID,
    horizon_s: float = QA_HORIZON_S,
    repost_s: float = QA_REPOST_S,
    tick: float = TICK_USDT,
    pause_intervals: list[tuple[int, int]] | None = None,
) -> "pl.DataFrame":
    """Queue-aware resting-order outcomes (fill-model-consistent estimator)."""
    schema = {
        "post_ts": pl.Int64, "side": pl.Int8,
        "delta_nominal": pl.Float64, "delta_usdt": pl.Float64,
        "price": pl.Float64, "queue_ahead": pl.Float64,
        "filled": pl.Boolean, "time_to_event_s": pl.Float64,
        "mechanism": pl.String,
    }
    if trades.is_empty() or book.is_empty():
        return pl.DataFrame(schema=schema)
    levels = sorted(int(c.split("_")[-1]) for c in book.columns
                    if c.startswith("bid_p_"))
    L = len(levels)
    b = book.sort("ts_ms")
    ts = b["ts_ms"].to_numpy()
    valid = (b["valid"].fill_null(False).to_numpy()
             if "valid" in b.columns else np.ones(ts.size, dtype=bool))
    bid_p = np.column_stack([b[f"bid_p_{i}"].fill_null(0.0).to_numpy()
                             for i in range(L)]).astype(np.float64)
    ask_p = np.column_stack([b[f"ask_p_{i}"].fill_null(0.0).to_numpy()
                             for i in range(L)]).astype(np.float64)
    bid_q = np.column_stack([b[f"bid_q_{i}"].fill_null(0.0).to_numpy()
                             for i in range(L)]).astype(np.float64)
    ask_q = np.column_stack([b[f"ask_q_{i}"].fill_null(0.0).to_numpy()
                             for i in range(L)]).astype(np.float64)

    # cohort rows: every repost_s on the grid, valid and uncrossed only
    repost_ms = int(repost_s * 1000)
    cohort = (ts % repost_ms == 0) & valid \
        & (bid_p[:, 0] > 0) & (ask_p[:, 0] > bid_p[:, 0])
    if pause_intervals:
        for s0, s1 in pause_intervals:
            cohort &= ~((ts >= s0) & (ts < s1))
    ci = np.flatnonzero(cohort)
    if ci.size == 0:
        return pl.DataFrame(schema=schema)
    c_ts = ts[ci]
    mid = 0.5 * (bid_p[ci, 0] + ask_p[ci, 0])

    # censor end: horizon, clipped at the next gap start after post
    t_end = c_ts + int(horizon_s * 1000)
    if pause_intervals:
        gap_starts = np.asarray(sorted(s0 for s0, _ in pause_intervals),
                                dtype=np.int64)
        gi = np.searchsorted(gap_starts, c_ts, side="right")
        has_gap = gi < gap_starts.size
        t_end = np.where(has_gap,
                         np.minimum(t_end, gap_starts[np.minimum(
                             gi, gap_starts.size - 1)]),
                         t_end)

    sort_keys = ["EventTime", "id"] if "id" in trades.columns else ["EventTime"]
    t = trades.sort(sort_keys)
    tts_all = t["EventTime"].cast(pl.Int64).to_numpy()
    tpx_all = t["Price"].to_numpy().astype(np.float64)
    tqty_all = t["Quantity"].to_numpy().astype(np.float64)
    taker_sell = t["MakerWasBuyer"].to_numpy()  # True -> taker sold -> hits bids
    if tts_all.size:
        # never accrue exposure past the end of the observed trade stream
        t_end = np.minimum(t_end, int(tts_all[-1]))

    frames = []
    eps = tick * 0.499
    for side in (+1, -1):
        sel = taker_sell if side > 0 else ~taker_sell
        tts, tpx, tqty = tts_all[sel], tpx_all[sel], tqty_all[sel]
        lvl_p = bid_p[ci] if side > 0 else ask_p[ci]
        lvl_q = bid_q[ci] if side > 0 else ask_q[ci]
        for d in depth_grid:
            if side > 0:
                px = np.floor((mid - d) / tick + 1e-9) * tick
                ok = px < ask_p[ci, 0] - eps        # maker only, never crossed
            else:
                px = np.ceil((mid + d) / tick - 1e-9) * tick
                ok = px > bid_p[ci, 0] + eps
            ok &= px > 0
            if not ok.any():
                continue
            qa = _qa_queue_ahead(px[ok], lvl_p[ok], lvl_q[ok], side, tick)
            post_ts = c_ts[ok]
            post_idx = np.searchsorted(tts, post_ts, side="right")
            filled, t_event, mech = _qa_scan(
                post_idx.astype(np.int64), px[ok], qa,
                t_end[ok].astype(np.int64),
                tts.astype(np.int64), tpx, tqty, side > 0, eps, 1e-12)
            depth_actual = (mid[ok] - px[ok]) if side > 0 else (px[ok] - mid[ok])
            frames.append(pl.DataFrame({
                "post_ts": post_ts,
                "side": np.full(post_ts.size, side, dtype=np.int8),
                "delta_nominal": np.full(post_ts.size, float(d)),
                "delta_usdt": depth_actual,
                "price": px[ok],
                "queue_ahead": qa,
                "filled": filled,
                "time_to_event_s": (t_event - post_ts) / 1000.0,
                "mechanism": np.array(["censored", "sweep", "at_level"]
                                      )[mech].tolist(),
            }, schema=schema))
    if not frames:
        return pl.DataFrame(schema=schema)
    return pl.concat(frames)


def queue_aware_fit(records: "pl.DataFrame",
                    min_fills_bin: int = QA_MIN_FILLS_BIN,
                    horizon_s: float | None = None) -> dict:
    """Censoring-aware Poisson rates per depth bin + weighted log-linear fit."""
    if horizon_s is not None:
        records = records.with_columns([
            (pl.col("filled")
             & (pl.col("time_to_event_s") <= horizon_s)).alias("filled"),
            pl.min_horizontal(pl.col("time_to_event_s"), pl.lit(horizon_s))
            .alias("time_to_event_s"),
            pl.when(pl.col("filled")
                    & (pl.col("time_to_event_s") <= horizon_s))
            .then(pl.col("mechanism")).otherwise(pl.lit("censored"))
            .alias("mechanism"),
        ])
    g = (records.group_by("delta_nominal").agg([
            pl.col("delta_usdt").mean().alias("depth_mean"),
            pl.col("filled").sum().alias("fills"),
            pl.col("time_to_event_s").sum().alias("exposure_s"),
            (pl.col("mechanism") == "sweep").sum().alias("n_sweep"),
            (pl.col("mechanism") == "at_level").sum().alias("n_at_level"),
            pl.len().alias("n_orders"),
        ]).sort("delta_nominal"))
    g = g.with_columns([
        (pl.col("fills") / pl.col("exposure_s")).alias("lam"),
        (pl.col("fills").sqrt() / pl.col("exposure_s")).alias("lam_se"),
    ])
    fit = g.filter((pl.col("fills") >= min_fills_bin) & (pl.col("lam") > 0))
    out = {"per_depth": g.to_dicts(), "A": float("nan"), "k": float("nan"),
           "A_se": float("nan"), "k_se": float("nan"), "r2": float("nan"),
           "n_bins_fit": fit.height, "method": "queue_aware"}
    if fit.height < 3:
        return out
    x = fit["depth_mean"].to_numpy()
    y = np.log(fit["lam"].to_numpy())
    w = fit["fills"].to_numpy().astype(np.float64)
    W = np.diag(w)
    X = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(np.sqrt(W) @ X, np.sqrt(w) * y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum(w * (y - yhat) ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    cov = np.linalg.inv(X.T @ W @ X) * (ss_res / max(fit.height - 2, 1))
    out["k"] = -float(beta[0])
    out["A"] = math.exp(float(beta[1]))
    out["k_se"] = float(np.sqrt(cov[0, 0]))
    out["A_se"] = out["A"] * float(np.sqrt(cov[1, 1]))
    out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return out


def measure_mid_volatility(
    book: "pl.DataFrame",
    *,
    sample_s: float = 1.0,
    pause_intervals: list[tuple[int, int]] | None = None,
) -> dict:
    """Realised top-of-book mid volatility in USDT/sqrt(s) on one funding day."""
    if book.is_empty():
        return {"sigma_std": float("nan"), "sigma_robust": float("nan"),
                "n_steps": 0, "sample_s": sample_s}
    b = book.sort("ts_ms")
    ts = b["ts_ms"].to_numpy()
    valid = (b["valid"].fill_null(False).to_numpy()
             if "valid" in b.columns else np.ones(ts.size, dtype=bool))
    bid = b["bid_p_0"].fill_null(0.0).to_numpy().astype(np.float64)
    ask = b["ask_p_0"].fill_null(0.0).to_numpy().astype(np.float64)
    ok = valid & (bid > 0) & (ask > bid)
    if pause_intervals:
        for s0, s1 in pause_intervals:
            ok &= ~((ts >= s0) & (ts < s1))
    if int(ok.sum()) < 10:
        return {"sigma_std": float("nan"), "sigma_robust": float("nan"),
                "n_steps": 0, "sample_s": sample_s}
    ts, mid = ts[ok], 0.5 * (bid[ok] + ask[ok])
    step = int(sample_s * 1000)
    bucket = ts // step
    last_in_bucket = np.append(np.diff(bucket) != 0, True)   # keep last per bucket
    bkt = bucket[last_in_bucket]
    mg = mid[last_in_bucket]
    adj = np.diff(bkt) == 1                                   # gap-free steps only
    dmid = np.diff(mg)[adj]
    if dmid.size < 10:
        return {"sigma_std": float("nan"), "sigma_robust": float("nan"),
                "n_steps": int(dmid.size), "sample_s": sample_s}
    sqrt_dt = math.sqrt(sample_s)
    cap = float(np.quantile(np.abs(dmid), 0.99))             # jump-tail clip
    dwin = np.clip(dmid, -cap, cap) if cap > 0 else dmid
    return {
        "sigma_std": float(np.std(dmid, ddof=1)) / sqrt_dt,
        "sigma_robust": float(np.std(dwin, ddof=1)) / sqrt_dt,
        "n_steps": int(dmid.size), "sample_s": sample_s,
    }


def h_timescale_block(sigma_usdt_per_sqrt_s: float, k: float,
                      depths: tuple = (2.0, 4.0)) -> dict:
    """Diffusion timescale tau(delta) = (delta / sigma)^2: the time for mid diffusion of scale."""
    out = {"sigma_usdt_per_sqrt_s": _f(sigma_usdt_per_sqrt_s),
           "headline_h_s": QA_HEADLINE_H, "tau_at_depth": {}}
    if not (math.isfinite(sigma_usdt_per_sqrt_s) and sigma_usdt_per_sqrt_s > 0):
        return out
    grid = list(depths)
    if math.isfinite(k) and k > 0:
        d_char = 1.0 / k
        out["delta_char_usdt"] = _f(d_char)
        out["tau_at_char_s"] = _f((d_char / sigma_usdt_per_sqrt_s) ** 2)
        grid = sorted(set(grid) | {round(d_char, 3)})
    out["tau_at_depth"] = {
        str(d): _f((d / sigma_usdt_per_sqrt_s) ** 2) for d in grid}
    return out


def repost_cadence_check(base: Path, out_dir: Path,
                         fdays: list[str] | None = None,
                         horizon_s: float = QA_HORIZON_S,
                         repost_grid: tuple = (30.0, 45.0, 60.0, 120.0),
                         headline_h: float = QA_HEADLINE_H) -> dict:
    """Falsifiable robustness check for QA_REPOST_S (pre-analysis only)."""
    from data_gap_handler import load_pause_intervals, merge_intervals

    splits = pu.load_splits(base)["splits"]
    pre = list(splits["pre_analysis"])
    if fdays is None:
        fdays = pre[:3]
    bad = [d for d in fdays if d not in pre]
    if bad:
        raise SystemExit(f"repost-cadence check is pre-analysis only; "
                         f"refusing OOS days: {bad}")
    out_dir.mkdir(parents=True, exist_ok=True)
    book_cols = (["ts_ms", "valid"]
                 + [f"{s}_{w}_{i}" for s in ("bid", "ask")
                    for w in ("p", "q") for i in range(20)])
    loaded = []
    for fday in fdays:
        trades = load_for_funding_day(
            base, fday, "trades",
            ["EventTime", "id", "Price", "Quantity", "MakerWasBuyer"])
        book = load_for_funding_day(base, fday, "book20", book_cols)
        if trades.is_empty() or book.is_empty():
            continue
        pauses = merge_intervals(load_pause_intervals(base, fday))
        loaded.append((trades, book, pauses))
    rows = []
    for rp in repost_grid:
        frames = [extract_queue_fill_records(
            tr, bk, horizon_s=horizon_s, repost_s=rp, pause_intervals=pz)
            for tr, bk, pz in loaded]
        recs = pl.concat(frames) if frames else None
        fit = (queue_aware_fit(recs, horizon_s=headline_h)
               if recs is not None else {})
        rows.append({
            "repost_s": rp, "n_orders": (recs.height if recs is not None else 0),
            "A": _f(fit.get("A")), "k": _f(fit.get("k")),
            "A_se": _f(fit.get("A_se")), "k_se": _f(fit.get("k_se")),
            "r2": _f(fit.get("r2")),
        })
        print(f"  repost={rp:>5}s  A={rows[-1]['A']}  k={rows[-1]['k']}  "
              f"A_se={rows[-1]['A_se']}  k_se={rows[-1]['k_se']}")
    out = {"days": fdays, "horizon_s": horizon_s, "headline_h_s": headline_h,
           "by_cadence": rows,
           "note": ("point estimates (A, k) should be invariant to cadence; "
                    "SEs shrink toward independence as cadence lengthens")}
    (out_dir / "repost_cadence_check.json").write_text(json.dumps(out, indent=2))
    print(f"  written: {out_dir}/repost_cadence_check.json")
    return out


DEFAULT_QA_OUT = "reports/intensity_queue_aware"


def run_queue_aware(base: Path, out_dir: Path,
                    fdays: list[str] | None = None,
                    horizon_s: float = QA_HORIZON_S,
                    repost_s: float = QA_REPOST_S) -> dict:
    """Queue-aware (A, k) on the PRE-ANALYSIS days only (hard rule: the
    calibration must never touch the 64 OOS sim days). Writes per-day record
    parquets, a pooled fit, and the trade-through comparison."""
    from data_gap_handler import load_pause_intervals, merge_intervals

    splits = pu.load_splits(base)["splits"]
    pre = list(splits["pre_analysis"])
    if fdays is None:
        fdays = pre
    bad = [d for d in fdays if d not in pre]
    if bad:
        raise SystemExit(f"queue-aware calibration is pre-analysis only; "
                         f"refusing OOS days: {bad}")
    out_dir.mkdir(parents=True, exist_ok=True)
    book_cols = (["ts_ms", "valid"]
                 + [f"{s}_{w}_{i}" for s in ("bid", "ask")
                    for w in ("p", "q") for i in range(20)])
    per_day = []
    pooled_frames = []
    sigma_robust_days: list[float] = []
    for fday in fdays:
        trades = load_for_funding_day(
            base, fday, "trades",
            ["EventTime", "id", "Price", "Quantity", "MakerWasBuyer"])
        book = load_for_funding_day(base, fday, "book20", book_cols)
        if trades.is_empty() or book.is_empty():
            per_day.append({"fday": fday, "status": "no_data"})
            print(f"  {fday}: no data, skipped")
            continue
        pauses = merge_intervals(load_pause_intervals(base, fday))
        vol = measure_mid_volatility(book, pause_intervals=pauses)
        if math.isfinite(vol["sigma_robust"]):
            sigma_robust_days.append(vol["sigma_robust"])
        rec = extract_queue_fill_records(
            trades, book, horizon_s=horizon_s, repost_s=repost_s,
            pause_intervals=pauses)
        rec.write_parquet(out_dir / f"queue_records_{fday}.parquet",
                          compression="zstd")
        fit = queue_aware_fit(rec, horizon_s=QA_HEADLINE_H)
        per_day.append({
            "fday": fday, "status": "ok", "n_orders": rec.height,
            "n_fills": int(rec["filled"].sum()),
            "A": _f(fit["A"]), "k": _f(fit["k"]), "r2": _f(fit["r2"]),
            "sigma_robust": _f(vol["sigma_robust"]),
        })
        pooled_frames.append(rec)
        print(f"  {fday}: orders={rec.height:>6}  fills={per_day[-1]['n_fills']:>6}  "
              f"A_qa={per_day[-1]['A']}  k_qa={per_day[-1]['k']}")

    pooled_recs = pl.concat(pooled_frames) if pooled_frames else None
    pooled = (queue_aware_fit(pooled_recs, horizon_s=QA_HEADLINE_H)
              if pooled_recs is not None else {})
    ladder = {}
    if pooled_recs is not None:
        for H in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 120.0):
            fH = queue_aware_fit(pooled_recs, horizon_s=H)
            ladder[str(H)] = {k: fH[k] for k in
                              ("A", "k", "A_se", "k_se", "r2", "n_bins_fit")}

    # trade-through reference on the SAME days (rolling parquets, medians)
    tt_ref: dict = {}
    roll_dir = Path("reports/intensity_rolling")
    rolls = [pl.read_parquet(roll_dir / f"intensity_rolling_{d}.parquet")
             for d in fdays
             if (roll_dir / f"intensity_rolling_{d}.parquet").exists()]
    if rolls:
        rr = pl.concat(rolls).filter(pl.col("valid"))
        tt_ref = {
            "A_pooled_med": float(rr["A"].median()),
            "k_mid_med": float(rr["k"].median()),
            "k_touch_med": float(rr["k_touch"].median()),
            "penetrate_frac_med": float(rr["penetrate_frac"].median()),
            "source": "reports/intensity_rolling medians over the same days",
        }
        if math.isfinite(pooled.get("A", float("nan"))):
            def _wedge(d_ref: float) -> tuple[float, float, float]:
                lam_tt = 0.5 * tt_ref["A_pooled_med"] * math.exp(
                    -tt_ref["k_touch_med"] * d_ref)
                lam_qa = pooled["A"] * math.exp(-pooled["k"] * d_ref)
                return lam_tt, lam_qa, lam_tt / lam_qa
            lam_tt4, lam_qa4, wedge4 = _wedge(4.0)
            tt_ref["lam_tt_side_at_4usdt"] = lam_tt4
            tt_ref["lam_qa_at_4usdt"] = lam_qa4
            tt_ref["wedge_tt_over_qa_at_4usdt"] = wedge4
            d_char = 1.0 / pooled["k"]
            lam_ttc, lam_qac, wedgec = _wedge(d_char)
            tt_ref["d_char_usdt"] = d_char
            tt_ref["lam_tt_side_at_char"] = lam_ttc
            tt_ref["lam_qa_at_char"] = lam_qac
            tt_ref["wedge_tt_over_qa_at_char"] = wedgec
            tt_ref["anchor_note"] = ("d_char = 1/k is the GLT operating "
                                     "half-spread (principled); 4 USDT is a "
                                     "fixed display anchor only")

    # measured pre-analysis mid volatility -> data-grounded H-timescale
    sigma_pooled = (float(np.median(sigma_robust_days))
                    if sigma_robust_days else float("nan"))
    h_timescale = h_timescale_block(sigma_pooled, pooled.get("k", float("nan")))
    sigma_block = {
        "sigma_robust_pooled_usdt_per_sqrt_s": _f(sigma_pooled),
        "n_days": len(sigma_robust_days),
        "estimator": ("median over per-day winsorised (99th-pct-clipped) std "
                      "of 1 s mid increments"),
    }

    summary = {
        "method": "queue_aware",
        "extraction_horizon_s": horizon_s, "repost_s": repost_s,
        "headline_horizon_s": QA_HEADLINE_H,
        "depth_grid": list(QA_DEPTH_GRID),
        "days": fdays, "per_day": per_day,
        "pooled_fit": {k: v for k, v in pooled.items() if k != "per_depth"},
        "per_depth": pooled.get("per_depth", []),
        "horizon_ladder": ladder,
        "mid_volatility": sigma_block,
        "h_timescale": h_timescale,
        "trade_through_reference": tt_ref,
        "convention_note": ("A_qa is PER-SIDE by construction (each record is "
                            "one resting order on one side); compare against "
                            "the trade-through A/2."),
    }
    (out_dir / "queue_aware_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\n  pooled: A_qa={pooled.get('A'):.4f}/s/side  "
          f"k_qa={pooled.get('k'):.4f}/USDT  r2={pooled.get('r2'):.3f}")
    if tt_ref.get("wedge_tt_over_qa_at_char"):
        print(f"  trade-through/queue-aware wedge at d_char=1/k="
              f"{tt_ref['d_char_usdt']:.1f} USDT: "
              f"{tt_ref['wedge_tt_over_qa_at_char']:.1f}x "
              f"(at 4 USDT: {tt_ref['wedge_tt_over_qa_at_4usdt']:.1f}x)")
    if math.isfinite(sigma_pooled):
        print(f"  measured sigma_robust={sigma_pooled:.3f} USDT/sqrt(s)  "
              f"-> tau(d_char)={h_timescale.get('tau_at_char_s')}s  "
              f"(headline H={QA_HEADLINE_H}s)")
    print(f"  written: {out_dir}/queue_aware_summary.json (+ per-day records)")
    return summary


def calibrate_window(base: Path, fdays: list[str]) -> IntensityFit:
    """Pool trade-through depths across `fdays` (funding days [04:00, 04:00))
    and fit (A, k). Also computes the floor-free decay k_touch from penetration
    past the touch."""
    all_depths, all_reach, total_exposure = [], [], 0.0
    for fday in fdays:
        trades = load_for_funding_day(base, fday, "trades", ["EventTime", "Price", "MakerWasBuyer"])
        book = load_for_funding_day(base, fday, "book20", ["ts_ms", "last_event_time", "bid_p_0", "ask_p_0"])
        if trades.is_empty() or book.is_empty():
            continue
        depths, reach, exposure = extract_market_order_depths(trades, book)
        if depths.size:
            all_depths.append(depths)
            all_reach.append(reach)
            total_exposure += exposure
    if not all_depths:
        return mle_intensity(np.empty(0), 0.0)
    fit = mle_intensity(np.concatenate(all_depths), total_exposure)
    fit.k_touch, fit.penetrate_frac = decay_from_touch(np.concatenate(all_reach))
    return fit


def ordered_funding_days(base: Path) -> list[str]:
    splits = pu.load_splits(base)["splits"]
    return list(splits["pre_analysis"]) + list(splits["sim"])


def production_folds(days: list[str], window_days: int = WINDOW_DAYS) -> list[dict]:
    return [
        {"fold_id": i, "train_days": days[i:i + window_days],
         "predict_day": days[i + window_days]}
        for i in range(len(days) - window_days)
    ]


def run_real(base: Path, out_dir: Path, only_fold: int | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    days = ordered_funding_days(base)
    folds = production_folds(days, WINDOW_DAYS)
    if only_fold is not None:
        folds = [f for f in folds if f["fold_id"] == only_fold]
        if not folds:
            raise SystemExit(f"fold {only_fold} out of range")
    summary = {"n_folds": len(folds), "window_days": WINDOW_DAYS,
               "method": "trade_through", "folds": []}
    for f in folds:
        tdays = f["train_days"]
        fit = calibrate_window(base, tdays)
        man = {"fold_id": f["fold_id"], "train_start": tdays[0], "train_end": tdays[-1],
               "predict_day": f["predict_day"], **fit.to_dict()}
        (out_dir / f"intensity_calib_fold{f['fold_id']:02d}.json").write_text(json.dumps(man, indent=2))
        summary["folds"].append({"fold_id": f["fold_id"], "A": man["A"], "k": man["k"],
                                 "n_orders": man["n_orders"], "predict_day": f["predict_day"]})
        print(f"  fold {f['fold_id']:>2} [{tdays[0]}..{tdays[-1]}] "
              f"n={man['n_orders']:>8}  A={man['A']}  k_mid={man['k']}  "
              f"k_touch={man['k_touch']}  pen={man['penetrate_frac']}")
    (out_dir / "intensity_calib_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  written: {out_dir}/intensity_calib_fold*.json (+ summary)")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Avellaneda-Stoikov (A,k) calibrator "
                                            "(trade-through + queue-aware).")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--queue-aware", action="store_true",
                   help="run the queue-aware survival calibration "
                        "(pre-analysis days only)")
    p.add_argument("--qa-out-dir", default=DEFAULT_QA_OUT)
    p.add_argument("--qa-horizon-s", type=float, default=QA_HORIZON_S)
    p.add_argument("--qa-repost-s", type=float, default=QA_REPOST_S)
    p.add_argument("--qa-repost-check", action="store_true",
                   help="refit (A,k) across posting cadences (pre-analysis "
                        "only)")
    p.add_argument("--fdays", nargs="+", default=None,
                   help="explicit funding days (queue-aware mode)")
    args = p.parse_args()

    if not _HAVE_DATA:
        raise SystemExit("polars/pipeline_utils unavailable")
    if args.qa_repost_check:
        repost_cadence_check(Path(args.base), Path(args.qa_out_dir),
                             fdays=args.fdays, horizon_s=args.qa_horizon_s)
        return
    if args.queue_aware:
        run_queue_aware(Path(args.base), Path(args.qa_out_dir),
                        fdays=args.fdays, horizon_s=args.qa_horizon_s,
                        repost_s=args.qa_repost_s)
        return
    run_real(Path(args.base), Path(args.out_dir), only_fold=args.fold)


if __name__ == "__main__":
    main()
