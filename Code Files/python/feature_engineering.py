"""Stage 03: feature engineering on the one-second grid."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
TOP_N = 20
TRADE_WINDOWS_S = [1, 5, 30]

DEPTH_BANDS: dict[str, tuple[int, int]] = {
    "l1": (0, 1),
    "l2_5": (1, 5),
    "l6_10": (5, 10),
    "l11_20": (10, 20),
}

DEPTH_NORM_WINDOW_S = 300
DEPTH_NORM_MIN_S = 60

REALIZED_VOL_WINDOW_S = 60
REALIZED_VOL_MIN_S = 30

VPIN_BASE_VOLUME = 70.3307
VPIN_NUM_BUCKETS = 50

VPIN_SCALE_MULTIPLIERS: dict[str, float] = {
    "micro": 0.2,   # ~5-10 min window:
    "base": 1.0,    # ~30-45 min window: 
    "macro": 5.0,   # ~2.5-4 h window: 
}
VPIN_MAX_LOOKBACK_HOURS = 12  # cap on the warm-up back-walk over trade files

# Columns of a trades file needed for VPIN (Price is not used here).
VPIN_TRADE_COLS = ["EventTime", "Quantity", "MakerWasBuyer"]


def _hour_bounds_ms(date_str: str, hour: int) -> tuple[int, int]:
    start = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=hour, tzinfo=timezone.utc
    )
    end = start + timedelta(hours=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _step_back_hour(date_str: str, hour: int) -> tuple[str, int]:
    """The (date, hour) exactly one hour before the given one."""
    if hour > 0:
        return date_str, hour - 1
    prev = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    return prev.strftime("%Y-%m-%d"), 23


def _book_to_1s(book: pl.DataFrame, start_ms: int, end_ms: int) -> pl.DataFrame:
    """
    Sample the 100ms book onto the 1s grid by taking the row whose ts_ms
    matches the grid second exactly (ts_ms % 1000 == 0).
    """
    if book.is_empty():
        return pl.DataFrame()

    return book.filter(pl.col("ts_ms") % 1000 == 0).filter(
        (pl.col("ts_ms") >= start_ms) & (pl.col("ts_ms") < end_ms)
    )


def _add_book_features(df: pl.DataFrame, prev_depth: pl.DataFrame | None = None) -> pl.DataFrame:
    """All features that depend only on the top-20 LOB columns."""
    # Sort once: the relative-depth rolling mean below is row-ordered.
    df = df.sort("ts_ms")
    bid_p = [pl.col(f"bid_p_{i}") for i in range(TOP_N)]
    bid_q = [pl.col(f"bid_q_{i}") for i in range(TOP_N)]
    ask_p = [pl.col(f"ask_p_{i}") for i in range(TOP_N)]
    ask_q = [pl.col(f"ask_q_{i}") for i in range(TOP_N)]

    def _safe_sum(exprs):
        return sum((e.fill_null(0.0) for e in exprs), pl.lit(0.0))

    bid_q_sum_5 = _safe_sum(bid_q[:5])
    bid_q_sum_20 = _safe_sum(bid_q)
    ask_q_sum_5 = _safe_sum(ask_q[:5])
    ask_q_sum_20 = _safe_sum(ask_q)

    bid_pq_sum_5 = sum(
        (bid_p[i].fill_null(0.0) * bid_q[i].fill_null(0.0) for i in range(5)),
        pl.lit(0.0),
    )
    bid_pq_sum_20 = sum(
        (bid_p[i].fill_null(0.0) * bid_q[i].fill_null(0.0) for i in range(TOP_N)),
        pl.lit(0.0),
    )
    ask_pq_sum_5 = sum(
        (ask_p[i].fill_null(0.0) * ask_q[i].fill_null(0.0) for i in range(5)),
        pl.lit(0.0),
    )
    ask_pq_sum_20 = sum(
        (ask_p[i].fill_null(0.0) * ask_q[i].fill_null(0.0) for i in range(TOP_N)),
        pl.lit(0.0),
    )

    bid_depth = {
        name: _safe_sum(bid_q[lo:hi]) for name, (lo, hi) in DEPTH_BANDS.items()
    }
    ask_depth = {
        name: _safe_sum(ask_q[lo:hi]) for name, (lo, hi) in DEPTH_BANDS.items()
    }

    df = df.with_columns(
        [
            ((pl.col("bid_p_0") + pl.col("ask_p_0")) / 2.0).alias("mid_price"),
            (pl.col("ask_p_0") - pl.col("bid_p_0")).alias("spread"),
        ]
        + [bid_depth[n].alias(f"bid_depth_{n}") for n in DEPTH_BANDS]
        + [ask_depth[n].alias(f"ask_depth_{n}") for n in DEPTH_BANDS]
    )

    df = df.with_columns(
        [
            (
                (pl.col("bid_p_0") * pl.col("ask_q_0")
                 + pl.col("ask_p_0") * pl.col("bid_q_0"))
                / (pl.col("bid_q_0") + pl.col("ask_q_0"))
            ).alias("micro_price"),
            (pl.col("spread") / pl.col("mid_price")).alias("rel_spread"),
            (pl.col("spread") / pl.col("mid_price") * 10_000).alias("spread_bps"),
        ]
        + [
            (
                (pl.col(f"bid_depth_{n}") - pl.col(f"ask_depth_{n}"))
                / (pl.col(f"bid_depth_{n}") + pl.col(f"ask_depth_{n}"))
            ).alias(f"imbalance_{n}")
            for n in DEPTH_BANDS
        ]
        + [
            (
                bid_q_sum_20
                / pl.when((pl.col("bid_p_0") - pl.col("bid_p_19")) > 0)
                .then(pl.col("bid_p_0") - pl.col("bid_p_19"))
                .otherwise(None)
            ).alias("bid_slope"),
            (
                ask_q_sum_20
                / pl.when((pl.col("ask_p_19") - pl.col("ask_p_0")) > 0)
                .then(pl.col("ask_p_19") - pl.col("ask_p_0"))
                .otherwise(None)
            ).alias("ask_slope"),
            (
                0.5 * (bid_pq_sum_5 / pl.when(bid_q_sum_5 > 0).then(bid_q_sum_5).otherwise(None))
                + 0.5 * (ask_pq_sum_5 / pl.when(ask_q_sum_5 > 0).then(ask_q_sum_5).otherwise(None))
            ).alias("weighted_mid_5"),
            (
                0.5 * (bid_pq_sum_20 / pl.when(bid_q_sum_20 > 0).then(bid_q_sum_20).otherwise(None))
                + 0.5 * (ask_pq_sum_20 / pl.when(ask_q_sum_20 > 0).then(ask_q_sum_20).otherwise(None))
            ).alias("weighted_mid_20"),
            (pl.col("ts_ms") - pl.col("last_event_time")).alias("book_age_ms"),
        ]
    )

    df = df.with_columns(
        [
            (
                (pl.col("bid_slope") - pl.col("ask_slope"))
                / pl.when((pl.col("bid_slope") + pl.col("ask_slope")) > 0)
                .then(pl.col("bid_slope") + pl.col("ask_slope"))
                .otherwise(None)
            ).alias("slope_imbalance"),
        ]
    )

    book_feature_cols = (
        ["mid_price", "spread", "micro_price", "rel_spread", "spread_bps"]
        + [f"bid_depth_{n}" for n in DEPTH_BANDS]
        + [f"ask_depth_{n}" for n in DEPTH_BANDS]
        + [f"imbalance_{n}" for n in DEPTH_BANDS]
        + ["bid_slope", "ask_slope", "weighted_mid_5", "weighted_mid_20",
           "book_age_ms", "slope_imbalance"]
    )
    is_valid = pl.col("valid").fill_null(False)
    df = df.with_columns(
        [
            pl.when(is_valid).then(pl.col(c)).otherwise(None).alias(c)
            for c in book_feature_cols
        ]
    )

    depth_cols = (
        [f"bid_depth_{n}" for n in DEPTH_BANDS]
        + [f"ask_depth_{n}" for n in DEPTH_BANDS]
    )
    if prev_depth is not None and not prev_depth.is_empty():
        depth_src = pl.concat(
            [prev_depth.select(["ts_ms"] + depth_cols), df.select(["ts_ms"] + depth_cols)],
            how="vertical_relaxed",
        ).sort("ts_ms")
    else:
        depth_src = df.select(["ts_ms"] + depth_cols)
    norm = (
        depth_src.with_columns(
            [
                pl.col(c).rolling_mean(
                    window_size=DEPTH_NORM_WINDOW_S,
                    min_samples=DEPTH_NORM_MIN_S,
                ).alias(f"_rm_{c}")
                for c in depth_cols
            ]
        )
        .filter(pl.col("ts_ms").is_in(df["ts_ms"].implode()))
        .select(["ts_ms"] + [f"_rm_{c}" for c in depth_cols])
    )
    df = (
        df.join(norm, on="ts_ms", how="left")
        .with_columns(
            [(pl.col(c) / pl.col(f"_rm_{c}")).alias(f"rel_{c}") for c in depth_cols]
        )
        .drop([f"_rm_{c}" for c in depth_cols])
    )

    return df


def _ofi_cols() -> list[str]:
    """book20 columns needed to difference OFI (prices, sizes, validity)."""
    cols = ["ts_ms", "valid"]
    for i in range(TOP_N):
        cols += [f"bid_p_{i}", f"bid_q_{i}", f"ask_p_{i}", f"ask_q_{i}"]
    return cols


def _read_book_tail(path: Path, lo_ms: int, hi_ms: int) -> pl.DataFrame:
    """OFI-relevant columns of a book20 file with ts_ms in [lo_ms, hi_ms)."""
    if not path.exists():
        return pl.DataFrame()
    return (
        pl.scan_parquet(path)
        .filter((pl.col("ts_ms") >= lo_ms) & (pl.col("ts_ms") < hi_ms))
        .select(_ofi_cols())
        .collect()
    )


def _read_prev_book_micro(path: Path, lo_ms: int, hi_ms: int) -> pl.DataFrame:
    """
    Load micro_price on the 1s grid from a previous-hour book file for
    [lo_ms, hi_ms). Returns ts_ms + micro_price; NULL on invalid rows.

    """
    empty = pl.DataFrame({
        "ts_ms": pl.Series([], dtype=pl.Int64),
        "micro_price": pl.Series([], dtype=pl.Float64),
    })
    if not path.exists():
        return empty
    df = (
        pl.scan_parquet(path)
        .filter(
            (pl.col("ts_ms") >= lo_ms)
            & (pl.col("ts_ms") < hi_ms)
            & (pl.col("ts_ms") % 1000 == 0)
        )
        .select(["ts_ms", "valid", "bid_p_0", "ask_p_0", "bid_q_0", "ask_q_0"])
        .collect()
    )
    if df.is_empty():
        return empty
    return df.with_columns(
        pl.when(pl.col("valid").fill_null(False))
        .then(
            (pl.col("bid_p_0") * pl.col("ask_q_0") + pl.col("ask_p_0") * pl.col("bid_q_0"))
            / (pl.col("bid_q_0") + pl.col("ask_q_0"))
        )
        .otherwise(None)
        .alias("micro_price")
    ).select(["ts_ms", "micro_price"])


def _read_prev_depth_1s(
    features_path: Path,
    lo_ms: int,
    hi_ms: int,
) -> pl.DataFrame:
    """
    Load valid-gated bid/ask depth columns from the previous hour's features
    parquet for [lo_ms, hi_ms), for use as rolling-mean warmup history in
    _add_book_features. 
    """
    depth_col_names = (
        [f"bid_depth_{n}" for n in DEPTH_BANDS]
        + [f"ask_depth_{n}" for n in DEPTH_BANDS]
    )
    empty = pl.DataFrame(
        {"ts_ms": pl.Series([], dtype=pl.Int64)}
        | {c: pl.Series([], dtype=pl.Float64) for c in depth_col_names}
    )
    if not features_path.exists():
        return empty
    try:
        return (
            pl.scan_parquet(features_path)
            .filter((pl.col("ts_ms") >= lo_ms) & (pl.col("ts_ms") < hi_ms))
            .select(["ts_ms"] + depth_col_names)
            .collect()
        )
    except Exception:
        return empty


def _ofi_per_tick(book: pl.DataFrame) -> pl.DataFrame:
    """Multi-level Order Flow Imbalance (Cont, Kukanov, Stoikov 2014), level-indexed."""
    b = book.sort("ts_ms")

    ofi_exprs = []
    for i in range(TOP_N):
        bp, bq = pl.col(f"bid_p_{i}"), pl.col(f"bid_q_{i}")
        ap, aq = pl.col(f"ask_p_{i}"), pl.col(f"ask_q_{i}")
        bp0, bq0 = bp.shift(1), bq.shift(1)
        ap0, aq0 = ap.shift(1), aq.shift(1)
        e_bid = (
            pl.when(bp > bp0)
            .then(bq)
            .when(bp == bp0)
            .then(bq - bq0)
            .otherwise(-bq0)
        )
        e_ask = (
            pl.when(ap > ap0)
            .then(aq0)
            .when(ap == ap0)
            .then(aq0 - aq)
            .otherwise(-aq)
        )
        ofi_exprs.append(e_bid + e_ask)

    ofi_band = {
        name: sum(ofi_exprs[lo:hi], pl.lit(0.0))
        for name, (lo, hi) in DEPTH_BANDS.items()
    }

    tick_ok = (
        pl.col("valid").fill_null(False)
        & pl.col("valid").shift(1).fill_null(False)
        & ((pl.col("ts_ms") - pl.col("ts_ms").shift(1)) == 100)
    )

    return b.select(
        [pl.col("ts_ms")]
        + [
            pl.when(tick_ok).then(ofi_band[n]).alias(f"ofi_{n}_tick")
            for n in DEPTH_BANDS
        ]
    )


def _ofi_to_1s(per_tick: pl.DataFrame, grid_ts_ms: pl.Series) -> pl.DataFrame:
    """Accumulate per-tick OFI onto the 1s grid: second t sums the per-tick OFI
    over [t-1000 ms, t)."""
    ofi_cols = [f"ofi_{n}" for n in DEPTH_BANDS]
    if per_tick.is_empty():
        return pl.DataFrame({"ts_ms": grid_ts_ms}).with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in ofi_cols]
        )

    pt = per_tick.with_columns(
        (((pl.col("ts_ms") + 999) // 1000) * 1000).alias("grid_sec")
    )
    agg = pt.group_by("grid_sec").agg(
        [pl.len().alias("_n")]
        + [pl.col(f"ofi_{n}_tick").sum().alias(f"ofi_{n}") for n in DEPTH_BANDS]
        + [
            pl.col(f"ofi_{n}_tick").null_count().alias(f"_nn_{n}")
            for n in DEPTH_BANDS
        ]
    )
    full = pl.col("_n") == 10
    agg = agg.with_columns(
        [
            pl.when(full & (pl.col(f"_nn_{n}") == 0))
            .then(pl.col(f"ofi_{n}"))
            .alias(f"ofi_{n}")
            for n in DEPTH_BANDS
        ]
    )
    return (
        agg.rename({"grid_sec": "ts_ms"})
        .select(["ts_ms"] + ofi_cols)
        .filter(pl.col("ts_ms").is_in(grid_ts_ms.implode()))
    )


def _trade_window_features(
    trades: pl.DataFrame,
    grid_ts_ms: pl.Series,
    window_s: int,
) -> pl.DataFrame:
    """
    For each grid second t, compute trade aggregates over [t-window_s*1000, t).
    """
    window_ms = window_s * 1000
    if trades.is_empty():
        out = pl.DataFrame({"ts_ms": grid_ts_ms}).with_columns(
            pl.lit(0).cast(pl.Int64).alias(f"n_trades_{window_s}s")
        )
        for c, val in [
            (f"buy_vol_{window_s}s", 0.0),
            (f"sell_vol_{window_s}s", 0.0),
            (f"signed_vol_{window_s}s", 0.0),
            (f"trade_imbalance_{window_s}s", None),
            (f"vwap_{window_s}s", None),
            (f"largest_trade_{window_s}s", 0.0),
            (f"return_{window_s}s", None),
        ]:
            out = out.with_columns(pl.lit(val).cast(pl.Float64).alias(c))
        return out

    t = trades.with_columns(
        [
            pl.col("EventTime").alias("ts_event"),
            (~pl.col("MakerWasBuyer").cast(pl.Boolean)).cast(pl.Float64).alias("is_buy"),
            pl.col("Quantity").cast(pl.Float64).alias("qty"),
            pl.col("Price").cast(pl.Float64).alias("price"),
        ]
    ).select(["ts_event", "is_buy", "qty", "price"])

    t = t.with_columns(
        [
            (pl.col("ts_event") // 1000 * 1000).alias("bin_ms"),
            (pl.col("qty") * pl.col("is_buy")).alias("buy_qty"),
            (pl.col("qty") * (1.0 - pl.col("is_buy"))).alias("sell_qty"),
            (pl.col("price") * pl.col("qty")).alias("pq"),
        ]
    )

    bins = (
        t.group_by("bin_ms")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("buy_qty").sum().alias("buy"),
                pl.col("sell_qty").sum().alias("sell"),
                pl.col("qty").sum().alias("qty_sum"),
                pl.col("pq").sum().alias("pq_sum"),
                pl.col("qty").max().alias("qty_max"),
            ]
        )
        .sort("bin_ms")
    )

    if grid_ts_ms.len() == 0:
        return pl.DataFrame({"ts_ms": grid_ts_ms})

    bin_start = int(grid_ts_ms.min()) - 2 * window_ms
    bin_end = int(grid_ts_ms.max())
    full_bins = pl.DataFrame(
        {"bin_ms": list(range(bin_start, bin_end + 1, 1000))}
    ).with_columns(pl.col("bin_ms").cast(pl.Int64))

    bins = full_bins.join(bins, on="bin_ms", how="left").with_columns(
        [
            pl.col("n").fill_null(0),
            pl.col("buy").fill_null(0.0),
            pl.col("sell").fill_null(0.0),
            pl.col("qty_sum").fill_null(0.0),
            pl.col("pq_sum").fill_null(0.0),
            pl.col("qty_max").fill_null(0.0),
        ]
    )

    bins = bins.with_columns(
        [
            pl.col("n").rolling_sum(window_size=window_s).alias("n_rs"),
            pl.col("buy").rolling_sum(window_size=window_s).alias("buy_rs"),
            pl.col("sell").rolling_sum(window_size=window_s).alias("sell_rs"),
            pl.col("qty_sum").rolling_sum(window_size=window_s).alias("qty_rs"),
            pl.col("pq_sum").rolling_sum(window_size=window_s).alias("pq_rs"),
            pl.col("qty_max").rolling_max(window_size=window_s).alias("max_rs"),
        ]
    )

    aligned = bins.with_columns((pl.col("bin_ms") + 1000).alias("ts_ms")).select(
        [
            "ts_ms",
            pl.col("n_rs").cast(pl.Int64).alias(f"n_trades_{window_s}s"),
            pl.col("buy_rs").alias(f"buy_vol_{window_s}s"),
            pl.col("sell_rs").alias(f"sell_vol_{window_s}s"),
            (pl.col("buy_rs") - pl.col("sell_rs")).alias(
                f"signed_vol_{window_s}s"
            ),
            (
                (pl.col("buy_rs") - pl.col("sell_rs"))
                / pl.when(pl.col("n_rs") > 0).then(pl.col("qty_rs")).otherwise(None)
            ).alias(f"trade_imbalance_{window_s}s"),
            (
                pl.col("pq_rs")
                / pl.when(pl.col("n_rs") > 0).then(pl.col("qty_rs")).otherwise(None)
            ).alias(f"vwap_{window_s}s"),
            pl.col("max_rs").alias(f"largest_trade_{window_s}s"),
        ]
    )

    aligned = aligned.with_columns(
        (
            (pl.col(f"vwap_{window_s}s") / pl.col(f"vwap_{window_s}s").shift(window_s)).log()
        ).alias(f"return_{window_s}s")
    )

    # Restrict to grid seconds
    aligned = aligned.filter(pl.col("ts_ms").is_in(grid_ts_ms.implode()))

    return aligned


def _collect_trades_back(
    base: Path,
    date_str: str,
    hour: int,
    start_ms: int,
    *,
    min_volume: float,
    max_hours: int,
) -> pl.DataFrame:
    """Gather trades with EventTime < start_ms by walking back over previous hourly trade."""
    frames: list[pl.DataFrame] = []
    vol = 0.0
    d, h = date_str, hour
    for _ in range(max_hours):
        d, h = _step_back_hour(d, h)
        path = base / d / f"trades_{h:02d}h.parquet"
        if not path.exists():
            break
        tf = (
            pl.scan_parquet(path)
            .filter(pl.col("EventTime") < start_ms)
            .select(VPIN_TRADE_COLS)
            .collect()
        )
        if tf.is_empty():
            # An hour with no trades is itself a gap for the volume clock.
            break
        frames.append(tf)
        vol += float(tf["Quantity"].sum())
        if vol >= min_volume:
            break
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed").sort("EventTime")


def _compute_vpin(
    trades: pl.DataFrame,
    grid_ts_ms: pl.Series,
    bucket_volume: float,
    n_buckets: int,
) -> pl.DataFrame:
    """Volume-synchronized VPIN on the 1s grid."""
    grid = grid_ts_ms.to_numpy()
    null_vpin = pl.DataFrame(
        {
            "ts_ms": pl.Series("ts_ms", grid, dtype=pl.Int64),
            "vpin": pl.Series("vpin", [None] * len(grid), dtype=pl.Float64),
            "vpin_window_s": pl.Series(
                "vpin_window_s", [None] * len(grid), dtype=pl.Float64
            ),
        }
    )
    if trades.is_empty() or bucket_volume <= 0.0 or n_buckets < 1:
        return null_vpin

    qty = trades["Quantity"].cast(pl.Float64).to_numpy()
    is_buy = (~trades["MakerWasBuyer"].cast(pl.Boolean)).cast(pl.Float64).to_numpy()
    ev = trades["EventTime"].cast(pl.Int64).to_numpy()

    cum_total = np.cumsum(qty)
    cum_buy = np.cumsum(qty * is_buy)
    total_vol = float(cum_total[-1])

    n_full = int(total_vol // bucket_volume)  # number of complete buckets
    if n_full < n_buckets:
        return null_vpin

    edges = np.arange(n_full + 1, dtype=np.float64) * bucket_volume
    xp = np.concatenate(([0.0], cum_total))
    fp = np.concatenate(([0.0], cum_buy))
    buy_at_edge = np.interp(edges, xp, fp)

    bucket_buy = np.diff(buy_at_edge)  # length n_full; buy + sell == bucket_volume
    bucket_imb = np.clip(
        np.abs(2.0 * bucket_buy - bucket_volume) / bucket_volume, 0.0, 1.0
    )

    close_idx = np.searchsorted(cum_total, edges[1:], side="left")
    close_idx = np.clip(close_idx, 0, len(ev) - 1)
    bucket_close_ms = ev[close_idx]  # length n_full, non-decreasing

    # vpin per bucket = trailing mean of n_buckets bucket imbalances.
    csum = np.concatenate(([0.0], np.cumsum(bucket_imb)))
    vpin_bucket = np.full(n_full, np.nan)
    roll = (csum[n_buckets:] - csum[: n_full - n_buckets + 1]) / n_buckets
    vpin_bucket[n_buckets - 1 :] = roll

    prev_close = np.full(n_full, np.nan)
    prev_close[n_buckets:] = bucket_close_ms[: n_full - n_buckets]
    prev_close[n_buckets - 1] = float(ev[0])
    vpin_window_bucket = (bucket_close_ms - prev_close) / 1000.0

    # Map onto the grid: latest bucket closed strictly before each second.
    pos = np.searchsorted(bucket_close_ms, grid, side="left") - 1
    vpin_grid = np.full(len(grid), np.nan)
    vpin_window_grid = np.full(len(grid), np.nan)
    ok = pos >= 0
    vpin_grid[ok] = vpin_bucket[pos[ok]]
    vpin_window_grid[ok] = vpin_window_bucket[pos[ok]]

    return pl.DataFrame(
        {
            "ts_ms": pl.Series("ts_ms", grid, dtype=pl.Int64),
            "vpin": pl.Series("vpin", vpin_grid, dtype=pl.Float64),
            "vpin_window_s": pl.Series(
                "vpin_window_s", vpin_window_grid, dtype=pl.Float64
            ),
        }
    ).with_columns(
        [
            pl.when(pl.col("vpin").is_nan())
            .then(None)
            .otherwise(pl.col("vpin"))
            .alias("vpin"),
            pl.when(pl.col("vpin_window_s").is_nan())
            .then(None)
            .otherwise(pl.col("vpin_window_s"))
            .alias("vpin_window_s"),
        ]
    )


def _compute_vpin_multiscale(
    trades: pl.DataFrame,
    grid_ts_ms: pl.Series,
    base_volume: float,
    n_buckets: int,
) -> pl.DataFrame:
    """VPIN at the three fixed volume scales in VPIN_SCALE_MULTIPLIERS."""
    out = pl.DataFrame({"ts_ms": grid_ts_ms})
    for scale, mult in VPIN_SCALE_MULTIPLIERS.items():
        v = _compute_vpin(
            trades, grid_ts_ms, base_volume * mult, n_buckets
        ).rename(
            {
                "vpin": f"vpin_{scale}",
                "vpin_window_s": f"vpin_window_{scale}_s",
            }
        )
        out = out.join(v, on="ts_ms", how="left")
    return out


def features_one_hour(
    base: Path,
    date_str: str,
    hour: int,
    *,
    overwrite: bool = False,
    base_volume: float = VPIN_BASE_VOLUME,
    n_buckets: int = VPIN_NUM_BUCKETS,
) -> dict:
    out = base / date_str / f"features_{hour:02d}h.parquet"
    if out.exists() and not overwrite:
        return {"hour": hour, "status": "already_done"}

    book_path = base / date_str / f"book20_{hour:02d}h.parquet"
    trades_path = base / date_str / f"trades_{hour:02d}h.parquet"
    mark_path = base / date_str / f"markprice_ffill_{hour:02d}h.parquet"

    if not book_path.exists():
        return {"hour": hour, "status": "missing_book"}
    if not mark_path.exists():
        return {"hour": hour, "status": "missing_mark"}

    start_ms, end_ms = _hour_bounds_ms(date_str, hour)
    prev_d, prev_h = _step_back_hour(date_str, hour)

    # 1) Book at 1s
    book_100ms = pl.read_parquet(book_path)
    book_1s = _book_to_1s(book_100ms, start_ms, end_ms)
    if book_1s.is_empty():
        return {"hour": hour, "status": "empty_book_grid"}
    prev_depth = _read_prev_depth_1s(
        base / prev_d / f"features_{prev_h:02d}h.parquet",
        start_ms - DEPTH_NORM_WINDOW_S * 1000,
        start_ms,
    )
    book_1s = _add_book_features(book_1s, prev_depth=prev_depth)

    # 2) MarkPrice on 1s grid (already aligned)
    mark = pl.read_parquet(mark_path).filter(
        (pl.col("ts_ms") >= start_ms) & (pl.col("ts_ms") < end_ms)
    )
    mark = mark.rename(
        {
            "MarkPrice": "mark_price",
            "FundingRate": "funding_rate",
            "is_stale": "mark_is_stale",
        }
    ).select(
        [
            "ts_ms",
            "mark_price",
            "funding_rate",
            "seconds_to_funding",
            "funding_interval_s",
            "funding_transition",
            "mark_is_stale",
            pl.col("stale_age_ms").alias("mark_stale_age_ms"),
        ]
    )

    def _read_trades(path: Path, range_start: int, range_end: int) -> pl.DataFrame:
        if not path.exists():
            return pl.DataFrame()
        return (
            pl.read_parquet(path)
            .filter(
                (pl.col("EventTime") >= range_start)
                & (pl.col("EventTime") < range_end)
            )
            .select(["EventTime", "Price", "Quantity", "MakerWasBuyer"])
        )

    max_window_s = max(TRADE_WINDOWS_S)
    lookback_ms = 2 * max_window_s * 1000
    range_start = start_ms - lookback_ms

    trades_this = _read_trades(trades_path, start_ms, end_ms)
    prev_hour = hour - 1
    if prev_hour >= 0:
        prev_path = base / date_str / f"trades_{prev_hour:02d}h.parquet"
    else:
        prev_day = (
            datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
        ).strftime("%Y-%m-%d")
        prev_path = base / prev_day / "trades_23h.parquet"
    trades_prev = _read_trades(prev_path, range_start, start_ms)
    if trades_prev.is_empty():
        trades = trades_this.sort("EventTime")
    elif trades_this.is_empty():
        trades = trades_prev.sort("EventTime")
    else:
        trades = pl.concat(
            [trades_prev, trades_this], how="vertical_relaxed"
        ).sort("EventTime")

    # 4) Trade-window features
    grid_ts = book_1s["ts_ms"]
    feat = book_1s
    for w in TRADE_WINDOWS_S:
        twf = _trade_window_features(trades, grid_ts, w)
        feat = feat.join(twf, on="ts_ms", how="left")

    prev_tail = _read_book_tail(
        base / prev_d / f"book20_{prev_h:02d}h.parquet",
        start_ms - 1100,
        start_ms,
    )
    book_for_ofi = book_100ms.select(_ofi_cols())
    if not prev_tail.is_empty():
        book_for_ofi = pl.concat([prev_tail, book_for_ofi], how="vertical_relaxed")
    ofi = _ofi_to_1s(_ofi_per_tick(book_for_ofi), grid_ts)
    feat = feat.join(ofi, on="ts_ms", how="left")

    max_mult = max(VPIN_SCALE_MULTIPLIERS.values())
    vpin_hist = _collect_trades_back(
        base,
        date_str,
        hour,
        start_ms,
        min_volume=(n_buckets + 3) * base_volume * max_mult,
        max_hours=VPIN_MAX_LOOKBACK_HOURS,
    )
    vpin_parts: list[pl.DataFrame] = []
    if not vpin_hist.is_empty():
        vpin_parts.append(vpin_hist)
    if not trades_this.is_empty():
        vpin_parts.append(trades_this.select(VPIN_TRADE_COLS))
    vpin_trades = (
        pl.concat(vpin_parts, how="vertical_relaxed").sort("EventTime")
        if vpin_parts
        else pl.DataFrame()
    )
    vpin = _compute_vpin_multiscale(vpin_trades, grid_ts, base_volume, n_buckets)
    feat = feat.join(vpin, on="ts_ms", how="left")

    # 5) Join markprice
    feat = feat.join(mark, on="ts_ms", how="left")

    feat = feat.with_columns(
        pl.when(pl.col("mark_stale_age_ms") <= 5_000)
        .then(
            (pl.col("mark_price") - pl.col("mid_price"))
            / pl.col("mid_price")
            * 10_000
        )
        .otherwise(None)
        .alias("basis_bps")
    )

    prev_micro = _read_prev_book_micro(
        base / prev_d / f"book20_{prev_h:02d}h.parquet",
        start_ms - max(max(TRADE_WINDOWS_S), REALIZED_VOL_WINDOW_S) * 1000,
        start_ms,
    )
    micro_lag_src = pl.concat(
        [prev_micro, feat.select(["ts_ms", "micro_price"])],
        how="vertical_relaxed",
    )
    for w in TRADE_WINDOWS_S:
        lagged = micro_lag_src.select(
            (pl.col("ts_ms") + w * 1000).alias("ts_ms"),
            pl.col("micro_price").alias("_micro_lag"),
        )
        feat = (
            feat.join(lagged, on="ts_ms", how="left")
            .with_columns(
                (pl.col("micro_price") / pl.col("_micro_lag"))
                .log()
                .alias(f"micro_return_{w}s")
            )
            .drop("_micro_lag")
        )

    rv = (
        micro_lag_src.sort("ts_ms")
        .with_columns(
            (pl.col("micro_price") / pl.col("micro_price").shift(1))
            .log()
            .alias("_ret_1s")
        )
        .with_columns(
            pl.col("_ret_1s")
            .rolling_std(
                window_size=REALIZED_VOL_WINDOW_S,
                min_samples=REALIZED_VOL_MIN_S,
            )
            .alias(f"realized_vol_{REALIZED_VOL_WINDOW_S}s")
        )
        .select(["ts_ms", f"realized_vol_{REALIZED_VOL_WINDOW_S}s"])
    )
    feat = feat.join(rv, on="ts_ms", how="left")

    drop_cols = [f"bid_p_{i}" for i in range(TOP_N)]
    drop_cols += [f"bid_q_{i}" for i in range(TOP_N)]
    drop_cols += [f"ask_p_{i}" for i in range(TOP_N)]
    drop_cols += [f"ask_q_{i}" for i in range(TOP_N)]
    drop_cols += ["last_local_time_us", "last_update_id", "ticks_since_anchor"]
    drop_cols += [f"vwap_{w}s" for w in TRADE_WINDOWS_S]
    feat = feat.drop([c for c in drop_cols if c in feat.columns])

    # Reorder to put ts_ms and valid first
    cols = feat.columns
    front = [c for c in ("ts_ms", "valid", "anchor_kind", "last_event_time") if c in cols]
    rest = [c for c in cols if c not in front]
    feat = feat.select(front + rest)

    out.parent.mkdir(parents=True, exist_ok=True)
    feat.write_parquet(out, compression="zstd")

    return {
        "hour": hour,
        "status": "ok",
        "rows": feat.height,
        "n_valid": int(feat["valid"].sum() or 0),
        "n_trades_used": trades.height,
        "n_ofi_null": int(feat["ofi_l1"].is_null().sum()),
        "n_vpin_null": {
            s: int(feat[f"vpin_{s}"].is_null().sum())
            for s in VPIN_SCALE_MULTIPLIERS
        },
    }


def process_date(
    base: Path,
    date_str: str,
    *,
    overwrite: bool = False,
    base_volume: float = VPIN_BASE_VOLUME,
    n_buckets: int = VPIN_NUM_BUCKETS,
) -> None:
    print(f"\nfeatures | {date_str}")
    for h in range(24):
        rep = features_one_hour(
            base,
            date_str,
            h,
            overwrite=overwrite,
            base_volume=base_volume,
            n_buckets=n_buckets,
        )
        if rep["status"] == "ok":
            vn = rep["n_vpin_null"]
            print(
                f"   {h:02d}h rows={rep['rows']} valid={rep['n_valid']} "
                f"trades_used={rep['n_trades_used']} "
                f"ofi_null={rep['n_ofi_null']} "
                f"vpin_null=micro:{vn['micro']}/base:{vn['base']}/"
                f"macro:{vn['macro']}"
            )
            if rep["rows"] and vn["base"] == rep["rows"]:
                print(
                    "        base VPIN all-NULL, trade history too short to "
                    "warm up; check VPIN_BASE_VOLUME calibration / data gaps."
                )
        elif rep["status"] in ("missing_book", "missing_mark", "empty_book_grid"):
            print(f"   {h:02d}h {rep['status']}")
        elif rep["status"] == "already_done":
            print(f"   {h:02d}h ")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--date")
    p.add_argument("--from-date")
    p.add_argument("--to-date")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--vpin-base-volume",
        type=float,
        default=VPIN_BASE_VOLUME,
        help="VPIN base-scale bucket volume in base-asset units "
        "(micro/macro scales are fixed multiples)",
    )
    p.add_argument(
        "--vpin-num-buckets",
        type=int,
        default=VPIN_NUM_BUCKETS,
        help="number of buckets in the VPIN moving average",
    )
    args = p.parse_args()

    base = Path(args.base)
    if args.date:
        dates = [args.date]
    else:
        all_dates = sorted(
            d.name for d in base.iterdir()
            if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"
        )
        if args.from_date:
            all_dates = [d for d in all_dates if d >= args.from_date]
        if args.to_date:
            all_dates = [d for d in all_dates if d <= args.to_date]
        dates = all_dates

    for d in dates:
        process_date(
            base,
            d,
            overwrite=args.overwrite,
            base_volume=args.vpin_base_volume,
            n_buckets=args.vpin_num_buckets,
        )


if __name__ == "__main__":
    main()
