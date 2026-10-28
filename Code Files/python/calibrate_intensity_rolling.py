"""Rolling (A, k) estimates for the simulator."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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

from calibrate_intensity import (
    DEFAULT_BASE,
    MAX_BOOK_STALENESS_MS,  # noqa: F401  (re-exported for diagnostics)
    QA_DEPTH_GRID,
    QA_HEADLINE_H,
    QA_HORIZON_S,
    QA_MIN_FILLS_BIN,
    QA_REPOST_S,
    decay_from_touch,
    extract_queue_fill_records,
    joined_market_orders,
    mle_intensity,
    queue_aware_fit,
)


DEFAULT_OUT = "reports/intensity_rolling"
DEFAULT_OUT_QA = "reports/intensity_qa_rolling"
WINDOW_S_DEFAULT = 15 * 60
WINDOW_S_QA_DEFAULT = 45 * 60          # queue-aware fills are ~20x sparser than
STRIDE_S_DEFAULT = 60
MIN_ORDERS_DEFAULT = 100
MIN_FILLS_WINDOW_QA_DEFAULT = 30       # min queue-aware fills in a window for


@dataclass
class RollConfig:
    window_s: int = WINDOW_S_DEFAULT
    stride_s: int = STRIDE_S_DEFAULT
    min_orders_per_window: int = MIN_ORDERS_DEFAULT
    queue_aware: bool = False
    qa_horizon_s: float = QA_HEADLINE_H
    qa_extract_horizon_s: float = QA_HORIZON_S
    qa_repost_s: float = QA_REPOST_S
    qa_min_fills_window: int = MIN_FILLS_WINDOW_QA_DEFAULT


def rolling_mle(
    joined: "pl.DataFrame",
    window_start_ms: int,
    window_end_ms_exclusive: int,
    cfg: RollConfig,
) -> "pl.DataFrame":
    """Slide a trailing window of length cfg.window_s with stride cfg.stride_s across."""
    win_ms = cfg.window_s * 1000
    stride_ms = cfg.stride_s * 1000
    schema = {
        "ts_ms": pl.Int64,
        "window_start_ms": pl.Int64,
        "window_end_ms": pl.Int64,
        "A": pl.Float64, "k": pl.Float64,
        "A_se": pl.Float64, "k_se": pl.Float64,
        "k_touch": pl.Float64, "penetrate_frac": pl.Float64,
        "seed_A": pl.Float64, "seed_k": pl.Float64, "seed_r2": pl.Float64,
        "n_orders": pl.Int64, "exposure_s": pl.Float64,
        "valid": pl.Boolean,
    }
    if joined.is_empty() or window_end_ms_exclusive <= window_start_ms:
        return pl.DataFrame(schema=schema)

    ts_all = joined["ts_ms"].to_numpy()
    depth_all = joined["depth_mid"].to_numpy()
    reach_all = joined["reach_touch"].to_numpy()

    # window-end candidates strictly inside (window_start_ms, end]
    n_strides = (window_end_ms_exclusive - window_start_ms) // stride_ms
    if n_strides < 1:
        return pl.DataFrame(schema=schema)
    ends = window_start_ms + np.arange(1, int(n_strides) + 1, dtype=np.int64) * stride_ms
    # clip the last to <= end - 1 (we want windows inside the funding day)
    ends = ends[ends <= window_end_ms_exclusive]

    rows: list[dict] = []
    for t_end in ends:
        t_start = int(t_end - win_ms)
        lo = int(np.searchsorted(ts_all, t_start, side="left"))
        hi = int(np.searchsorted(ts_all, t_end, side="left"))
        n_raw = hi - lo
        if n_raw < cfg.min_orders_per_window:
            rows.append({
                "ts_ms": int(t_end),
                "window_start_ms": t_start,
                "window_end_ms": int(t_end),
                "A": float("nan"), "k": float("nan"),
                "A_se": float("nan"), "k_se": float("nan"),
                "k_touch": float("nan"), "penetrate_frac": float("nan"),
                "seed_A": float("nan"), "seed_k": float("nan"), "seed_r2": float("nan"),
                "n_orders": int(n_raw), "exposure_s": float(cfg.window_s),
                "valid": False,
            })
            continue

        d = depth_all[lo:hi]
        r = reach_all[lo:hi]
        d = d[np.isfinite(d) & (d > 0)]
        r = r[np.isfinite(r)]
        if d.size < cfg.min_orders_per_window:
            rows.append({
                "ts_ms": int(t_end),
                "window_start_ms": t_start,
                "window_end_ms": int(t_end),
                "A": float("nan"), "k": float("nan"),
                "A_se": float("nan"), "k_se": float("nan"),
                "k_touch": float("nan"), "penetrate_frac": float("nan"),
                "seed_A": float("nan"), "seed_k": float("nan"), "seed_r2": float("nan"),
                "n_orders": int(d.size), "exposure_s": float(cfg.window_s),
                "valid": False,
            })
            continue

        fit = mle_intensity(d, float(cfg.window_s))
        k_touch, pen_frac = decay_from_touch(r)
        rows.append({
            "ts_ms": int(t_end),
            "window_start_ms": t_start,
            "window_end_ms": int(t_end),
            "A": float(fit.A), "k": float(fit.k),
            "A_se": float(fit.A_se), "k_se": float(fit.k_se),
            "k_touch": float(k_touch), "penetrate_frac": float(pen_frac),
            "seed_A": float(fit.seed_A), "seed_k": float(fit.seed_k), "seed_r2": float(fit.seed_r2),
            "n_orders": int(fit.n_orders), "exposure_s": float(fit.exposure_s),
            "valid": True,
        })

    return pl.DataFrame(rows, schema=schema)


def rolling_queue_aware(
    records: "pl.DataFrame",
    window_start_ms: int,
    window_end_ms_exclusive: int,
    cfg: RollConfig,
) -> "pl.DataFrame":
    """Slide a trailing window of length cfg.window_s with stride cfg.stride_s over
    the queue-aware fill `records` and emit one censoring-aware (A, k) fit per
    stride, in the same schema rolling_mle emits.

    Window endpoints mirror rolling_mle: ts_ms in
    {window_start_ms + j*stride : j >= 1} <= window_end_ms_exclusive, so the
    first row has a full trailing window."""
    win_ms = cfg.window_s * 1000
    stride_ms = cfg.stride_s * 1000
    schema = {
        "ts_ms": pl.Int64,
        "window_start_ms": pl.Int64,
        "window_end_ms": pl.Int64,
        "A": pl.Float64, "k": pl.Float64,
        "A_se": pl.Float64, "k_se": pl.Float64,
        "k_touch": pl.Float64, "penetrate_frac": pl.Float64,
        "seed_A": pl.Float64, "seed_k": pl.Float64, "seed_r2": pl.Float64,
        "n_orders": pl.Int64, "exposure_s": pl.Float64,
        "valid": pl.Boolean,
    }
    if records.is_empty() or window_end_ms_exclusive <= window_start_ms:
        return pl.DataFrame(schema=schema)

    recs = records.sort("post_ts")
    post_ts = recs["post_ts"].to_numpy()

    H_ms = int(round(cfg.qa_horizon_s * 1000.0))

    n_strides = (window_end_ms_exclusive - window_start_ms) // stride_ms
    if n_strides < 1:
        return pl.DataFrame(schema=schema)
    ends = window_start_ms + np.arange(1, int(n_strides) + 1, dtype=np.int64) * stride_ms
    ends = ends[ends <= window_end_ms_exclusive]

    def _invalid_row(t_end: int, t_start: int, n: int) -> dict:
        return {
            "ts_ms": int(t_end), "window_start_ms": int(t_start),
            "window_end_ms": int(t_end),
            "A": float("nan"), "k": float("nan"),
            "A_se": float("nan"), "k_se": float("nan"),
            "k_touch": float("nan"), "penetrate_frac": float("nan"),
            "seed_A": float("nan"), "seed_k": float("nan"),
            "seed_r2": float("nan"),
            "n_orders": int(n), "exposure_s": float(cfg.window_s),
            "valid": False,
        }

    rows: list[dict] = []
    for t_end in ends:
        t_start = int(t_end - win_ms)
        lo = int(np.searchsorted(post_ts, t_start, side="left"))
        hi = int(np.searchsorted(post_ts, t_end - H_ms, side="left"))
        win = recs.slice(lo, hi - lo)
        if win.height == 0:
            rows.append(_invalid_row(int(t_end), t_start, 0))
            continue

        fit = queue_aware_fit(win, min_fills_bin=QA_MIN_FILLS_BIN,
                              horizon_s=cfg.qa_horizon_s)
        # window fill count AFTER re-censoring at the headline horizon
        n_fills = int(win.filter(
            pl.col("filled")
            & (pl.col("time_to_event_s") <= cfg.qa_horizon_s)).height)
        A_side, k = fit["A"], fit["k"]
        valid = (math.isfinite(A_side) and math.isfinite(k) and k > 0.0
                 and n_fills >= cfg.qa_min_fills_window)
        if not valid:
            rows.append(_invalid_row(int(t_end), t_start, win.height))
            continue

        A_pooled = 2.0 * float(A_side)          # see A-CONVENTION note above
        A_se_pooled = 2.0 * float(fit["A_se"]) if math.isfinite(fit["A_se"]) \
            else float("nan")
        rows.append({
            "ts_ms": int(t_end), "window_start_ms": t_start,
            "window_end_ms": int(t_end),
            "A": A_pooled, "k": float(k),
            "A_se": A_se_pooled, "k_se": float(fit["k_se"]),
            "k_touch": float(k),          # queue-aware fit is touch-anchored
            "penetrate_frac": 1.0,        # no separate penetration leg
            "seed_A": float("nan"), "seed_k": float("nan"),
            "seed_r2": float(fit["r2"]),
            "n_orders": int(n_fills), "exposure_s": float(cfg.window_s),
            "valid": True,
        })

    return pl.DataFrame(rows, schema=schema)


def _hour_files_for(base: "Path", paths: list[tuple[str, int]],
                    stem: str, cols: list[str]) -> "pl.DataFrame":
    """Concatenate <base>/<date>/<stem>_<HH>h.parquet for the given
    (date, hour) list. Missing files are skipped (gap hour = absence of
    evidence, mirrors calibrate_volatility._load_features_hours)."""
    frames = []
    for d, h in paths:
        p = base / d / f"{stem}_{h:02d}h.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p).select(cols))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def _warmup_hour_paths(fday: str, lead_seconds: int) -> list[tuple[str, int]]:
    """(calendar_date, hour) tuples covering ~lead_seconds immediately BEFORE
    the funding-day start 04:00 UTC of fday, so the first rolling window
    after 04:00 is full. Same shape as calibrate_volatility._warmup_paths."""
    n_hours = max(1, math.ceil(lead_seconds / 3600))
    start = pu.parse_date(fday).replace(hour=pu.FUNDING_DAY_START_HOUR)
    paths: list[tuple[str, int]] = []
    for i in range(n_hours, 0, -1):
        t = start - timedelta(hours=i)
        paths.append((t.strftime("%Y-%m-%d"), t.hour))
    return paths


def _load_funding_day_with_warmup(
    base: "Path", fday: str, cfg: RollConfig
) -> tuple["pl.DataFrame", "pl.DataFrame", int, int]:
    """Load trades + book20 for funding-day [04:00, 04:00) plus a warm-up
    tail of cfg.window_s seconds. Returns (trades, book, start_ms, end_ms).
    Both frames are sorted by their time key and span
    [start_ms - cfg.window_s*1000, end_ms)."""
    start_ms, end_ms = pu.funding_day_bounds(fday)
    warmup_ms = cfg.window_s * 1000
    warm_paths = _warmup_hour_paths(fday, cfg.window_s)
    day_paths = pu.funding_day_paths(base, fday)
    all_paths = warm_paths + day_paths

    trades = _hour_files_for(
        base, all_paths, "trades", ["EventTime", "id", "Price", "MakerWasBuyer"]
    )
    book = _hour_files_for(
        base, all_paths, "book20",
        ["ts_ms", "last_event_time", "bid_p_0", "ask_p_0"],
    )
    if not trades.is_empty():
        trades = (trades
                  .sort(["EventTime", "id"])
                  .filter((pl.col("EventTime") >= start_ms - warmup_ms)
                          & (pl.col("EventTime") < end_ms)))
    if not book.is_empty():
        book = (book
                .sort("ts_ms")
                .unique(subset=["ts_ms"], keep="first")
                .filter((pl.col("ts_ms") >= start_ms - warmup_ms)
                        & (pl.col("ts_ms") < end_ms)))
    return trades, book, start_ms, end_ms


def run_real(base: "Path", fdays: list[str], out_dir: "Path",
             cfg: RollConfig) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "window_s": cfg.window_s, "stride_s": cfg.stride_s,
        "min_orders_per_window": cfg.min_orders_per_window,
        "method": "trade_through_rolling",
        "boundary": "funding_day_04utc",
        "funding_days": {},
    }
    for fday in fdays:
        trades, book, start_ms, end_ms = _load_funding_day_with_warmup(
            base, fday, cfg
        )
        if trades.is_empty() or book.is_empty():
            summary["funding_days"][fday] = {"status": "no_data"}
            print(f"  {fday}: no data, skipped")
            continue

        joined = joined_market_orders(trades, book)
        out = rolling_mle(joined, start_ms, end_ms, cfg)
        out.write_parquet(
            out_dir / f"intensity_rolling_{fday}.parquet",
            compression="zstd",
        )

        valid = out.filter(pl.col("valid"))
        summary["funding_days"][fday] = {
            "status": "ok",
            "rows": out.height,
            "valid_rows": valid.height,
            "n_trades_in_joined": int(joined.height),
            "A_median": float(valid["A"].median()) if valid.height else None,
            "k_median": float(valid["k"].median()) if valid.height else None,
            "k_touch_median": float(valid["k_touch"].median()) if valid.height else None,
            "n_orders_median": int(valid["n_orders"].median()) if valid.height else None,
        }
        s = summary["funding_days"][fday]
        print(
            f"  {fday}: rows={s['rows']:>5}  valid={s['valid_rows']:>5}  "
            f"A~{s['A_median']}  k~{s['k_median']}  "
            f"k_touch~{s['k_touch_median']}  N~{s['n_orders_median']}"
        )

    (out_dir / "intensity_rolling_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\n  written: {out_dir}/intensity_rolling_*.parquet (+ summary.json)")
    return summary


def _load_qa_day_with_warmup(
    base: "Path", fday: str, cfg: RollConfig
) -> tuple["pl.DataFrame", "pl.DataFrame", int, int]:
    """Like _load_funding_day_with_warmup but loads the FULL 20-level book20
    (prices and sizes) plus the `valid` flag and trade `Quantity`, which the
    queue-aware ratchet kernel needs. Span is [start_ms - window_s, end_ms)."""
    start_ms, end_ms = pu.funding_day_bounds(fday)
    warmup_ms = cfg.window_s * 1000
    all_paths = _warmup_hour_paths(fday, cfg.window_s) + pu.funding_day_paths(base, fday)

    book_cols = (["ts_ms", "valid"]
                 + [f"{s}_{w}_{i}" for s in ("bid", "ask")
                    for w in ("p", "q") for i in range(20)])
    trades = _hour_files_for(
        base, all_paths, "trades",
        ["EventTime", "id", "Price", "Quantity", "MakerWasBuyer"])
    book = _hour_files_for(base, all_paths, "book20", book_cols)
    if not trades.is_empty():
        trades = (trades.sort(["EventTime", "id"])
                  .filter((pl.col("EventTime") >= start_ms - warmup_ms)
                          & (pl.col("EventTime") < end_ms)))
    if not book.is_empty():
        book = (book.sort("ts_ms")
                .unique(subset=["ts_ms"], keep="first")
                .filter((pl.col("ts_ms") >= start_ms - warmup_ms)
                        & (pl.col("ts_ms") < end_ms)))
    return trades, book, start_ms, end_ms


def run_real_queue_aware(base: "Path", fdays: list[str], out_dir: "Path",
                         cfg: RollConfig) -> dict:
    """Rolling QUEUE-AWARE (A, k) per funding day. Runs on ANY split (the
    rolling series is causal: each window looks only backward, so there is no
    OOS leakage, exactly like the trade-through rolling series). The static
    run_queue_aware's pre-analysis-only rule does NOT apply here for the same
    reason it does not apply to rolling_mle."""
    from data_gap_handler import load_pause_intervals, merge_intervals

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "window_s": cfg.window_s, "stride_s": cfg.stride_s,
        "qa_horizon_s": cfg.qa_horizon_s,
        "qa_extract_horizon_s": cfg.qa_extract_horizon_s,
        "qa_repost_s": cfg.qa_repost_s,
        "qa_min_fills_window": cfg.qa_min_fills_window,
        "method": "queue_aware_rolling",
        "A_convention": "pooled (2 x per-side); consumer halves",
        "boundary": "funding_day_04utc",
        "funding_days": {},
    }
    for fday in fdays:
        trades, book, start_ms, end_ms = _load_qa_day_with_warmup(base, fday, cfg)
        if trades.is_empty() or book.is_empty():
            summary["funding_days"][fday] = {"status": "no_data"}
            print(f"  {fday}: no data, skipped")
            continue

        pauses = merge_intervals(load_pause_intervals(base, fday))
        recs = extract_queue_fill_records(
            trades, book,
            horizon_s=cfg.qa_extract_horizon_s, repost_s=cfg.qa_repost_s,
            pause_intervals=pauses)
        out = rolling_queue_aware(recs, start_ms, end_ms, cfg)
        out.write_parquet(
            out_dir / f"intensity_rolling_{fday}.parquet", compression="zstd")

        valid = out.filter(pl.col("valid"))
        summary["funding_days"][fday] = {
            "status": "ok", "rows": out.height, "valid_rows": valid.height,
            "n_records_extracted": int(recs.height),
            "A_pooled_median": float(valid["A"].median()) if valid.height else None,
            "k_median": float(valid["k"].median()) if valid.height else None,
            "r2_exp_median": float(valid["seed_r2"].median()) if valid.height else None,
            "n_fills_median": int(valid["n_orders"].median()) if valid.height else None,
        }
        s = summary["funding_days"][fday]
        print(f"  {fday}: rows={s['rows']:>5}  valid={s['valid_rows']:>5}  "
              f"A_pool~{s['A_pooled_median']}  k~{s['k_median']}  "
              f"fills~{s['n_fills_median']}")

    (out_dir / "intensity_qa_rolling_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\n  written: {out_dir}/intensity_rolling_*.parquet "
          f"(+ intensity_qa_rolling_summary.json)")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Rolling (A, k) for the simulator")
    p.add_argument("--base", default=DEFAULT_BASE,
                   help=f"data root (default: {DEFAULT_BASE})")
    p.add_argument("--out-dir", default=DEFAULT_OUT,
                   help=f"output dir (default: {DEFAULT_OUT})")
    p.add_argument("--window-s", type=int, default=WINDOW_S_DEFAULT,
                   help=f"trailing window length in seconds (default: {WINDOW_S_DEFAULT})")
    p.add_argument("--stride-s", type=int, default=STRIDE_S_DEFAULT,
                   help=f"emit cadence in seconds (default: {STRIDE_S_DEFAULT})")
    p.add_argument("--min-orders", type=int, default=MIN_ORDERS_DEFAULT,
                   help=f"min orders per window for valid=True "
                        f"(default: {MIN_ORDERS_DEFAULT})")
    p.add_argument("--fdays", nargs="+", default=None,
                   help="explicit funding-day list (default: all from splits.json)")
    p.add_argument("--split", choices=["pre_analysis", "sim", "all"],
                   default="all",
                   help="restrict to a split when --fdays not given (default: all)")
    p.add_argument("--queue-aware", action="store_true",
                   help="estimate the last-in-queue FILL hazard (the GLT-"
                        "consistent object) instead of trade-through reach; "
                        "defaults --window-s to %d s and --out-dir to %s"
                        % (WINDOW_S_QA_DEFAULT, DEFAULT_OUT_QA))
    p.add_argument("--qa-horizon-s", type=float, default=QA_HEADLINE_H,
                   help=f"queue-aware re-censoring horizon H "
                        f"(default: {QA_HEADLINE_H})")
    args = p.parse_args()


    if not (_HAVE_PL and _HAVE_PU):
        raise SystemExit("polars/pipeline_utils unavailable")

    base = Path(args.base)
    if args.fdays is not None:
        fdays = list(args.fdays)
    else:
        splits = pu.load_splits(base)["splits"]
        if args.split == "all":
            fdays = list(splits["pre_analysis"]) + list(splits["sim"])
        else:
            fdays = list(splits[args.split])

    if args.queue_aware:
        window_s = (WINDOW_S_QA_DEFAULT if args.window_s == WINDOW_S_DEFAULT
                    else args.window_s)
        out_dir = Path(DEFAULT_OUT_QA if args.out_dir == DEFAULT_OUT
                       else args.out_dir)
        cfg = RollConfig(window_s=window_s, stride_s=args.stride_s,
                         queue_aware=True, qa_horizon_s=args.qa_horizon_s)
        print(f"  rolling QUEUE-AWARE intensity: window={cfg.window_s}s  "
              f"stride={cfg.stride_s}s  H={cfg.qa_horizon_s}s  "
              f"fdays={len(fdays)}  (A column = POOLED = 2 x per-side)")
        run_real_queue_aware(base, fdays, out_dir, cfg)
    else:
        cfg = RollConfig(window_s=args.window_s, stride_s=args.stride_s,
                         min_orders_per_window=args.min_orders)
        out_dir = Path(args.out_dir)
        print(f"  rolling intensity: window={cfg.window_s}s  stride={cfg.stride_s}s  "
              f"min_orders={cfg.min_orders_per_window}  fdays={len(fdays)}")
        run_real(base, fdays, out_dir, cfg)


if __name__ == "__main__":
    main()
