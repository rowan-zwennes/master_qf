"""Select the ML target horizon from the fill-intensity timescale."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import polars as pl
    from calibrate_intensity import (
        load_for_funding_day, ordered_funding_days, production_folds,
        joined_market_orders, WINDOW_DAYS, MAX_BOOK_STALENESS_MS,
    )
    _HAVE_DATA = True
except Exception:  # pragma: no cover
    _HAVE_DATA = False
    WINDOW_DAYS = 30
    MAX_BOOK_STALENESS_MS = 200


DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
DEFAULT_INTENSITY_DIR = "reports/intensity_calib"
DEFAULT_OUT_DIR = "reports"
CANDIDATES_S: tuple[int, ...] = (1, 5, 10, 30)
BOOK_DT_S = 0.1

JENSEN_BIN_WIDTHS_S: tuple[int, ...] = (30, 60, 300, 600)
JENSEN_MIN_BOOK_FRAC = 0.5   # require >= 50% book coverage to keep a bin


@dataclass
class QueueStats:
    """Per-funding-day pooled aggregates needed for tau_queue."""
    fday: str
    sum_l1_btc: float
    n_book_rows: int
    sum_qty_L1_btc: float
    n_trades_L1: int
    sum_qty_all_btc: float
    n_trades_all: int

    def to_dict(self) -> dict:
        return {
            "fday": self.fday,
            "sum_l1_btc": self.sum_l1_btc, "n_book_rows": self.n_book_rows,
            "sum_qty_L1_btc": self.sum_qty_L1_btc, "n_trades_L1": self.n_trades_L1,
            "sum_qty_all_btc": self.sum_qty_all_btc, "n_trades_all": self.n_trades_all,
            "v_ahead_btc": self.sum_l1_btc / self.n_book_rows if self.n_book_rows else None,
            "L1_volume_share": (self.sum_qty_L1_btc / self.sum_qty_all_btc
                                if self.sum_qty_all_btc else None),
        }


@dataclass
class DayBins:
    """Per-bin (V_avg, L1_volume) arrays for one funding day at ONE bin width."""
    bin_s: int
    V_avg: np.ndarray
    L1_volume_btc: np.ndarray
    n_book_per_bin: np.ndarray


def tau_queue(sum_l1_btc: float, sum_qty_L1_btc: float,
              dt_book_s: float = BOOK_DT_S) -> float:
    """tau_queue = dt_book * sum_l1 / sum_qty_L1."""
    for x in (sum_l1_btc, sum_qty_L1_btc):
        if x is None or not (math.isfinite(x) and x > 0):
            return float("nan")
    return dt_book_s * sum_l1_btc / sum_qty_L1_btc


def snap_to_candidate(tau: float, candidates: tuple[int, ...] = CANDIDATES_S
                      ) -> tuple[int, list[dict]]:
    """Snap tau to the closest candidate in LOG-space. Returns (selected_h,
    table) with table[i] = {'horizon_s': c_i, 'log_distance': |log(tau)-log(c_i)|}
    so the snap is auditable."""
    if not (math.isfinite(tau) and tau > 0):
        table = [{"horizon_s": c, "log_distance": float("nan")} for c in candidates]
        return int(candidates[0]), table
    lt = math.log(tau)
    table = [{"horizon_s": int(c), "log_distance": abs(lt - math.log(c))}
             for c in candidates]
    best = min(table, key=lambda r: r["log_distance"])
    return int(best["horizon_s"]), table


def aggregate(values: list[float], method: str) -> float:
    arr = np.array([v for v in values if math.isfinite(v) and v > 0], dtype=float)
    if arr.size == 0:
        return float("nan")
    if method == "median":
        return float(np.median(arr))
    if method == "geomean":
        return float(math.exp(float(np.mean(np.log(arr)))))
    if method == "mean":
        return float(np.mean(arr))
    raise ValueError(f"unknown aggregation: {method!r}")


def jensen_corrected_tau(jensen_by_day: dict[str, dict[int, DayBins]],
                         train_days: list[str],
                         bin_s: int,
                         min_book_frac: float = JENSEN_MIN_BOOK_FRAC,
                         dt_book_s: float = BOOK_DT_S) -> dict | None:
    """Pool per-bin (V, L1_volume) pairs across a fold's training days at one bin
    width and compute E[V / L1_rate] as the mean over bins of V_bin / rate_bin,
    where rate_bin = L1_volume_bin / bin_s.

    Returns mean / median / quartiles / n_bins / coverage, or None if no valid
    bins remain. Bins below min_book_frac book coverage and bins with zero L1
    volume are dropped."""
    expected_n_book = max(1, int(bin_s / dt_book_s))
    min_n_book = max(1, int(expected_n_book * min_book_frac))
    V_list: list[np.ndarray] = []
    L1_list: list[np.ndarray] = []
    raw_bins = kept_bins = 0
    drop_thin_book = drop_zero_volume = 0
    for d in train_days:
        per_day = jensen_by_day.get(d)
        if per_day is None:
            continue
        db = per_day.get(bin_s)
        if db is None or db.V_avg.size == 0:
            continue
        raw_bins += int(db.V_avg.size)
        thin_book_mask = db.n_book_per_bin < min_n_book
        zero_volume_mask = (~thin_book_mask) & (db.L1_volume_btc <= 0)
        keep_mask = (~thin_book_mask) & (db.L1_volume_btc > 0)
        drop_thin_book += int(thin_book_mask.sum())
        drop_zero_volume += int(zero_volume_mask.sum())
        if keep_mask.any():
            V_list.append(db.V_avg[keep_mask])
            L1_list.append(db.L1_volume_btc[keep_mask])
            kept_bins += int(keep_mask.sum())
    if not V_list:
        return None
    V = np.concatenate(V_list)
    L1 = np.concatenate(L1_list)
    rate = L1 / bin_s
    tau_per_bin = V / rate
    return {
        "bin_s": bin_s,
        "n_bins_kept": kept_bins,
        "n_bins_raw": raw_bins,
        "n_bins_dropped_thin_book": drop_thin_book,
        "n_bins_dropped_zero_volume": drop_zero_volume,
        "tau_mean_s": float(np.mean(tau_per_bin)),     # = E[V / L1_rate]
        "tau_median_s": float(np.median(tau_per_bin)),
        "tau_p25_s": float(np.percentile(tau_per_bin, 25)),
        "tau_p75_s": float(np.percentile(tau_per_bin, 75)),
        "tau_p10_s": float(np.percentile(tau_per_bin, 10)),
        "tau_p90_s": float(np.percentile(tau_per_bin, 90)),
        "rate_p10_btcps": float(np.percentile(rate, 10)),
        "rate_p50_btcps": float(np.percentile(rate, 50)),
        "rate_p90_btcps": float(np.percentile(rate, 90)),
        "rate_mean_btcps": float(np.mean(rate)),
    }


def pool_fold(qs_by_day: dict[str, QueueStats], train_days: list[str]
              ) -> tuple[float, int, float, int, float, int]:
    """Pool per-day QueueStats over a fold's training days. Returns
    (sum_l1_btc, n_book_rows, sum_qty_L1_btc, n_trades_L1,
     sum_qty_all_btc, n_trades_all). Missing days are skipped."""
    sum_l1 = sum_qty_L1 = sum_qty_all = 0.0
    n_book = n_L1 = n_all = 0
    for d in train_days:
        qs = qs_by_day.get(d)
        if qs is None:
            continue
        sum_l1 += qs.sum_l1_btc
        n_book += qs.n_book_rows
        sum_qty_L1 += qs.sum_qty_L1_btc
        n_L1 += qs.n_trades_L1
        sum_qty_all += qs.sum_qty_all_btc
        n_all += qs.n_trades_all
    return sum_l1, n_book, sum_qty_L1, n_L1, sum_qty_all, n_all


def compute_day_queue_stats(
    base: "Path", fday: str,
    jensen_bin_widths_s: tuple[int, ...] = JENSEN_BIN_WIDTHS_S,
) -> tuple[QueueStats, dict[int, DayBins]]:
    """Read book20 + trades for one funding day [04:00, 04:00) and return the
    day's pooled aggregate sums for the at-touch queue formula
    tau = dt_book * sum_l1 / sum_qty_L1, plus per-bin (V_avg, L1_volume) arrays
    at each bin width for jensen_corrected_tau.

    Both paths share the same filtered book rows and the same L1 trade sample;
    the bin computation differs only by a group-by on ts_ms // bin_ms."""
    book = load_for_funding_day(
        base, fday, "book20",
        ["ts_ms", "last_event_time", "bid_p_0", "ask_p_0", "bid_q_0", "ask_q_0"],
    )
    trades = load_for_funding_day(
        base, fday, "trades",
        ["EventTime", "Price", "MakerWasBuyer", "Quantity"],
    )

    bv = None
    sum_l1 = 0.0
    n_book = 0
    if not book.is_empty():
        bv = (book
              .filter(pl.col("last_event_time").is_not_null()
                      & ((pl.col("ts_ms") - pl.col("last_event_time"))
                         <= MAX_BOOK_STALENESS_MS)
                      & (pl.col("bid_p_0") > 0)
                      & (pl.col("ask_p_0") > pl.col("bid_p_0"))
                      & (pl.col("bid_q_0") > 0)
                      & (pl.col("ask_q_0") > 0))
              .select([
                  pl.col("ts_ms"),
                  (pl.col("bid_q_0") + pl.col("ask_q_0")).alias("L1_btc"),
              ]))
        if not bv.is_empty():
            sum_l1 = float(bv["L1_btc"].sum())
            n_book = int(bv.height)
        else:
            bv = None

    jL1 = None
    sum_qty_L1 = sum_qty_all = 0.0
    n_L1 = n_all = 0
    if not trades.is_empty() and not book.is_empty():
        j = joined_market_orders(trades, book)
        if not j.is_empty():
            jq = j.select([
                pl.col("ts_ms"),
                pl.col("Quantity").cast(pl.Float64),
                pl.col("reach_touch").cast(pl.Float64),
            ]).drop_nulls()
            jq = jq.filter(pl.col("Quantity") > 0)
            if not jq.is_empty():
                sum_qty_all = float(jq["Quantity"].sum())
                n_all = int(jq.height)
                # reach_touch == 0 picks the records that took volume from L1.
                jL1 = jq.filter(pl.col("reach_touch") == 0)
                if not jL1.is_empty():
                    sum_qty_L1 = float(jL1["Quantity"].sum())
                    n_L1 = int(jL1.height)
                else:
                    jL1 = None

    jensen: dict[int, DayBins] = {}
    for bin_s in jensen_bin_widths_s:
        jensen[int(bin_s)] = _compute_jensen_for_bin(bv, jL1, int(bin_s))

    queue = QueueStats(
        fday=fday, sum_l1_btc=sum_l1, n_book_rows=n_book,
        sum_qty_L1_btc=sum_qty_L1, n_trades_L1=n_L1,
        sum_qty_all_btc=sum_qty_all, n_trades_all=n_all,
    )
    return queue, jensen


def _compute_jensen_for_bin(bv, jL1, bin_s: int) -> DayBins:
    """Group bv (valid book rows with L1_btc, ts_ms) and jL1 (L1 trades with
    Quantity, ts_ms) into bins of width bin_s seconds. Bins are keyed by
    (ts_ms // (bin_s * 1000)). Returns a DayBins keeping only bins that have
    BOTH at least one book row AND at least one L1 trade, the inner join
    drops bins where either side is empty (which would force a NaN ratio
    downstream anyway)."""
    empty = DayBins(bin_s=bin_s, V_avg=np.empty(0),
                    L1_volume_btc=np.empty(0), n_book_per_bin=np.empty(0, dtype=int))
    if bv is None or jL1 is None or bv.is_empty() or jL1.is_empty():
        return empty
    bin_ms = bin_s * 1000
    b_binned = (bv
                .with_columns((pl.col("ts_ms") // bin_ms).alias("bin_id"))
                .group_by("bin_id")
                .agg([
                    pl.col("L1_btc").mean().alias("V_avg"),
                    pl.col("L1_btc").count().alias("n_book"),
                ]))
    t_binned = (jL1
                .with_columns((pl.col("ts_ms") // bin_ms).alias("bin_id"))
                .group_by("bin_id")
                .agg([pl.col("Quantity").sum().alias("L1_volume")]))
    merged = b_binned.join(t_binned, on="bin_id", how="inner")
    if merged.is_empty():
        return empty
    return DayBins(
        bin_s=bin_s,
        V_avg=merged["V_avg"].to_numpy(),
        L1_volume_btc=merged["L1_volume"].to_numpy(),
        n_book_per_bin=merged["n_book"].to_numpy().astype(int),
    )


def _load_intensity_folds(intensity_dir: Path) -> list[dict]:
    files = sorted(intensity_dir.glob("intensity_calib_fold*.json"))
    return [json.loads(f.read_text()) for f in files]


def run(base: Path, intensity_dir: Path, out_dir: Path) -> dict:
    fold_jsons = _load_intensity_folds(intensity_dir)
    if not fold_jsons:
        raise SystemExit(f"no intensity_calib_fold*.json found in {intensity_dir}")
    fold0_calib = next((f for f in fold_jsons if f.get("fold_id") == 0), None)
    if fold0_calib is None:
        raise SystemExit("fold 0 missing from intensity calibration; cannot derive headline.")

    # 1) Per-day book + trades aggregates (one pass over 94 funding days).
    days = ordered_funding_days(base)
    print(f"computing per-day queue stats over {len(days)} funding days ...")
    qs_by_day: dict[str, QueueStats] = {}
    jensen_by_day: dict[str, dict[int, DayBins]] = {}
    for d in days:
        qs, jbins = compute_day_queue_stats(base, d, JENSEN_BIN_WIDTHS_S)
        qs_by_day[d] = qs
        jensen_by_day[d] = jbins
        v_avg = qs.sum_l1_btc / qs.n_book_rows if qs.n_book_rows else float("nan")
        T_day = qs.n_book_rows * BOOK_DT_S
        L1_rate = qs.sum_qty_L1_btc / T_day if T_day > 0 else float("nan")
        L1_share = (qs.sum_qty_L1_btc / qs.sum_qty_all_btc
                    if qs.sum_qty_all_btc > 0 else float("nan"))
        print(f"  {d}: V_avg={v_avg:.3f} BTC  L1_rate={L1_rate:.4f} BTC/s  "
              f"L1_share={L1_share:.3f}  n_book={qs.n_book_rows}  "
              f"n_L1={qs.n_trades_L1}  n_all={qs.n_trades_all}")

    # 2) Per-fold pooling over the 30-day training window.
    folds = production_folds(days, WINDOW_DAYS)
    per_fold = []
    for f in folds:
        sum_l1, n_book, sum_qty_L1, n_L1, sum_qty_all, n_all = pool_fold(qs_by_day, f["train_days"])
        tau = tau_queue(sum_l1, sum_qty_L1, BOOK_DT_S)
        snap_h, _ = snap_to_candidate(tau)
        T_fold = n_book * BOOK_DT_S
        per_fold.append({
            "fold_id": f["fold_id"],
            "train_start": f["train_days"][0], "train_end": f["train_days"][-1],
            "predict_day": f["predict_day"],
            "V_avg_btc": sum_l1 / n_book if n_book > 0 else None,
            "L1_depletion_pooled_btcps": sum_qty_L1 / T_fold if T_fold > 0 else None,
            "L1_volume_share": sum_qty_all > 0 and (sum_qty_L1 / sum_qty_all) or None,
            "n_book_rows": n_book, "n_trades_L1": n_L1, "n_trades_all": n_all,
            "T_fold_s": T_fold,
            "tau_queue_s": tau if math.isfinite(tau) else None,
            "fold_snap_horizon_s": int(snap_h) if math.isfinite(tau) else None,
        })

    # 3) Headline = fold 0 only.
    pf0 = next((pf for pf in per_fold if pf["fold_id"] == 0), None)
    if pf0 is None or pf0["tau_queue_s"] is None:
        raise SystemExit("fold 0 tau_queue could not be computed; aborting.")
    tau0 = float(pf0["tau_queue_s"])
    headline_h, snap_table = snap_to_candidate(tau0)

    # 4) Stability diagnostic across all 64 folds.
    taus = [pf["tau_queue_s"] for pf in per_fold if pf["tau_queue_s"] is not None]
    tau_med = aggregate(taus, "median")
    tau_gm = aggregate(taus, "geomean")
    med_h, _ = snap_to_candidate(tau_med)

    fold0_train_days = next(f["train_days"] for f in folds if f["fold_id"] == 0)
    jensen_fold0: dict[str, dict | None] = {}
    for bin_s in JENSEN_BIN_WIDTHS_S:
        jensen_fold0[str(int(bin_s))] = jensen_corrected_tau(
            jensen_by_day, fold0_train_days, int(bin_s),
            min_book_frac=JENSEN_MIN_BOOK_FRAC, dt_book_s=BOOK_DT_S,
        )

    result = {
        "headline_horizon_s": int(headline_h),
        "headline_source": "fold 0 (pre-analysis only; no OOS leakage)",
        "tau_queue_fold0_s": tau0,
        "fold0_inputs": {
            "V_avg_btc": pf0["V_avg_btc"],
            "L1_depletion_pooled_btcps": pf0["L1_depletion_pooled_btcps"],
            "L1_volume_share": pf0["L1_volume_share"],
            "T_fold_s": pf0["T_fold_s"],
            "from_calibration": {
                "A_pooled": fold0_calib.get("A"),
                "k_touch": fold0_calib.get("k_touch"),
                "A_touch_pooled": fold0_calib.get("A_touch"),
                "A_touch_per_side": fold0_calib.get("A_touch_per_side"),
                "penetrate_frac": fold0_calib.get("penetrate_frac"),
            },
        },
        "snap_distances_fold0": snap_table,
        "candidates_s": list(CANDIDATES_S),
        "method": ("tau_queue = dt_book * sum_l1 / sum_qty_L1; "
                   "sum_l1 over valid book rows (bid_q_0 + ask_q_0); "
                   "sum_qty_L1 over trades with reach_touch == 0; "
                   "dt_book = 0.1 s (book20 grid)."),
        "stability_diagnostic": {
            "n_folds": len(taus),
            "tau_median_s": tau_med, "tau_geomean_s": tau_gm,
            "median_snap_horizon_s": int(med_h),
            "tau_min_s": min(taus) if taus else None,
            "tau_max_s": max(taus) if taus else None,
            "headline_matches_median": bool(int(med_h) == int(headline_h)),
        },
        "jensen_corrected_fold0": {
            "lower_bound_s": tau0,         # E[V]/E[rate], the implemented formula
            "by_bin_s": jensen_fold0,      # E[V/rate] at each bin width
            "min_book_frac_required": JENSEN_MIN_BOOK_FRAC,
            "note": ("lower_bound_s is E[V]/E[L1_rate] (the headline). "
                     "by_bin_s[<w>] is E[V/L1_rate] at bin width <w> seconds. "
                     "Jensen + V-rate anti-correlation imply by_bin_s >= lower_bound_s."),
        },
        "per_fold": per_fold,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "select_horizon.json").write_text(json.dumps(result, indent=2))

    print(f"\nfold 0 inputs:  V_avg = {pf0['V_avg_btc']:.3f} BTC  "
          f"L1_depletion = {pf0['L1_depletion_pooled_btcps']:.4f} BTC/s (pooled both sides)  "
          f"L1_share = {pf0['L1_volume_share']:.3f}")
    print(f"fold 0 tau_queue = {tau0:.3f} s  ->  HEADLINE HORIZON = {headline_h} s")
    print("snap distances (log): "
          + "  ".join(f"{r['horizon_s']}s:{r['log_distance']:.3f}" for r in snap_table))
    sd = result["stability_diagnostic"]
    print(f"stability across {sd['n_folds']} folds: "
          f"tau median={sd['tau_median_s']:.3f}s geomean={sd['tau_geomean_s']:.3f}s "
          f"range [{sd['tau_min_s']:.3f}, {sd['tau_max_s']:.3f}]s  "
          f"median->snap = {sd['median_snap_horizon_s']}s "
          f"({'matches' if sd['headline_matches_median'] else 'DIFFERS from'} headline)")
    print(f"\nJensen-corrected E[V / L1_rate] on fold 0 "
          f"(lower bound E[V]/E[rate] = {tau0:.3f} s):")
    for bin_s in JENSEN_BIN_WIDTHS_S:
        j = jensen_fold0[str(int(bin_s))]
        if j is None:
            print(f"  bin={bin_s:>4}s: (no valid bins)")
            continue
        snap_j, _ = snap_to_candidate(j["tau_mean_s"])
        print(f"  bin={bin_s:>4}s: kept={j['n_bins_kept']:>6}/{j['n_bins_raw']:>6}  "
              f"dropped thin={j['n_bins_dropped_thin_book']:>5} "
              f"zero={j['n_bins_dropped_zero_volume']:>5}  "
              f"E[V/rate]={j['tau_mean_s']:7.3f}s  median={j['tau_median_s']:6.3f}s  "
              f"IQR=[{j['tau_p25_s']:6.3f},{j['tau_p75_s']:6.3f}]s  "
              f"rate p10/50/90={j['rate_p10_btcps']:.3f}/{j['rate_p50_btcps']:.3f}/"
              f"{j['rate_p90_btcps']:.3f} BTC/s  snap={snap_j}s")
    print(f"\nwritten: {out_dir / 'select_horizon.json'}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Headline ML horizon from the AS intensity.")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--intensity-dir", default=DEFAULT_INTENSITY_DIR,
                   help="directory of intensity_calib_fold*.json (from calibrate_intensity.py)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    if not _HAVE_DATA:
        raise SystemExit("polars/pipeline_utils/calibrate_intensity unavailable")
    run(Path(args.base), Path(args.intensity_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
