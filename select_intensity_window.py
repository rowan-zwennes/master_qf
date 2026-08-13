"""Select the rolling intensity window length."""
from __future__ import annotations

import argparse
import json
import math
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
    QA_HEADLINE_H,
    QA_HORIZON_S,
    QA_MIN_FILLS_BIN,
    QA_REPOST_S,
    extract_queue_fill_records,
    queue_aware_fit,
)

DEFAULT_RECORDS_DIR = "reports/intensity_queue_aware"   # cached per-day records
DEFAULT_OUT = "reports/select_intensity_window"
CANDIDATES_S: tuple[int, ...] = tuple(m * 60 for m in (15, 30, 45, 60, 90))  # seconds
EVAL_BLOCK_S = 1800            # 30-min non-overlapping scoring blocks
TIE_REL = 0.01                 # treat deviances within 1% as tied -> shorter W


def poisson_deviance(fit: dict, eval_records: "pl.DataFrame",
                     horizon_s: float, min_fills_bin: int) -> dict:
    """Poisson deviance of the fitted lambda(delta) against the realized fills in
    `eval_records`. Depth bins with at least min_fills_bin observed fills are
    scored individually; sparser bins are folded into one pooled tail cell.

    Per cell (saturated-model Poisson form):
        n > 0 : 2 [ n ln(n / mu) - (n - mu) ]
        n = 0 : 2 mu

    Lower is better."""
    nan_ret = {"dev": float("nan"), "n_bins": 0, "n_tail_bins": 0,
               "tail_n": 0.0, "tail_mu": 0.0}
    if not (math.isfinite(fit.get("A", float("nan")))
            and math.isfinite(fit.get("k", float("nan")))):
        return nan_ret
    ev = queue_aware_fit(eval_records, min_fills_bin=1, horizon_s=horizon_s)
    A, k = fit["A"], fit["k"]
    dev, nbin = 0.0, 0
    tail_n, tail_mu, n_tail = 0.0, 0.0, 0
    for row in ev.get("per_depth", []):
        E = float(row["exposure_s"])
        if E <= 0:
            continue
        n = float(row["fills"])
        mu = A * math.exp(-k * float(row["depth_mean"])) * E
        if n >= min_fills_bin:
            mu = max(mu, 1e-12)
            dev += 2.0 * (n * math.log(n / mu) - (n - mu))
            nbin += 1
        else:                                   # fold into the pooled tail cell
            tail_n += n
            tail_mu += mu
            n_tail += 1
    if tail_mu > 0 or tail_n > 0:               # one Poisson term for the tail
        mu = max(tail_mu, 1e-12)
        dev += (2.0 * (tail_n * math.log(tail_n / mu) - (tail_n - mu))
                if tail_n > 0 else 2.0 * mu)
        nbin += 1
    return {"dev": dev, "n_bins": nbin, "n_tail_bins": n_tail,
            "tail_n": tail_n, "tail_mu": tail_mu}


def evaluate_window(records: "pl.DataFrame", window_s: int, *,
                    eval_block_s: int = EVAL_BLOCK_S,
                    horizon_s: float = QA_HEADLINE_H,
                    min_fills_bin: int = QA_MIN_FILLS_BIN) -> dict:
    """Walk non-overlapping eval blocks over one day's `records`; at each block
    fit on the preceding `window_s` and score the next block. Returns aggregate
    predictive deviance (with per-bin / per-block accounting), the pooled-tail
    diagnostics, and a k-trajectory volatility (stability) proxy."""
    out = {"window_s": window_s, "dev_total": 0.0, "n_blocks_scored": 0,
           "n_blocks_valid_fit": 0, "n_bins_scored": 0, "n_tail_bins": 0,
           "tail_n_total": 0.0, "tail_mu_total": 0.0,
           "k_values": [], "block_dev_per_bin": []}
    if records.is_empty():
        return out
    recs = records.sort("post_ts")
    post_ts = recs["post_ts"].to_numpy()
    t0, t1 = int(post_ts[0]), int(post_ts[-1])
    win_ms, blk_ms = window_s * 1000, eval_block_s * 1000
    # first block start that has a full preceding window inside the day
    t = t0 + win_ms
    while t + blk_ms <= t1:
        lo = int(np.searchsorted(post_ts, t - win_ms, side="left"))
        hi = int(np.searchsorted(post_ts, t, side="left"))
        e_lo = hi
        e_hi = int(np.searchsorted(post_ts, t + blk_ms, side="left"))
        fit_recs = recs.slice(lo, hi - lo)
        eval_recs = recs.slice(e_lo, e_hi - e_lo)
        t += blk_ms
        if fit_recs.height == 0 or eval_recs.height == 0:
            continue
        fit = queue_aware_fit(fit_recs, min_fills_bin=min_fills_bin,
                              horizon_s=horizon_s)
        k = fit.get("k", float("nan"))
        if not (math.isfinite(k) and k > 0.0):   # reject degenerate / non-decaying fits
            continue
        out["n_blocks_valid_fit"] += 1
        out["k_values"].append(float(k))
        d = poisson_deviance(fit, eval_recs, horizon_s, min_fills_bin)
        if math.isfinite(d["dev"]) and d["n_bins"] > 0:
            out["dev_total"] += d["dev"]
            out["n_blocks_scored"] += 1
            out["n_bins_scored"] += d["n_bins"]
            out["n_tail_bins"] += d["n_tail_bins"]
            out["tail_n_total"] += d["tail_n"]
            out["tail_mu_total"] += d["tail_mu"]
            out["block_dev_per_bin"].append(d["dev"] / d["n_bins"])
    return out


def summarize_window(window_s: int, a: dict) -> dict:
    nb = a["n_blocks_scored"]
    nbin = a["n_bins_scored"]
    kv = np.asarray(a["k_values"], dtype=float)          # k>0 by construction
    bdev = np.asarray(a["block_dev_per_bin"], dtype=float)
    return {
        "window_s": window_s, "window_min": window_s / 60.0,
        "dev_total": a["dev_total"],
        # PRIMARY criterion: deviance per scored Poisson cell
        "dev_per_bin": (a["dev_total"] / nbin) if nbin else float("nan"),
        # robustness companion: median over blocks, immune to one outlier block
        "dev_per_bin_median": (float(np.median(bdev)) if bdev.size
                               else float("nan")),
        "dev_per_block": (a["dev_total"] / nb) if nb else float("nan"),
        "n_blocks_scored": nb,
        "n_bins_scored": nbin,
        "n_blocks_valid_fit": a["n_blocks_valid_fit"],
        "n_tail_bins": a["n_tail_bins"],
        "tail_mu_over_n": (a["tail_mu_total"] / a["tail_n_total"]
                           if a["tail_n_total"] > 0 else float("nan")),
        # stability: dispersion of the fitted-k trajectory (lower = steadier)
        "k_median": float(np.median(kv)) if kv.size else float("nan"),
        "k_iqr": (float(np.quantile(kv, 0.75) - np.quantile(kv, 0.25))
                  if kv.size else float("nan")),
        "k_cv": (float(np.std(kv) / np.mean(kv))
                 if kv.size and np.mean(kv) != 0 else float("nan")),
    }


def _load_records(base: Path, records_dir: Path, fday: str
                  ) -> "pl.DataFrame | None":
    """Prefer the cached per-day queue records (reports/intensity_queue_aware/
    queue_records_*.parquet from run_queue_aware); fall back to extracting them
    if the cache is absent."""
    cached = records_dir / f"queue_records_{fday}.parquet"
    if cached.exists():
        return pl.read_parquet(cached)
    from calibrate_intensity import load_for_funding_day
    from data_gap_handler import load_pause_intervals, merge_intervals
    trades = load_for_funding_day(
        base, fday, "trades",
        ["EventTime", "id", "Price", "Quantity", "MakerWasBuyer"])
    book_cols = (["ts_ms", "valid"]
                 + [f"{s}_{w}_{i}" for s in ("bid", "ask")
                    for w in ("p", "q") for i in range(20)])
    book = load_for_funding_day(base, fday, "book20", book_cols)
    if trades.is_empty() or book.is_empty():
        return None
    pauses = merge_intervals(load_pause_intervals(base, fday))
    return extract_queue_fill_records(
        trades, book, horizon_s=QA_HORIZON_S, repost_s=QA_REPOST_S,
        pause_intervals=pauses)


def _new_accumulator() -> dict:
    return {"dev_total": 0.0, "n_blocks_scored": 0, "n_blocks_valid_fit": 0,
            "n_bins_scored": 0, "n_tail_bins": 0,
            "tail_n_total": 0.0, "tail_mu_total": 0.0,
            "k_values": [], "block_dev_per_bin": []}


def _accumulate(a: dict, r: dict) -> None:
    a["dev_total"] += r["dev_total"]
    a["n_blocks_scored"] += r["n_blocks_scored"]
    a["n_blocks_valid_fit"] += r["n_blocks_valid_fit"]
    a["n_bins_scored"] += r["n_bins_scored"]
    a["n_tail_bins"] += r["n_tail_bins"]
    a["tail_n_total"] += r["tail_n_total"]
    a["tail_mu_total"] += r["tail_mu_total"]
    a["k_values"].extend(r["k_values"])
    a["block_dev_per_bin"].extend(r["block_dev_per_bin"])


def run(base: Path, records_dir: Path, out_dir: Path,
        candidates: tuple[int, ...] = CANDIDATES_S,
        eval_block_s: int = EVAL_BLOCK_S,
        horizon_s: float = QA_HEADLINE_H) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = list(pu.load_splits(base)["splits"]["pre_analysis"])
    agg = {W: _new_accumulator() for W in candidates}
    days_used = 0
    for fday in pre:
        recs = _load_records(base, records_dir, fday)
        if recs is None or recs.is_empty():
            print(f"  {fday}: no records, skipped")
            continue
        days_used += 1
        for W in candidates:
            r = evaluate_window(recs, W, eval_block_s=eval_block_s,
                                horizon_s=horizon_s)
            _accumulate(agg[W], r)

    per_window = [summarize_window(W, agg[W]) for W in candidates]

    scored = [w for w in per_window if math.isfinite(w["dev_per_bin"])]
    headline_W = None
    if scored:
        best = min(scored, key=lambda w: w["dev_per_bin"])
        # tie-break: among windows within TIE_REL of the best, pick the SHORTEST
        thr = best["dev_per_bin"] * (1.0 + TIE_REL)
        tied = sorted((w for w in scored if w["dev_per_bin"] <= thr),
                      key=lambda w: w["window_s"])
        headline_W = tied[0]["window_s"]

    result = {
        "method": "leave_future_out_predictive_deviance_per_bin",
        "eval_block_s": eval_block_s, "horizon_s": horizon_s,
        "candidates_s": list(candidates),
        "split": "pre_analysis_only_no_oos_leakage",
        "days_used": days_used,
        "per_window": per_window,
        "headline_window_s": headline_W,
        "headline_window_min": (headline_W / 60.0) if headline_W else None,
        "tie_rel": TIE_REL,
    }
    (out_dir / "select_intensity_window.json").write_text(
        json.dumps(result, indent=2))

    print(f"\n  pre-analysis days used: {days_used}")
    print(f"  {'window':>8} {'dev/bin':>9} {'dev/bin~':>9} {'blocks':>7} "
          f"{'bins':>7} {'tail mu/n':>9} {'k_med':>8} {'k_cv':>7}")
    for w in per_window:
        mark = "  <-- pick" if w["window_s"] == headline_W else ""
        print(f"  {w['window_min']:>6.0f}m {w['dev_per_bin']:>9.3f} "
              f"{w['dev_per_bin_median']:>9.3f} {w['n_blocks_scored']:>7} "
              f"{w['n_bins_scored']:>7} {w['tail_mu_over_n']:>9.3f} "
              f"{w['k_median']:>8.4f} {w['k_cv']:>7.3f}{mark}")
    print(f"\n  headline window: {result['headline_window_min']} min "
          f"(min predictive deviance/bin, ties -> shorter; "
          f"dev/bin~ = per-block median)")
    print(f"  written: {out_dir}/select_intensity_window.json")
    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description="Select the rolling queue-aware intensity window "
                    "(pre-analysis only).")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--records-dir", default=DEFAULT_RECORDS_DIR,
                   help=f"cached queue records dir (default: {DEFAULT_RECORDS_DIR})")
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--candidates-min", type=int, nargs="+", default=None,
                   help="candidate windows in minutes (default: 15 30 45 60 90)")
    p.add_argument("--eval-block-s", type=int, default=EVAL_BLOCK_S)
    p.add_argument("--horizon-s", type=float, default=QA_HEADLINE_H)
    args = p.parse_args()

    if not (_HAVE_PL and _HAVE_PU):
        raise SystemExit("polars/pipeline_utils unavailable")

    cands = (tuple(m * 60 for m in args.candidates_min)
             if args.candidates_min else CANDIDATES_S)
    run(Path(args.base), Path(args.records_dir), Path(args.out_dir),
        candidates=cands, eval_block_s=args.eval_block_s, horizon_s=args.horizon_s)


if __name__ == "__main__":
    main()
