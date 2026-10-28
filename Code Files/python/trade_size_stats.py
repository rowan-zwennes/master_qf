"""Aggressing-trade size distribution over the 30 pre-analysis funding days."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from pipeline_utils import funding_day_bounds, funding_day_paths

DATA_ROOT = Path("/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt")
N_PRE = 30
PCTS = [25, 50, 75, 90, 95, 99]


def hist_add(acc: np.ndarray, milli: np.ndarray) -> np.ndarray:
    """Accumulate a bincount of integer milli-BTC sizes into `acc`."""
    bc = np.bincount(milli)
    if bc.size > acc.size:
        acc = np.concatenate([acc, np.zeros(bc.size - acc.size, dtype=np.int64)])
    acc[: bc.size] += bc
    return acc


def hist_percentile(acc: np.ndarray, p: float) -> float:
    """Percentile from a milli-BTC histogram."""
    n = acc.sum()
    cum = np.cumsum(acc)
    rank = p / 100.0 * (n - 1)
    lo = int(np.searchsorted(cum, np.floor(rank) + 1))
    return lo / 1000.0


def main() -> None:
    splits = json.loads((DATA_ROOT / "splits.json").read_text())
    pre_days = splits["available_funding_days"][:N_PRE]

    hist = np.zeros(1, dtype=np.int64)
    tot_sum, tot_n = 0.0, 0
    per_day: list[dict] = []

    for fday in pre_days:
        start_ms, end_ms = funding_day_bounds(fday)
        frames = []
        for d, h in funding_day_paths(DATA_ROOT, fday):
            p = DATA_ROOT / d / f"trades_{h:02d}h.parquet"
            if p.exists():
                frames.append(pl.read_parquet(p, columns=["EventTime", "Quantity"]))
        df = pl.concat(frames).filter(
            (pl.col("EventTime") >= start_ms) & (pl.col("EventTime") < end_ms)
        )
        q = df["Quantity"].to_numpy()
        milli = np.rint(q * 1000.0).astype(np.int64)
        hist = hist_add(hist, milli)
        tot_sum += float(q.sum())
        tot_n += q.size
        per_day.append({"funding_day": fday, "n": int(q.size),
                        "median": float(np.median(q)), "mean": float(q.mean())})
        print(f"  {fday} : n={q.size:>9,}  median={np.median(q):.4f}  mean={q.mean():.4f}")

    out = {
        "window": f"04:00 {pre_days[0]} -> 04:00 next of {pre_days[-1]} "
                  f"({N_PRE} pre-analysis funding days)",
        "alignment": "funding-day (04:00 UTC -> 04:00 UTC), pipeline_utils",
        "n_trades": int(tot_n),
        "pooled_mean": tot_sum / tot_n,
        "pooled_median": hist_percentile(hist, 50),
        "pooled_percentiles_btc": {f"p{p}": hist_percentile(hist, p) for p in PCTS},
        "median_of_daily_medians": float(np.median([d["median"] for d in per_day])),
        "per_day": per_day,
    }
    Path("reports").mkdir(exist_ok=True)
    with open("reports/trade_size_stats.json", "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\npooled over {N_PRE} pre-analysis funding days")
    print(f"  n trades      : {tot_n:,}")
    print(f"  mean  trade   : {out['pooled_mean']:.4f} BTC")
    print(f"  median trade  : {out['pooled_median']:.4f} BTC")
    print("  percentiles   : " + "  ".join(f"{k}={v:.4f}"
          for k, v in out["pooled_percentiles_btc"].items()))
    print("  -> reports/trade_size_stats.json")


if __name__ == "__main__":
    main()
