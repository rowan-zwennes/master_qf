"""Hard-pause intervals from the reconstructed book's invalid intervals."""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from pipeline_utils import funding_day_bounds, funding_day_paths, load_splits

BASE_DEFAULT = Path("/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt")

# post-resume warm-up before queue positions are trusted again
WARMUP_MS = 5_000

Interval = tuple[int, int]


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Sort and merge overlapping or touching [start, end) intervals."""
    if not intervals:
        return []
    ivs = sorted(intervals)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def load_pause_intervals(
    base: Path, fday: str, *, warmup_ms: int = WARMUP_MS
) -> list[Interval]:
    """Pause intervals (ms, half-open) for one funding day [04:00, +24h)."""
    start_ms, end_ms = funding_day_bounds(fday)
    raw: list[Interval] = []
    for d, h in funding_day_paths(base, fday):
        p = base / d / f"book20_invalid_{h:02d}h.parquet"
        if not p.exists():
            continue
        df = pl.read_parquet(p)
        for s, e in zip(df["ts_ms_start"], df["ts_ms_end"]):
            raw.append((int(s), int(e) + warmup_ms))
    merged = merge_intervals(raw)
    return [
        (max(s, start_ms), min(e, end_ms))
        for s, e in merged
        if s < end_ms and e > start_ms
    ]


def in_pause(ts_ms: int, intervals: list[Interval]) -> bool:
    """True if ts_ms falls inside any pause interval."""
    return any(s <= ts_ms < e for s, e in intervals)


def downtime_report(
    base: Path, days: list[str], *, warmup_ms: int = WARMUP_MS
) -> pl.DataFrame:
    """Per-funding-day downtime table."""
    rows = []
    for d in days:
        ivs = load_pause_intervals(base, d, warmup_ms=warmup_ms)
        paused_ms = sum(e - s for s, e in ivs)
        rows.append({
            "funding_day": d,
            "n_pauses": len(ivs),
            "paused_s": paused_ms / 1000.0,
            "paused_pct": 100.0 * paused_ms / 86_400_000.0,
        })
    return pl.DataFrame(rows)


def exclude_contaminated_rows(
    df: pl.DataFrame,
    intervals: list[Interval],
    *,
    lookback_s: float,
    horizon_s: float,
    ts_col: str = "ts_ms",
) -> pl.Series:
    """Keep-mask: False where [t - lookback, t + horizon] overlaps a pause."""
    keep = pl.Series([True] * df.height)
    if not intervals:
        return keep
    lb_ms = int(lookback_s * 1000)
    hz_ms = int(horizon_s * 1000)
    ts = df[ts_col]
    for s, e in intervals:
        overlap = (ts + hz_ms >= s) & (ts - lb_ms < e)
        keep = keep & ~overlap
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description="Gap-pause intervals and downtime report.")
    ap.add_argument("--base", type=Path, default=BASE_DEFAULT)
    ap.add_argument("--out", type=Path, default=Path("reports/downtime_report.csv"))
    args = ap.parse_args()

    splits = load_splits(args.base)["splits"]
    days = splits["pre_analysis"] + splits["sim"]
    rep = downtime_report(args.base, days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rep.write_csv(args.out)
    tot = rep["paused_s"].sum()
    print(rep)
    print(f"total downtime: {tot:.1f} s over {len(days)} days "
          f"({100 * tot / (86_400 * len(days)):.4f}%)  ->  {args.out}")


if __name__ == "__main__":
    main()
