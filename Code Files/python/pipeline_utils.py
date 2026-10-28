"""Shared loaders, split helpers and row filters for the processed dataset."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import polars as pl


FUNDING_DAY_START_HOUR = 4
FUNDING_SETTLEMENT_HOURS_UTC = (0, 8, 16)


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def funding_settlement_times(date_str: str) -> list[int]:
    """The three T_fund timestamps (ms, UTC) for a calendar date."""
    d = parse_date(date_str)
    return [int(d.replace(hour=h).timestamp() * 1000) for h in FUNDING_SETTLEMENT_HOURS_UTC]


def funding_day_bounds(date_str: str) -> tuple[int, int]:
    d = parse_date(date_str).replace(hour=FUNDING_DAY_START_HOUR)
    end = d + timedelta(days=1)
    return int(d.timestamp() * 1000), int(end.timestamp() * 1000)


def funding_day_paths(base: Path, fday: str) -> list[tuple[str, int]]:
    """(calendar_date, hour) tuples covering [04:00 fday, 04:00 fday+1)."""
    out = []
    out += [(fday, h) for h in range(FUNDING_DAY_START_HOUR, 24)]
    next_d = (parse_date(fday) + timedelta(days=1)).strftime("%Y-%m-%d")
    out += [(next_d, h) for h in range(0, FUNDING_DAY_START_HOUR)]
    return out


def load_labels_for_funding_day(
    base: Path, fday: str, stem: str = "labels"
) -> pl.DataFrame:
    """Stitch the labelled parquets across a funding day."""
    start_ms, end_ms = funding_day_bounds(fday)
    frames = []
    for d, h in funding_day_paths(base, fday):
        p = base / d / f"{stem}_{h:02d}h.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p))
    if not frames:
        return pl.DataFrame()
    return (
        pl.concat(frames, how="vertical_relaxed")
        .sort("ts_ms")
        .filter((pl.col("ts_ms") >= start_ms) & (pl.col("ts_ms") < end_ms))
    )


def load_book_for_funding_day(base: Path, fday: str) -> pl.DataFrame:
    """The 100 ms top-20 book stitched across a funding day."""
    start_ms, end_ms = funding_day_bounds(fday)
    frames = []
    for d, h in funding_day_paths(base, fday):
        p = base / d / f"book20_{h:02d}h.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p))
    if not frames:
        return pl.DataFrame()
    return (
        pl.concat(frames, how="vertical_relaxed")
        .sort("ts_ms")
        .filter((pl.col("ts_ms") >= start_ms) & (pl.col("ts_ms") < end_ms))
    )


def load_splits(base: Path) -> dict:
    return json.loads((base / "splits.json").read_text())


def load_features_for_split(
    base: Path,
    split_name: str,
    splits: dict | None = None,
    trim_boundary_s: int = 30,
) -> pl.DataFrame:
    """Concatenate labels for every funding day in a split.

    `trim_boundary_s` must equal max(LABEL_HORIZONS_S). The last
    `trim_boundary_s` seconds of the split are dropped because those rows have
    forward-label targets drawn from prices in the next split period.
    """
    if splits is None:
        splits = load_splits(base)
    if split_name not in splits["splits"]:
        raise KeyError(f"unknown split: {split_name}")
    frames = []
    for fday in splits["splits"][split_name]:
        df = load_labels_for_funding_day(base, fday)
        if not df.is_empty():
            frames.append(df)
    if not frames:
        return pl.DataFrame()
    df = pl.concat(frames, how="vertical_relaxed").sort("ts_ms")

    last_fday = splits["splits"][split_name][-1]
    split_end_ms = funding_day_bounds(last_fday)[1]
    df = df.filter(pl.col("ts_ms") < split_end_ms - trim_boundary_s * 1000)

    return df


def select_training_rows(
    df: pl.DataFrame,
    horizon_s: int,
    *,
    drop_stale: bool = True,
    drop_invalid_book: bool = True,
) -> pl.DataFrame:
    """Rows safe to train the `horizon_s`-second return on."""
    lv = f"label_valid_{horizon_s}"
    if lv not in df.columns:
        raise KeyError(
            f"missing column {lv}. Run stage 04 with horizon {horizon_s}."
        )
    out = df.filter(pl.col(lv))
    if drop_invalid_book:
        out = out.filter(pl.col("valid"))
    if drop_stale and "mark_is_stale" in out.columns:
        out = out.filter(~pl.col("mark_is_stale").fill_null(True))
    return out


def iter_valid_segments(
    book_df: pl.DataFrame,
    *,
    min_segment_seconds: float = 0.0,
) -> Iterator[tuple[int, int, pl.DataFrame]]:
    """Yield (start_ts_ms, end_ts_ms, segment) per contiguous run of valid rows."""
    if book_df.is_empty():
        return
    if "valid" not in book_df.columns:
        raise KeyError("book_df must have a `valid` column")
    valid_int = book_df["valid"].cast(pl.Int8).to_list()
    ts = book_df["ts_ms"].to_list()
    n = len(valid_int)
    i = 0
    while i < n:
        if valid_int[i] != 1:
            i += 1
            continue
        j = i
        while j < n and valid_int[j] == 1:
            j += 1
        seg = book_df.slice(i, j - i)
        seg_start = int(ts[i])
        seg_end = int(ts[j - 1]) + 100  # right-exclusive
        if (seg_end - seg_start) / 1000.0 >= min_segment_seconds:
            yield seg_start, seg_end, seg
        i = j


def describe(base: Path) -> None:
    """Print what is available on disk."""
    splits = None
    if (base / "splits.json").exists():
        splits = load_splits(base)
        print("splits.json found:")
        for s in ("compare", "train", "val", "sim"):
            days = splits["splits"].get(s, [])
            print(f"  {s}: {len(days)} days  [{days[0] if days else '-'} .. {days[-1] if days else '-'}]")
        missing = splits.get("missing_funding_days", [])
        if missing:
            print(f"  {len(missing)} missing days in spec")
    else:
        print("splits.json not found, run stage 04 with --build-splits first")

    cal = sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"
    )
    counts = {"book20": 0, "features": 0, "labels": 0}
    for d in cal:
        for h in range(24):
            for k in counts:
                if (base / d / f"{k}_{h:02d}h.parquet").exists():
                    counts[k] += 1
    print(f"calendar dates on disk: {len(cal)}")
    for k, v in counts.items():
        print(f"  {k}_*.parquet: {v} files")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt")
    args = p.parse_args()
    describe(Path(args.base))
