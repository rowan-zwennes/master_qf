"""Stage 04: multi-horizon labels, funding-day alignment and splits."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
LABEL_HORIZONS_S = [1, 5, 10, 30]
FUNDING_DAY_START_HOUR = 4  # 04:00 UTC


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def funding_day_bounds(date_str: str) -> tuple[int, int]:
    """Returns [start_ms, end_ms) for the funding day starting on `date_str`."""
    d = parse_date(date_str).replace(hour=FUNDING_DAY_START_HOUR)
    end = d + timedelta(days=1)
    return int(d.timestamp() * 1000), int(end.timestamp() * 1000)


def load_concatenated_features(
    base: Path, dates: list[str], start_ms: int, end_ms: int
) -> pl.DataFrame:
    """Concatenate features across the dates needed to cover [start_ms, end_ms)."""
    frames = []
    for d in dates:
        for h in range(24):
            p = base / d / f"features_{h:02d}h.parquet"
            if p.exists():
                frames.append(pl.read_parquet(p))
    if not frames:
        return pl.DataFrame()
    full = (
        pl.concat(frames, how="vertical_relaxed")
        .unique(subset="ts_ms", keep="first")
        .sort("ts_ms")
    )
    return full.filter(
        (pl.col("ts_ms") >= start_ms) & (pl.col("ts_ms") < end_ms)
    )


def add_labels(
    df: pl.DataFrame,
    horizons: list[int],
    grid_start_ms: int,
    grid_end_ms: int,
) -> pl.DataFrame:
    """Adds: target_mid_<h>, target_micro_<h>, return_mid_<h>, return_micro_<h>."""
    if df.is_empty():
        return df

    full_grid = pl.DataFrame(
        {"ts_ms": list(range(grid_start_ms, grid_end_ms, 1000))}
    ).with_columns(pl.col("ts_ms").cast(pl.Int64))

    df = full_grid.join(df, on="ts_ms", how="left").with_columns(
        pl.col("valid").fill_null(False)
    )

    new_cols = []
    for h in horizons:
        target_mid = pl.col("mid_price").shift(-h).alias(f"target_mid_{h}")
        target_micro = pl.col("micro_price").shift(-h).alias(f"target_micro_{h}")
        new_cols.extend([target_mid, target_micro])

    df = df.with_columns(new_cols)

    for h in horizons:
        forward_window = pl.col("valid")
        for k in range(1, h + 1):
            forward_window = forward_window & pl.col("valid").shift(-k)
        df = df.with_columns(forward_window.alias(f"_label_window_{h}"))

    df = df.with_columns(
        [
            (pl.col(f"target_mid_{h}") / pl.col("mid_price")).log().alias(
                f"return_mid_{h}"
            )
            for h in horizons
        ]
        + [
            (pl.col(f"target_micro_{h}") / pl.col("micro_price")).log().alias(
                f"return_micro_{h}"
            )
            for h in horizons
        ]
        + [
            ((pl.col(f"target_mid_{h}") - pl.col("mid_price")) / h).alias(
                f"drift_mid_{h}"
            )
            for h in horizons
        ]
        + [
            ((pl.col(f"target_micro_{h}") - pl.col("micro_price")) / h).alias(
                f"drift_micro_{h}"
            )
            for h in horizons
        ]
    )

    # label_valid_h: window valid AND both endpoint prices non-null
    for h in horizons:
        df = df.with_columns(
            (
                pl.col(f"_label_window_{h}")
                & pl.col("mid_price").is_not_null()
                & pl.col(f"target_mid_{h}").is_not_null()
                & pl.col("micro_price").is_not_null()
                & pl.col(f"target_micro_{h}").is_not_null()
            ).alias(f"label_valid_{h}")
        )

    df = df.drop([f"_label_window_{h}" for h in horizons])
    return df


def write_labels_per_hour(df: pl.DataFrame, base: Path) -> int:
    """Splits the labelled frame back into per-hour parquets keyed by date+HH."""
    if df.is_empty():
        return 0
    df = df.with_columns(
        [
            pl.from_epoch("ts_ms", time_unit="ms").dt.strftime("%Y-%m-%d").alias("_date"),
            pl.from_epoch("ts_ms", time_unit="ms").dt.hour().alias("_hour"),
        ]
    )
    n_written = 0
    for key, grp in df.partition_by(
        ["_date", "_hour"], as_dict=True
    ).items():
        # partition_by with multiple keys returns a tuple key.
        date_str, hour = key
        out = base / date_str / f"labels_{hour:02d}h.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        grp.drop(["_date", "_hour"]).write_parquet(out, compression="zstd")
        n_written += 1
    return n_written


def funding_day_labels_exist(base: Path, fday_date: str) -> bool:
    """True iff all 24 labels_HHh.parquet of the funding day already exist.

    A funding day's outputs are hours 04..23 of `fday_date` and hours 00..03
    of the following calendar date.
    """
    next_d = (parse_date(fday_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    for h in range(4, 24):
        if not (base / fday_date / f"labels_{h:02d}h.parquet").exists():
            return False
    for h in range(0, 4):
        if not (base / next_d / f"labels_{h:02d}h.parquet").exists():
            return False
    return True


def process_funding_day(
    base: Path, fday_date: str, *, overwrite: bool = False, horizons: list[int] = LABEL_HORIZONS_S
) -> dict:
    """
    Compute labels for the funding day starting on `fday_date` at 04:00 UTC.
    """
    if not overwrite and funding_day_labels_exist(base, fday_date):
        return {"funding_day": fday_date, "status": "skipped_exists"}

    start_ms, end_ms = funding_day_bounds(fday_date)
    horizon_buffer_ms = max(horizons) * 1000

    end_ms_plus = end_ms + horizon_buffer_ms

    # Calendar dates touched by [start_ms, end_ms_plus]
    s_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    e_dt = datetime.fromtimestamp(end_ms_plus / 1000, tz=timezone.utc)
    needed = []
    cur = s_dt.date()
    while cur <= e_dt.date():
        needed.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)

    df = load_concatenated_features(
        base, needed, start_ms, end_ms_plus
    )
    if df.is_empty():
        return {"funding_day": fday_date, "status": "missing_features"}

    df = add_labels(df, horizons, start_ms, end_ms_plus)

    # Trim to the funding-day proper 
    df = df.filter(pl.col("ts_ms") < end_ms)

    df = df.with_columns(
        [
            pl.lit(fday_date).alias("funding_day"),
            ((pl.col("ts_ms") - start_ms) // 1000).alias("seconds_into_funding_day"),
        ]
    )

    n_written = write_labels_per_hour(df, base)

    valid_summary = {}
    for h in horizons:
        col = f"label_valid_{h}"
        if col in df.columns:
            n_v = int(df[col].sum() or 0)
            valid_summary[h] = {"n_valid": n_v, "n_total": df.height}

    return {
        "funding_day": fday_date,
        "status": "ok",
        "rows": df.height,
        "files_written": n_written,
        "label_validity": valid_summary,
    }


def build_splits(
    first_funding_day: str,
    n_pre_analysis: int,
    n_sim: int,
) -> dict:
    """Returns the split spec as a dict."""
    start = parse_date(first_funding_day)

    def days_after(start: datetime, n: int) -> list[str]:
        return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]

    pre_analysis_days = days_after(start, n_pre_analysis)
    sim_days = days_after(start + timedelta(days=n_pre_analysis), n_sim)

    return {
        "first_funding_day": first_funding_day,
        "n_pre_analysis": n_pre_analysis,
        "n_sim": n_sim,
        "splits": {
            "pre_analysis": pre_analysis_days,
            "sim": sim_days,
        },
        "funding_day_start_hour_utc": FUNDING_DAY_START_HOUR,
        "label_horizons_s": LABEL_HORIZONS_S,
    }


def list_available_funding_days(base: Path) -> list[str]:
    """A funding day is "available" if all 24 hour-files exist at the underlying
    calendar dates spanning [04:00, +24h).

    We just check that hours 04..23 of date D AND hours 00..03 of date D+1 are
    present as features_*.parquet files.
    """
    cal_dates = sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"
    )
    if not cal_dates:
        return []

    cal_set = set(cal_dates)
    def has_hour(d: str, h: int) -> bool:
        return (base / d / f"features_{h:02d}h.parquet").exists()

    out = []
    for d in cal_dates:
        next_d = (parse_date(d) + timedelta(days=1)).strftime("%Y-%m-%d")
        if next_d not in cal_set:
            continue
        ok = all(has_hour(d, h) for h in range(4, 24)) and all(
            has_hour(next_d, h) for h in range(0, 4)
        )
        if ok:
            out.append(d)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--funding-day", help="single YYYY-MM-DD")
    p.add_argument("--from-day", help="first funding day inclusive")
    p.add_argument("--to-day", help="last funding day inclusive")
    p.add_argument("--build-splits", action="store_true",
                   help="Build splits.json instead of (or in addition to) labelling")
    p.add_argument("--first-funding-day", default=None,
                   help="for --build-splits: first FD")
    p.add_argument("--n-pre-analysis", type=int, default=30,
                   help="size of the pre-analysis block (default 30)")
    p.add_argument("--n-sim", type=int, default=64,
                   help="size of the out-of-sample sim block (default 64)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    base = Path(args.base)

    available = list_available_funding_days(base)
    print(f"Available funding days: {len(available)}")
    if available:
        print(f"   first: {available[0]}, last: {available[-1]}")

    if args.build_splits:
        first = args.first_funding_day or (available[0] if available else None)
        if first is None:
            raise SystemExit("No --first-funding-day and no available days.")
        spec = build_splits(
            first,
            n_pre_analysis=args.n_pre_analysis,
            n_sim=args.n_sim,
        )
        # Sanity: are all spec days available?
        all_days = (
            spec["splits"]["pre_analysis"]
            + spec["splits"]["sim"]
        )
        missing = [d for d in all_days if d not in available]
        spec["available_funding_days"] = available
        spec["missing_funding_days"] = missing
        out_path = base / "splits.json"
        out_path.write_text(json.dumps(spec, indent=2))
        print(f" wrote {out_path}")
        if missing:
            print(f"   {len(missing)} funding-days in spec are not yet available:")
            for m in missing[:5]:
                print(f"       {m}")
            if len(missing) > 5:
                print(f"       ... and {len(missing) - 5} more")


    if args.funding_day:
        days = [args.funding_day]
    else:
        days = available
        if args.from_day:
            days = [d for d in days if d >= args.from_day]
        if args.to_day:
            days = [d for d in days if d <= args.to_day]

    print(f" Labelling {len(days)} funding days")
    for d in days:
        rep = process_funding_day(base, d, overwrite=args.overwrite)
        if rep["status"] == "ok":
            valid_str = " ".join(
                f"h{h}={v['n_valid']}/{v['n_total']}"
                for h, v in rep["label_validity"].items()
            )
            print(f"   {d} rows={rep['rows']} files={rep['files_written']} "
                  f"valid: {valid_str}")
        elif rep["status"] == "skipped_exists":
            print(f"   {d} labels already exist (use --overwrite to rebuild)")
        else:
            print(f"   {d} {rep['status']}")


if __name__ == "__main__":
    main()
