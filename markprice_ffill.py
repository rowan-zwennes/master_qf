"""Stage 01: mark-price forward-fill onto a one-second grid."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
STATE_FILENAME = "_markprice_state.json"

# A mark price tick older than this is "very stale".
STALE_FLAG_THRESHOLD_MS = 5000  # matches basis_bps null-gate in feature_engineering


def _hour_bounds_ms(date_str: str, hour: int) -> tuple[int, int]:
    start = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=hour, tzinfo=timezone.utc
    )
    end = start + timedelta(hours=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _load_state(base: Path, date_str: str) -> dict | None:
    p = base / date_str / STATE_FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save_state(base: Path, date_str: str, state: dict) -> None:
    p = base / date_str / STATE_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def _seed_from_previous_hour(
    base: Path, date_str: str, hour: int
) -> dict | None:
    """
    Returns the last observed mark tick before `hour` on `date_str`,
    looking back into the previous hour file or previous day if needed.
    """
    if hour > 0:
        prev_path = base / date_str / f"markprice_{hour - 1:02d}h.parquet"
        if prev_path.exists():
            try:
                df = pl.read_parquet(prev_path).sort("EventTime").tail(1)
                if not df.is_empty():
                    r = df.row(0, named=True)
                    return {
                        "EventTime": int(r["EventTime"]),
                        "MarkPrice": float(r["MarkPrice"]),
                        "IndexPrice": float(r["IndexPrice"]),
                        "EstSettlePrice": float(r["EstSettlePrice"]),
                        "FundingRate": float(r["FundingRate"]),
                        "NextFundingTime": int(r["NextFundingTime"]),
                    }
            except Exception:
                pass
        return None

    # hour == 0: look at previous day, hour 23
    prev_day = (
        datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")
    prev_path = base / prev_day / "markprice_23h.parquet"
    if prev_path.exists():
        try:
            df = pl.read_parquet(prev_path).sort("EventTime").tail(1)
            if not df.is_empty():
                r = df.row(0, named=True)
                return {
                    "EventTime": int(r["EventTime"]),
                    "MarkPrice": float(r["MarkPrice"]),
                    "IndexPrice": float(r["IndexPrice"]),
                    "EstSettlePrice": float(r["EstSettlePrice"]),
                    "FundingRate": float(r["FundingRate"]),
                    "NextFundingTime": int(r["NextFundingTime"]),
                }
        except Exception:
            pass
    return None


def ffill_one_hour(
    base: Path,
    date_str: str,
    hour: int,
    *,
    overwrite: bool = False,
    seed_interval_s: int | None = None,
) -> dict:
    """
    Returns a small report dict.

    `seed_interval_s` is the last known funding interval, threaded in by
    process_date. An hour with no settlement transition of its own cannot
    derive the interval from its NextFundingTime stream, so it inherits this.
    """
    src = base / date_str / f"markprice_{hour:02d}h.parquet"
    dst = base / date_str / f"markprice_ffill_{hour:02d}h.parquet"

    if not src.exists():
        return {"hour": hour, "status": "missing_input"}

    if dst.exists() and not overwrite:
        return {"hour": hour, "status": "already_done"}

    start_ms, end_ms = _hour_bounds_ms(date_str, hour)

    df = (
        pl.read_parquet(src)
        .filter(
            (pl.col("EventTime") >= start_ms) & (pl.col("EventTime") < end_ms)
        )
        .sort("EventTime")
    )

    seed = _seed_from_previous_hour(base, date_str, hour)

    # Build the 1s grid
    grid = pl.DataFrame(
        {"ts_ms": list(range(start_ms, end_ms, 1000))}
    ).with_columns(pl.col("ts_ms").cast(pl.Int64))

    if df.is_empty() and seed is None:
        out = grid.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("MarkPrice"),
                pl.lit(None, dtype=pl.Float64).alias("IndexPrice"),
                pl.lit(None, dtype=pl.Float64).alias("EstSettlePrice"),
                pl.lit(None, dtype=pl.Float64).alias("FundingRate"),
                pl.lit(None, dtype=pl.Int64).alias("NextFundingTime"),
                pl.lit(seed_interval_s, dtype=pl.Int64).alias("funding_interval_s"),
                pl.lit(None, dtype=pl.Int64).alias("seconds_to_funding"),
                pl.lit(False, dtype=pl.Boolean).alias("funding_transition"),
                pl.lit(True, dtype=pl.Boolean).alias("is_stale"),
                pl.lit(None, dtype=pl.Int64).alias("stale_age_ms"),
            ]
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(dst, compression="zstd")
        return {
            "hour": hour,
            "status": "empty",
            "rows_in": 0,
            "rows_out": out.height,
            "funding_interval_s": seed_interval_s,
        }

    rows = []
    if seed is not None:
        rows.append(
            {
                "tick_event_time": seed["EventTime"],
                "MarkPrice": seed["MarkPrice"],
                "IndexPrice": seed["IndexPrice"],
                "EstSettlePrice": seed["EstSettlePrice"],
                "FundingRate": seed["FundingRate"],
                "NextFundingTime": seed["NextFundingTime"],
            }
        )

    if not df.is_empty():
        ticks_df = df.select(
            pl.col("EventTime").alias("tick_event_time"),
            pl.col("MarkPrice").cast(pl.Float64),
            pl.col("IndexPrice").cast(pl.Float64),
            pl.col("EstSettlePrice").cast(pl.Float64),
            pl.col("FundingRate").cast(pl.Float64),
            pl.col("NextFundingTime").cast(pl.Int64),
        )
        if rows:
            ticks_df = pl.concat(
                [pl.DataFrame(rows, schema=ticks_df.schema), ticks_df]
            )
    else:
        ticks_df = pl.DataFrame(rows)

    ticks_df = ticks_df.sort("tick_event_time")

    out = grid.join_asof(
        ticks_df,
        left_on="ts_ms",
        right_on="tick_event_time",
        strategy="backward",
    )

    # Compute staleness: age of last real tick at this grid second.
    out = out.with_columns(
        (pl.col("ts_ms") - pl.col("tick_event_time")).alias("stale_age_ms")
    ).with_columns(
        (pl.col("stale_age_ms") > STALE_FLAG_THRESHOLD_MS).alias("is_stale_raw"),
    )

    nft_unique = out.select("NextFundingTime").unique().drop_nulls()
    if nft_unique.height >= 2:
        sorted_nft = sorted(nft_unique.to_series().to_list())
        diffs_ms = [b - a for a, b in zip(sorted_nft, sorted_nft[1:]) if b > a]
        interval_s = (
            int(round(min(diffs_ms) / 1000)) if diffs_ms else None
        )
    else:
        interval_s = None

    if interval_s is None:
        interval_s = seed_interval_s

    out = out.with_columns(
        [
            pl.lit(interval_s, dtype=pl.Int64).alias("funding_interval_s"),
            (
                (pl.col("NextFundingTime") - pl.col("ts_ms")) // 1000
            ).alias("seconds_to_funding"),
            pl.col("is_stale_raw").alias("is_stale"),
        ]
    ).drop(["tick_event_time", "is_stale_raw"]).with_columns(
        [
            pl.when(pl.col("seconds_to_funding") < 0)
            .then(None)
            .otherwise(pl.col("seconds_to_funding"))
            .alias("seconds_to_funding"),
            pl.when(pl.col("seconds_to_funding") < 0)
            .then(None)
            .otherwise(pl.col("FundingRate"))
            .alias("FundingRate"),
            (pl.col("seconds_to_funding") < 0).alias("funding_transition"),
        ]
    )

    # Reorder columns
    out = out.select(
        [
            "ts_ms",
            "MarkPrice",
            "IndexPrice",
            "EstSettlePrice",
            "FundingRate",
            "NextFundingTime",
            "funding_interval_s",
            "seconds_to_funding",
            "funding_transition",
            "is_stale",
            "stale_age_ms",
        ]
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(dst, compression="zstd")

    n_stale = int(out["is_stale"].sum() or 0)
    n_null = int(out["MarkPrice"].is_null().sum() or 0)
    return {
        "hour": hour,
        "status": "ok",
        "rows_in": df.height,
        "rows_out": out.height,
        "n_stale": n_stale,
        "n_null_mark": n_null,
        "funding_interval_s": interval_s,
    }


def process_date(base: Path, date_str: str, *, overwrite: bool = False) -> None:
    print(f"\nmarkPrice ffill | {date_str}")

    carry_interval: int | None = None
    prev_day = (
        datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    ).strftime("%Y-%m-%d")
    ps = _load_state(base, prev_day)
    if ps and "hours" in ps:
        for h in reversed(ps["hours"]):
            if h.get("funding_interval_s"):
                carry_interval = h["funding_interval_s"]
                break

    reports = []
    for hour in range(24):
        rep = ffill_one_hour(
            base, date_str, hour, overwrite=overwrite,
            seed_interval_s=carry_interval,
        )
        reports.append(rep)
        if rep.get("funding_interval_s"):
            carry_interval = rep["funding_interval_s"]
        if rep["status"] == "ok":
            print(
                f"   {hour:02d}h in={rep['rows_in']:>5} out={rep['rows_out']} "
                f"stale={rep['n_stale']:>4} null_mark={rep['n_null_mark']:>4} "
                f"interval={rep['funding_interval_s']}s"
            )
        elif rep["status"] == "missing_input":
            print(f"   {hour:02d}h (no input)")
        elif rep["status"] == "empty":
            print(f"   {hour:02d}h  empty hour, skeleton written")
        elif rep["status"] == "already_done":
            print(f"   {hour:02d}h already done")

    # Save a summary state file
    _save_state(
        base,
        date_str,
        {
            "stage": "markprice_ffill",
            "hours": reports,
            "written_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--date", help="YYYY-MM-DD; if omitted, processes all dates")
    p.add_argument("--from-date", help="YYYY-MM-DD inclusive")
    p.add_argument("--to-date", help="YYYY-MM-DD inclusive")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    base = Path(args.base)
    if not base.exists():
        sys.exit(f"base not found: {base}")

    if args.date:
        dates = [args.date]
    else:
        all_dates = sorted(
            d.name
            for d in base.iterdir()
            if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"
        )
        if args.from_date:
            all_dates = [d for d in all_dates if d >= args.from_date]
        if args.to_date:
            all_dates = [d for d in all_dates if d <= args.to_date]
        dates = all_dates

    print(f"markPrice ffill | {len(dates)} dates")
    for d in dates:
        process_date(base, d, overwrite=args.overwrite)

    print("\nmarkPrice ffill complete")


if __name__ == "__main__":
    main()
