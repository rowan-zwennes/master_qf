"""Stage 01b: structural checks on the merged raw dataset."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path("/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt")
REPORT_CSV = Path("daily_validation_report.csv")

FIRST_DATE = "2026-01-19"
LAST_DATE = "2026-04-23"
LAST_DATE_FINAL_HOUR = 4

STREAMS = ("trades", "diffs", "book", "markprice")

# columns actually read from each stream (keep the scan cheap)
COLUMNS = {
    "trades": ["id", "Price", "Quantity", "TransactTime"],
    "diffs": ["FirstUpdateID", "FinalUpdateID", "PrevFinalUpdateID", "EventTime"],
    "book": ["EventTime", "FinalUpdateID"],
    "markprice": ["EventTime", "MarkPrice", "FundingRate", "NextFundingTime"],
}
# the column the merger partitioned each stream's hour-files on
TIME_COL = {
    "trades": "TransactTime",
    "diffs": "EventTime",
    "book": "EventTime",
    "markprice": "EventTime",
}

HOUR_MS = 3_600_000

SANE_PAYLOAD_MAX = 1_000_000

# book is the @depth20@100ms stream
BOOK_GRID_MS = 100
BOOK_DROP_LO_MS = 180   # a delta in [180, 1000] ms = one or two missed snapshots
BOOK_DROP_HI_MS = 1000  # a delta above this = an outage
# markprice is the @markPrice@1s stream
MARK_DELAY_MS = 1500    # a delta above this = a delayed update

STRUCTURAL_CATS = {
    "missing_file", "empty_file", "off_hour_rows", "dup_ids",
    "cross_day_overlap", "null_key", "bad_rows", "read_error",
}
INFO_CATS = {"day_start_restart", "midday_restart", "cross_day_gap"}


def _dates() -> list[str]:
    out, d = [], datetime.strptime(FIRST_DATE, "%Y-%m-%d")
    end = datetime.strptime(LAST_DATE, "%Y-%m-%d")
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _day_start_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _fmt_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _validate_trades(df: pl.DataFrame, carry: dict, date: str
                     ) -> tuple[dict, list]:
    anomalies: list = []
    null_keys = df["id"].null_count()
    if null_keys:
        anomalies.append(("null_key", f"{date} trades: {null_keys} null id"))
        df = df.drop_nulls(subset=["id"])

    df = df.sort("id")
    n = df.height
    dup = n - df["id"].n_unique()
    if dup:
        anomalies.append(("dup_ids", f"{date} trades: {dup} duplicate id(s)"))

    ids = df["id"]
    gaps = ids.diff().drop_nulls()
    missing = int((gaps.filter(gaps > 1) - 1).sum() or 0)

    bad = df.filter(
        pl.col("Price").is_null() | ~pl.col("Price").is_finite()
        | (pl.col("Price") <= 0)
        | pl.col("Quantity").is_null() | ~pl.col("Quantity").is_finite()
        | (pl.col("Quantity") <= 0)
    ).height
    if bad:
        anomalies.append(("bad_rows", f"{date} trades: {bad} row(s) with bad price/qty"))

    first_id, last_id = int(ids[0]), int(ids[-1])
    quality = round(n / (n + missing) * 100, 4) if (n + missing) else 0.0

    boundary = None
    prev = carry.get("last_trade_id")
    if prev is not None:
        boundary = first_id - prev - 1  # 0 = perfectly consecutive across midnight
        if boundary < 0:
            anomalies.append(("cross_day_overlap",
                              f"{date} trades: {-boundary} id(s) overlap previous day"))
        elif boundary > 0:
            anomalies.append(("cross_day_gap",
                              f"{date} trades: {boundary} id(s) missing at day boundary"))

    stats = {
        "Trades_Total": n,
        "Trades_DupIDs": dup,
        "Trades_MissingIDs": missing,
        "Trades_BadRows": bad,
        "Trades_Quality_pct": quality,
        "Trades_BoundaryGap": boundary,
        "Last_Trade_ID": last_id,
    }
    return stats, anomalies


def _validate_diffs(df: pl.DataFrame, carry: dict, date: str
                    ) -> tuple[dict, list]:
    anomalies: list = []
    null_keys = df["FinalUpdateID"].null_count()
    if null_keys:
        anomalies.append(("null_key", f"{date} diffs: {null_keys} null FinalUpdateID"))
        df = df.drop_nulls(subset=["FinalUpdateID"])
    # a null pu is treated as the recorder's restart sentinel
    df = df.with_columns(pl.col("PrevFinalUpdateID").fill_null(-1))

    df = df.sort("FinalUpdateID")
    n = df.height
    dup = n - df["FinalUpdateID"].n_unique()
    if dup:
        anomalies.append(("dup_ids", f"{date} diffs: {dup} duplicate FinalUpdateID(s)"))

    u = df["FinalUpdateID"].to_numpy().astype(np.int64)
    pu = df["PrevFinalUpdateID"].to_numpy().astype(np.int64)
    first_id = df["FirstUpdateID"].to_numpy().astype(np.int64)
    et = df["EventTime"].to_numpy().astype(np.int64)

    sentinel = pu < 0
    pu = pu.copy()
    pu[sentinel] = first_id[sentinel] - 1
    per = u - pu  # update ids delivered by each payload

    is_chained = np.ones(n, dtype=bool)
    step = np.zeros(n, dtype=np.int64)
    if n > 1:
        is_chained[1:] = pu[1:] == u[:-1]
        step[1:] = pu[1:] - u[:-1]

    forward_gap = (~is_chained) & (step > 0) & (per < SANE_PAYLOAD_MAX)
    keep = is_chained | forward_gap

    range_base = int(pu[0])

    pu0_restart = bool(sentinel[0])
    if pu0_restart:
        anomalies.append(("day_start_restart",
                          f"{date} diffs: first payload pu<0 (recorder restart at day start)"))
    midday_sentinel_idx = np.where(sentinel & (np.arange(n) > 0))[0]
    if midday_sentinel_idx.size:
        anomalies.append(("midday_restart",
                          f"{date} diffs: restart sentinel (pu<0) at row(s) {midday_sentinel_idx.tolist()}"))

    received = int(per[keep].sum())
    ids_range = int(u[-1]) - range_base
    missing = int(step[(step > 0) & ~sentinel].sum())
    forward_gaps = int(((step > 0) & ~sentinel).sum())
    restarts = int((sentinel | (step < 0)).sum())
    restart_loss = max(0, ids_range - received - missing)
    quality = round(received / ids_range * 100, 4) if ids_range > 0 else 0.0

    gap_rows = np.zeros(n, dtype=bool)
    if n > 1:
        gap_rows[1:] = forward_gap[1:] | (sentinel[1:])
    et_deltas = np.zeros(n, dtype=np.int64)
    if n > 1:
        et_deltas[1:] = et[1:] - et[:-1]
    total_gap_ms = int(et_deltas[gap_rows].sum())

    boundary = None
    prev = carry.get("last_diff_u")
    if prev is not None and not pu0_restart:
        boundary = int(pu[0]) - prev  # 0 = perfectly chained across midnight
        if boundary < 0:
            anomalies.append(("cross_day_overlap",
                              f"{date} diffs: pu of first payload is {-boundary} before previous day's last u"))
        elif boundary > 0:
            anomalies.append(("cross_day_gap",
                              f"{date} diffs: {boundary} update id(s) missing at day boundary"))

    stats = {
        "Diffs_Payloads": n,
        "Diffs_DupIDs": dup,
        "Diffs_IDs_Range": ids_range,
        "Diffs_IDs_Received": received,
        "Diffs_IDs_Missing": missing,
        "Diffs_ForwardGaps": forward_gaps,
        "Diffs_Restarts": restarts,
        "Diffs_RestartLoss": restart_loss,
        "Diffs_TotalGapTime_ms": total_gap_ms,
        "Diffs_Quality_pct": quality,
        "Diffs_BoundaryGap": boundary,
        "Last_Diff_U": int(u[-1]),
    }
    return stats, anomalies


def _validate_book(df: pl.DataFrame, carry: dict, date: str
                   ) -> tuple[dict, list]:
    anomalies: list = []
    null_keys = df["EventTime"].null_count()
    if null_keys:
        anomalies.append(("null_key", f"{date} book: {null_keys} null EventTime"))
        df = df.drop_nulls(subset=["EventTime"])

    df = df.sort("EventTime")
    n = df.height
    dup = n - df.select(["EventTime", "FinalUpdateID"]).unique().height
    if dup:
        anomalies.append(("dup_ids", f"{date} book: {dup} duplicate (EventTime, FinalUpdateID) row(s)"))

    et = df["EventTime"]
    deltas = et.diff().drop_nulls()
    drops = int(((deltas >= BOOK_DROP_LO_MS) & (deltas <= BOOK_DROP_HI_MS)).sum())
    outages = int((deltas > BOOK_DROP_HI_MS).sum())

    first_et, last_et = int(et[0]), int(et[-1])
    duration = last_et - first_et
    capture = round(n / (duration / BOOK_GRID_MS) * 100, 4) if duration > 0 else 0.0

    boundary = None
    prev = carry.get("last_book_et")
    if prev is not None:
        boundary = first_et - prev  # expected ~100 ms
        if boundary > BOOK_DROP_HI_MS:
            anomalies.append(("cross_day_gap",
                              f"{date} book: {boundary} ms gap across day boundary"))

    stats = {
        "Book_Rows": n,
        "Book_DupRows": dup,
        "Book_Drops": drops,
        "Book_Outages": outages,
        "Book_Capture_pct": capture,
        "Book_BoundaryGap_ms": boundary,
        "Last_Book_ET": last_et,
    }
    return stats, anomalies


def _validate_markprice(df: pl.DataFrame, carry: dict, date: str
                        ) -> tuple[dict, list]:
    anomalies: list = []
    null_keys = df["EventTime"].null_count()
    if null_keys:
        anomalies.append(("null_key", f"{date} markprice: {null_keys} null EventTime"))
        df = df.drop_nulls(subset=["EventTime"])

    df = df.sort("EventTime")
    n = df.height
    dup = n - df.select(["EventTime", "MarkPrice"]).unique().height
    if dup:
        anomalies.append(("dup_ids", f"{date} markprice: {dup} duplicate (EventTime, MarkPrice) row(s)"))

    et = df["EventTime"]
    deltas = et.diff().drop_nulls()
    delayed = int((deltas > MARK_DELAY_MS).sum())

    errors = df.filter(
        pl.col("FundingRate").is_null() | ~pl.col("FundingRate").is_finite()
        | pl.col("NextFundingTime").is_null()
        | (pl.col("NextFundingTime") <= pl.col("EventTime"))
    ).height

    first_et, last_et = int(et[0]), int(et[-1])
    boundary = None
    prev = carry.get("last_mark_et")
    if prev is not None:
        boundary = first_et - prev  # expected ~1000 ms
        if boundary > MARK_DELAY_MS:
            anomalies.append(("cross_day_gap",
                              f"{date} markprice: {boundary} ms gap across day boundary"))

    stats = {
        "Mark_Rows": n,
        "Mark_DupRows": dup,
        "Mark_Delayed": delayed,
        "Mark_Errors": errors,
        "Mark_BoundaryGap_ms": boundary,
        "Last_Mark_ET": last_et,
    }
    return stats, anomalies


VALIDATORS = {
    "trades": _validate_trades,
    "diffs": _validate_diffs,
    "book": _validate_book,
    "markprice": _validate_markprice,
}
EMPTY_STATS = {
    "trades": {"Trades_Total": 0, "Trades_DupIDs": 0, "Trades_MissingIDs": 0,
               "Trades_BadRows": 0, "Trades_Quality_pct": None,
               "Trades_BoundaryGap": None, "Last_Trade_ID": None},
    "diffs": {"Diffs_Payloads": 0, "Diffs_DupIDs": 0, "Diffs_IDs_Range": 0,
              "Diffs_IDs_Received": 0, "Diffs_IDs_Missing": 0,
              "Diffs_ForwardGaps": 0, "Diffs_Restarts": 0,
              "Diffs_RestartLoss": 0, "Diffs_TotalGapTime_ms": 0,
              "Diffs_Quality_pct": None,
              "Diffs_BoundaryGap": None, "Last_Diff_U": None},
    "book": {"Book_Rows": 0, "Book_DupRows": 0, "Book_Drops": 0,
             "Book_Outages": 0, "Book_Capture_pct": None,
             "Book_BoundaryGap_ms": None, "Last_Book_ET": None},
    "markprice": {"Mark_Rows": 0, "Mark_DupRows": 0, "Mark_Delayed": 0,
                  "Mark_Errors": 0, "Mark_BoundaryGap_ms": None,
                  "Last_Mark_ET": None},
}
CARRY_KEY = {
    "trades": ("last_trade_id", "Last_Trade_ID"),
    "diffs": ("last_diff_u", "Last_Diff_U"),
    "book": ("last_book_et", "Last_Book_ET"),
    "markprice": ("last_mark_et", "Last_Mark_ET"),
}


def _audit_day(date_str: str, carry: dict) -> tuple[dict, list, dict]:
    hours = range(LAST_DATE_FINAL_HOUR + 1) if date_str == LAST_DATE else range(24)
    hours = list(hours)
    day_start = _day_start_ms(date_str)

    row: dict = {"Date": date_str, "Hours_Expected": len(hours)}
    anomalies: list = []
    missing_files = empty_files = offhour_rows = 0
    frames: dict[str, pl.DataFrame | None] = {}

    for stream in STREAMS:
        tc = TIME_COL[stream]
        parts: list[pl.DataFrame] = []
        for h in hours:
            fp = BASE / date_str / f"{stream}_{h:02d}h.parquet"
            if not fp.exists():
                missing_files += 1
                anomalies.append(("missing_file", f"{date_str} {stream}_{h:02d}h"))
                continue
            try:
                df = pl.read_parquet(fp, columns=COLUMNS[stream])
            except Exception as e:  
                anomalies.append(("read_error", f"{date_str} {stream}_{h:02d}h: {e}"))
                continue
            if df.height == 0:
                empty_files += 1
                anomalies.append(("empty_file", f"{date_str} {stream}_{h:02d}h"))
                continue

            hs = day_start + h * HOUR_MS
            he = hs + HOUR_MS
            stray = df.filter((pl.col(tc) < hs) | (pl.col(tc) >= he))
            if stray.height:
                offhour_rows += stray.height
                example = _fmt_ms(int(stray[tc][0]))
                anomalies.append((
                    "off_hour_rows",
                    f"{date_str} {stream}_{h:02d}h: {stray.height} row(s) "
                    f"outside the hour (e.g. {example})",
                ))
            parts.append(df)

        frames[stream] = pl.concat(parts) if parts else None

    new_carry = dict(carry)
    for stream in STREAMS:
        df = frames[stream]
        if df is None:
            row.update(EMPTY_STATS[stream])
            continue
        stats, anos = VALIDATORS[stream](df, carry, date_str)
        row.update(stats)
        anomalies.extend(anos)
        carry_field, stats_field = CARRY_KEY[stream]
        new_carry[carry_field] = stats[stats_field]

    row["OffHour_Rows"] = offhour_rows
    row["MissingFiles"] = missing_files
    row["EmptyFiles"] = empty_files
    return row, anomalies, new_carry


def main() -> None:
    dates = _dates()
    print(f"merged-data validation -- {dates[0]} 00h .. {dates[-1]} "
          f"{LAST_DATE_FINAL_HOUR:02d}h  ({len(dates)} days)")
    print(f"   source: {BASE}\n")

    if not BASE.exists():
        sys.exit(f"source directory not found: {BASE}")

    rows: list[dict] = []
    all_anomalies: list = []
    carry: dict = {}

    for date_str in dates:
        row, anomalies, carry = _audit_day(date_str, carry)
        rows.append(row)
        all_anomalies.extend(anomalies)

        structural = sum(1 for c, _ in anomalies if c in STRUCTURAL_CATS)
        flag = f"   {structural} structural" if structural else ""
        hours_txt = "23h" if row["Hours_Expected"] == 24 else f"{row['Hours_Expected'] - 1:02d}h"
        print(
            f"  {date_str}  00-{hours_txt}  "
            f"T {row['Trades_Total']:>9,}  "
            f"D {row['Diffs_Payloads']:>8,}  "
            f"B {row['Book_Rows']:>8,}  "
            f"M {row['Mark_Rows']:>7,}  "
            f"diff_q {_pct(row['Diffs_Quality_pct'])}  "
            f"book_cap {_pct(row['Book_Capture_pct'])}{flag}"
        )

    rdf = pl.DataFrame(rows)
    rdf.write_csv(REPORT_CSV)

    print(f"\n{'=' * 78}")
    print(f"GLOBAL -- {len(rows)} days")
    print(f"  trades    : {_isum(rdf, 'Trades_Total'):>14,} rows   "
          f"missing ids {_isum(rdf, 'Trades_MissingIDs'):>10,}   "
          f"bad rows {_isum(rdf, 'Trades_BadRows'):>6,}")
    total_gap_s = _isum(rdf, 'Diffs_TotalGapTime_ms') / 1000
    print(f"  diffs     : {_isum(rdf, 'Diffs_Payloads'):>14,} payloads   "
          f"gap ids {_isum(rdf, 'Diffs_IDs_Missing'):>10,}   "
          f"restart-loss ids {_isum(rdf, 'Diffs_RestartLoss'):>10,}   "
          f"restarts {_isum(rdf, 'Diffs_Restarts'):>3,}   "
          f"total gap time {total_gap_s:,.1f} s")
    print(f"  book      : {_isum(rdf, 'Book_Rows'):>14,} rows   "
          f"drops {_isum(rdf, 'Book_Drops'):>8,}   "
          f"outages {_isum(rdf, 'Book_Outages'):>6,}")
    print(f"  markprice : {_isum(rdf, 'Mark_Rows'):>14,} rows   "
          f"delayed {_isum(rdf, 'Mark_Delayed'):>8,}   "
          f"errors {_isum(rdf, 'Mark_Errors'):>7,}")

    by_cat: dict[str, list[str]] = {}
    for cat, detail in all_anomalies:
        by_cat.setdefault(cat, []).append(detail)

    print(f"\n{'-' * 78}")
    print("STRUCTURAL CHECK  (every count below must be 0 for a clean merge)")
    structural_total = 0
    for cat in sorted(STRUCTURAL_CATS):
        hits = by_cat.get(cat, [])
        structural_total += len(hits)
        print(f"  {cat:<20} {len(hits)}")
        for detail in hits[:25]:
            print(f"       {detail}")
        if len(hits) > 25:
            print(f"       ... and {len(hits) - 25} more")

    info = sorted(c for c in INFO_CATS if by_cat.get(c))
    if info:
        print(f"\n{'-' * 78}")
        print("QUALITY NOTES  (recording artefacts, not merge defects)")
        for cat in info:
            hits = by_cat[cat]
            print(f"  {cat:<20} {len(hits)}")
            for detail in hits[:25]:
                print(f"       {detail}")
            if len(hits) > 25:
                print(f"       ... and {len(hits) - 25} more")

    print(f"\n{'=' * 78}")
    print(f"per-day report written to {REPORT_CSV}")
    if structural_total == 0:
        print("no structural anomalies -- the merge is clean over the validated range")
    else:
        print(f"{structural_total} structural anomaly/anomalies -- see breakdown above")
        sys.exit(1)


def _pct(value: float | None) -> str:
    return "  --   " if value is None else f"{value:7.2f}%"


def _isum(rdf: pl.DataFrame, col: str) -> int:
    return int(rdf[col].sum() or 0)


if __name__ == "__main__":
    main()
