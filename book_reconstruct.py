"""Stage 02: order-book reconstruction onto a 100 ms grid."""
from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import orjson
import polars as pl

DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
TOP_N = 20
GRID_MS = 100  # 100ms grid -> 36000 ticks/hour
TICK_SIZE = 0.10  # BTCUSDT perpetual futures minimum price increment (USDT)

STATE_FILE = "_book_state.json"


def _build_output_schema() -> dict:
    """Explicit schema for the book20 output. Without it, an hour that emits
    only invalid rows infers Null-typed columns and breaks cross-hour
    concatenation downstream."""
    schema: dict = {
        "ts_ms": pl.Int64,
        "last_event_time": pl.Int64,
        "last_local_time_us": pl.Int64,
        "last_update_id": pl.Int64,
    }
    for i in range(TOP_N):
        schema[f"bid_p_{i}"] = pl.Float64
    for i in range(TOP_N):
        schema[f"bid_q_{i}"] = pl.Float64
    for i in range(TOP_N):
        schema[f"ask_p_{i}"] = pl.Float64
    for i in range(TOP_N):
        schema[f"ask_q_{i}"] = pl.Float64
    schema["valid"] = pl.Boolean
    schema["anchor_kind"] = pl.Utf8
    schema["ticks_since_anchor"] = pl.Int32
    return schema


OUTPUT_SCHEMA = _build_output_schema()

INVALID_SCHEMA: dict = {
    "ts_ms_start": pl.Int64,
    "ts_ms_end": pl.Int64,
    "reason": pl.Utf8,
    "last_u_before": pl.Int64,
    "next_anchor_u": pl.Int64,
}


def _hour_bounds_ms(date_str: str, hour: int) -> tuple[int, int]:
    start = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=hour, tzinfo=timezone.utc
    )
    end = start + timedelta(hours=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _decode_levels(s: str) -> list[tuple[int, float]]:
    """Parse [[price,qty],...] and return integer tick indices.
    """
    if s is None or s == "":
        return []
    try:
        arr = orjson.loads(s)
    except Exception:
        return []
    out = []
    for lv in arr:
        try:
            p = int(round(float(lv[0]) / TICK_SIZE))
            q = float(lv[1])
            out.append((p, q))
        except Exception:
            continue
    return out


def _tick_key(k: str) -> int:
    """Convert a JSON carry-state key to an int tick index.
    """
    return int(k) if "." not in k else int(round(float(k) / TICK_SIZE))


def _apply_levels(
    side: dict, prices: list[int], levels: list[tuple[int, float]]
) -> None:
    """Mutate side: qty==0 deletes the level; otherwise overwrite.
    """
    for p, q in levels:
        if q == 0.0:
            if p in side:
                del side[p]
                del prices[bisect.bisect_left(prices, p)]
        else:
            if p not in side:
                bisect.insort(prices, p)
            side[p] = q


def _top_n(
    side: dict, prices: list[int], descending: bool, n: int
) -> list[tuple[float, float]]:
    """Return top-n (price, qty) sorted by price (desc for bids, asc for asks).
    """
    if descending:
        sel = prices[-n:]
        sel.reverse()
    else:
        sel = prices[:n]
    return [(p * TICK_SIZE, side[p]) for p in sel]


def _emit_row(
    ts_ms: int,
    bids: dict,
    asks: dict,
    bid_prices: list[int],
    ask_prices: list[int],
    last_event_time: int | None,
    last_local_time_us: int | None,
    last_update_id: int | None,
    valid: bool,
    anchor_kind: str,
    ticks_since_anchor: int | None,
) -> dict:
    row = {
        "ts_ms": ts_ms,
        "last_event_time": last_event_time,
        "last_local_time_us": last_local_time_us,
        "last_update_id": last_update_id,
        "valid": valid,
        "anchor_kind": anchor_kind,
        "ticks_since_anchor": ticks_since_anchor,
    }
    if valid:
        top_b = _top_n(bids, bid_prices, descending=True, n=TOP_N)
        top_a = _top_n(asks, ask_prices, descending=False, n=TOP_N)
        for i in range(TOP_N):
            if i < len(top_b):
                row[f"bid_p_{i}"] = top_b[i][0]
                row[f"bid_q_{i}"] = top_b[i][1]
            else:
                row[f"bid_p_{i}"] = None
                row[f"bid_q_{i}"] = None
            if i < len(top_a):
                row[f"ask_p_{i}"] = top_a[i][0]
                row[f"ask_q_{i}"] = top_a[i][1]
            else:
                row[f"ask_p_{i}"] = None
                row[f"ask_q_{i}"] = None
    else:
        for i in range(TOP_N):
            row[f"bid_p_{i}"] = None
            row[f"bid_q_{i}"] = None
            row[f"ask_p_{i}"] = None
            row[f"ask_q_{i}"] = None
    return row


def _load_state(base: Path, date_str: str) -> dict | None:
    p = base / date_str / STATE_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save_state(base: Path, date_str: str, state: dict) -> None:
    p = base / date_str / STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def reconstruct_hour(
    base: Path,
    date_str: str,
    hour: int,
    *,
    overwrite: bool = False,
    carry_state: dict | None = None,
) -> dict:
    """carry_state, if provided, must be: { 'bids': {price: qty, ...}, 'asks': {price: qty."""
    out_book = base / date_str / f"book20_{hour:02d}h.parquet"
    out_inv = base / date_str / f"book20_invalid_{hour:02d}h.parquet"

    carry_only = out_book.exists() and not overwrite

    book_src = base / date_str / f"book_{hour:02d}h.parquet"
    diff_src = base / date_str / f"diffs_{hour:02d}h.parquet"

    if not book_src.exists() and not diff_src.exists():
        return {"hour": hour, "status": "missing_input", "carry_state": carry_state}

    start_ms, end_ms = _hour_bounds_ms(date_str, hour)

    if book_src.exists():
        snaps = (
            pl.read_parquet(book_src)
            .filter((pl.col("EventTime") >= start_ms) & (pl.col("EventTime") < end_ms))
            .sort(["EventTime", "FinalUpdateID"])
        )
    else:
        snaps = pl.DataFrame()

    if diff_src.exists():
        diffs = (
            pl.read_parquet(diff_src)
            .filter((pl.col("EventTime") >= start_ms) & (pl.col("EventTime") < end_ms))
            .sort(["FinalUpdateID"])
        )
    else:
        diffs = pl.DataFrame()

    events: list[tuple[int, int, str, dict]] = []
    if not snaps.is_empty():
        for r in snaps.iter_rows(named=True):
            events.append((int(r["FinalUpdateID"]), 1, "S", r))
    if not diffs.is_empty():
        for r in diffs.iter_rows(named=True):
            events.append((int(r["FinalUpdateID"]), 0, "D", r))
    events.sort(key=lambda x: (x[0], x[1]))

    if carry_state is not None:
        bids = dict(carry_state["bids"])
        asks = dict(carry_state["asks"])
        last_update_id: int | None = carry_state.get("last_update_id")
        valid = bool(carry_state.get("valid", False))
        anchor_kind = carry_state.get("anchor_kind", "invalid")
        ticks_since_anchor = carry_state.get("ticks_since_anchor")
        last_event_time = carry_state.get("last_event_time")
        last_local_time_us = carry_state.get("last_local_time_us")
    else:
        bids: dict = {}
        asks: dict = {}
        last_update_id = None
        valid = False
        anchor_kind = "invalid"
        ticks_since_anchor = None
        last_event_time = None
        last_local_time_us = None

    bid_prices: list[int] = sorted(bids)
    ask_prices: list[int] = sorted(asks)

    invalid_intervals: list[dict] = []
    rows_out: list[dict] = []

    n_diff_applied = 0
    n_diff_dropped_stale = 0
    n_desyncs = 0
    n_anchor_snaps = 0
    n_snaps_ignored = 0

    cur_invalid_start: int | None = None
    cur_invalid_reason: str | None = None
    cur_invalid_last_u: int | None = None

    if not valid:
        cur_invalid_start = start_ms
        cur_invalid_reason = "no_anchor_yet"
        cur_invalid_last_u = last_update_id

    def _apply_event(_kind: str, r: dict) -> None:
        """Mutates the closed-over state."""
        nonlocal bids, asks, bid_prices, ask_prices, last_update_id, valid, anchor_kind
        nonlocal ticks_since_anchor, last_event_time, last_local_time_us
        nonlocal n_diff_applied, n_diff_dropped_stale, n_desyncs, n_anchor_snaps
        nonlocal n_snaps_ignored
        nonlocal cur_invalid_start, cur_invalid_reason, cur_invalid_last_u

        et = int(r["EventTime"])
        try:
            lt = int(r["LocalTime"]) if r.get("LocalTime") is not None else None
        except Exception:
            lt = None

        if _kind == "S":
            u = int(r["FinalUpdateID"])

            if valid:
                # Snapshots are ignored during valid operation.
                n_snaps_ignored += 1
                return

            # No-anchor or pending. Accept only if fresher than current state.
            if last_update_id is not None and u <= last_update_id:
                return  # stale snapshot

            new_bids_levels = _decode_levels(r["Bids"])
            new_asks_levels = _decode_levels(r["Asks"])
            bids = {p: q for p, q in new_bids_levels if q > 0}
            asks = {p: q for p, q in new_asks_levels if q > 0}
            bid_prices = sorted(bids)
            ask_prices = sorted(asks)
            last_update_id = u
            last_event_time = et
            last_local_time_us = lt
            anchor_kind = "invalid"
            ticks_since_anchor = None
            n_anchor_snaps += 1
            return

        u = int(r["FinalUpdateID"])
        fu = (
            int(r["FirstUpdateID"])
            if r.get("FirstUpdateID") is not None
            else None
        )
        pu = (
            int(r["PrevFinalUpdateID"])
            if r.get("PrevFinalUpdateID") is not None
            else None
        )

        if last_update_id is None:
            # No-anchor state: no snapshot has loaded yet.
            n_diff_dropped_stale += 1
            return

        if u <= last_update_id:
            # Fully stale: already covered by the pending snapshot or a prior diff.
            n_diff_dropped_stale += 1
            return

        if pu is None or pu <= 0:
            is_gap = fu is None or fu > last_update_id + 1
        else:
            is_gap = pu > last_update_id

        if is_gap:
            n_desyncs += 1

            if valid:
                gap_start = (
                    last_event_time + GRID_MS if last_event_time is not None else et
                )
                cur_invalid_start = gap_start
                cur_invalid_reason = "diff_pu_mismatch"
                cur_invalid_last_u = last_update_id

                for i in range(len(rows_out) - 1, -1, -1):
                    g_i = rows_out[i]["ts_ms"]
                    if g_i < gap_start:
                        break
                    rows_out[i]["valid"] = False
                    rows_out[i]["anchor_kind"] = "invalid"
                    for j in range(TOP_N):
                        rows_out[i][f"bid_p_{j}"] = None
                        rows_out[i][f"bid_q_{j}"] = None
                        rows_out[i][f"ask_p_{j}"] = None
                        rows_out[i][f"ask_q_{j}"] = None
                valid = False

            bids = {}
            bid_prices = []
            asks = {}
            ask_prices = []
            last_update_id = None
            anchor_kind = "invalid"
            ticks_since_anchor = None
            return

        new_bids_levels = _decode_levels(r["Bids"])
        new_asks_levels = _decode_levels(r["Asks"])
        _apply_levels(bids, bid_prices, new_bids_levels)
        _apply_levels(asks, ask_prices, new_asks_levels)
        last_update_id = u
        last_event_time = et
        last_local_time_us = lt
        anchor_kind = "diff"
        n_diff_applied += 1

        if not valid:
            valid = True
            ticks_since_anchor = 1
            if cur_invalid_start is not None:
                invalid_intervals.append({
                    "ts_ms_start": cur_invalid_start,
                    "ts_ms_end": et,
                    "reason": cur_invalid_reason or "no_anchor_yet",
                    "last_u_before": cur_invalid_last_u,
                    "next_anchor_u": u,
                })
                cur_invalid_start = None
                cur_invalid_reason = None
                cur_invalid_last_u = None
        else:
            if ticks_since_anchor is None:
                ticks_since_anchor = 0
            ticks_since_anchor += 1

    def _make_final() -> dict | None:
        """Outgoing carry state, after applying ALL of this hour's events."""
        if events or carry_state is not None:
            return {
                "bids": dict(bids),
                "asks": dict(asks),
                "last_update_id": last_update_id,
                "valid": valid,
                "anchor_kind": anchor_kind,
                "ticks_since_anchor": ticks_since_anchor,
                "last_event_time": last_event_time,
                "last_local_time_us": last_local_time_us,
            }
        return None

    if carry_only:
        for _u, _tb, kind, r in events:
            _apply_event(kind, r)
        return {
            "hour": hour,
            "status": "already_done",
            "carry_state": _make_final(),
        }

    ev_idx = 0
    n_events = len(events)

    g = start_ms
    while g < end_ms:
        while ev_idx < n_events and events[ev_idx][3]["EventTime"] <= g:
            _, _tb, kind, r = events[ev_idx]
            _apply_event(kind, r)
            ev_idx += 1

        rows_out.append(
            _emit_row(
                g,
                bids,
                asks,
                bid_prices,
                ask_prices,
                last_event_time,
                last_local_time_us,
                last_update_id,
                valid,
                anchor_kind,
                ticks_since_anchor,
            )
        )
        g += GRID_MS

    while ev_idx < n_events:
        _, _tb, kind, r = events[ev_idx]
        _apply_event(kind, r)
        ev_idx += 1

    if cur_invalid_start is not None:
        invalid_intervals.append({
            "ts_ms_start": cur_invalid_start,
            "ts_ms_end": end_ms,
            "reason": cur_invalid_reason or "no_anchor_yet",
            "last_u_before": cur_invalid_last_u,
            "next_anchor_u": None,
        })

    df_out = pl.DataFrame(rows_out, schema=OUTPUT_SCHEMA)
    out_book.parent.mkdir(parents=True, exist_ok=True)
    df_out.write_parquet(out_book, compression="zstd")

    if invalid_intervals:
        pl.DataFrame(invalid_intervals, schema=INVALID_SCHEMA).write_parquet(
            out_inv, compression="zstd"
        )
    elif out_inv.exists():
        out_inv.unlink()

    final = _make_final()

    valid_ticks = int(df_out["valid"].sum() or 0)
    return {
        "hour": hour,
        "status": "ok",
        "n_events": len(events),
        "n_diff_applied": n_diff_applied,
        "n_diff_dropped_stale": n_diff_dropped_stale,
        "n_desyncs": n_desyncs,
        "n_anchor_snaps": n_anchor_snaps,
        "n_snaps_ignored": n_snaps_ignored,
        "n_invalid_intervals": len(invalid_intervals),
        "n_grid_ticks": df_out.height,
        "n_valid_ticks": valid_ticks,
        "valid_ratio": round(valid_ticks / df_out.height, 6) if df_out.height else 0.0,
        "carry_state": final,
    }


def process_date(
    base: Path,
    date_str: str,
    *,
    overwrite: bool = False,
    seed_carry_from_previous_day: bool = True,
) -> None:
    print(f"\nbook20 reconstruct | {date_str}")

    carry_state = None
    if seed_carry_from_previous_day:
        prev_day = (
            datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
        ).strftime("%Y-%m-%d")
        ps = _load_state(base, prev_day)
        if ps and "final_carry_state" in ps:
            carry_state = ps["final_carry_state"]
            if carry_state and isinstance(carry_state.get("bids"), dict):
                carry_state["bids"] = {
                    _tick_key(k): v for k, v in carry_state["bids"].items()
                }
                carry_state["asks"] = {
                    _tick_key(k): v for k, v in carry_state["asks"].items()
                }
                print(
                    f"   seeded carry from {prev_day}/23h: "
                    f"u={carry_state.get('last_update_id')} valid={carry_state.get('valid')}"
                )

    reports = []
    for hour in range(24):
        rep = reconstruct_hour(
            base, date_str, hour, overwrite=overwrite, carry_state=carry_state
        )
        carry_state = rep.pop("carry_state", None)
        reports.append(rep)
        if rep["status"] == "ok":
            print(
                f"   {hour:02d}h events={rep['n_events']:>5} "
                f"diff_applied={rep['n_diff_applied']:>5} "
                f"gaps={rep['n_desyncs']:>3} "
                f"snap_ign={rep['n_snaps_ignored']:>4} "
                f"valid={rep['valid_ratio']*100:>6.2f}%"
            )
        elif rep["status"] == "missing_input":
            print(f"   {hour:02d}h (no input)")
        elif rep["status"] == "already_done":
            print(f"   {hour:02d}h already done")

    if carry_state:
        compact = {
            "last_update_id": carry_state.get("last_update_id"),
            "valid": carry_state.get("valid"),
            "anchor_kind": carry_state.get("anchor_kind"),
            "last_event_time": carry_state.get("last_event_time"),
            "n_bid_levels": len(carry_state.get("bids", {})),
            "n_ask_levels": len(carry_state.get("asks", {})),
        }
    else:
        compact = None

    _save_state(
        base,
        date_str,
        {
            "stage": "book_reconstruct",
            "hours": reports,
            "final_carry_state": carry_state,
            "final_carry_compact": compact,
            "written_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--date")
    p.add_argument("--from-date")
    p.add_argument("--to-date")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--no-seed",
        action="store_true",
        help="Don't seed from previous day's state",
    )
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

    print(f"book20 reconstruct | {len(dates)} dates")
    for d in dates:
        process_date(
            base, d, overwrite=args.overwrite, seed_carry_from_previous_day=not args.no_seed
        )

    print("\nbook reconstruct complete")


if __name__ == "__main__":
    main()
