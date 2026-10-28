"""Stage 02b: structural checks on the reconstructed book files."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

BASE = Path("/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt")
FIRST, LAST = "2026-01-19", "2026-04-23"
TOP_N = 20
GRID_MS = 100
ROWS_PER_HOUR = 36000
TICK_SIZE = 0.10  # BTCUSDT perpetual futures tick size (USDT)

ANOMALY_KEYS = [
    "crossed", "locked", "bid_unsorted", "ask_unsorted",
    "bid_gap", "ask_gap", "pair_bad", "neg_price", "neg_qty",
    "off_grid", "nan_inf", "empty_valid", "lui_unsorted", "bad_anchor",
    "ts_dup", "ts_unsorted", "ts_bad_step",
]


def _dates() -> list[str]:
    out, d = [], datetime.strptime(FIRST, "%Y-%m-%d")
    end = datetime.strptime(LAST, "%Y-%m-%d")
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def _audit_hour(df: pl.DataFrame) -> dict:
    r = {k: 0 for k in ANOMALY_KEYS}
    r["rows"] = df.height
    r["valid"] = int(df["valid"].sum() or 0)
    r["spread_min"] = None
    r["spread_max"] = None
    r["depth_full"] = 0  # valid rows with all 20 bid levels populated

    ts = df["ts_ms"]
    r["ts_dup"] = df.height - ts.n_unique()
    r["ts_unsorted"] = 0 if ts.is_sorted() else 1
    if df.height > 1:
        steps = ts.diff().drop_nulls()
        r["ts_bad_step"] = int((steps != GRID_MS).sum())

    lui = df["last_update_id"].drop_nulls()
    r["lui_unsorted"] = 0 if lui.is_sorted() else 1
    bad_anchor = set(df["anchor_kind"].unique().to_list()) - {
        "snapshot", "diff", "invalid", None
    }
    r["bad_anchor"] = len(bad_anchor)
    r["anchor_kinds"] = sorted(str(x) for x in df["anchor_kind"].unique().to_list())

    v = df.filter(pl.col("valid"))
    if v.height == 0:
        return r

    bp0, ap0 = pl.col("bid_p_0"), pl.col("ask_p_0")
    both0 = bp0.is_not_null() & ap0.is_not_null()

    exprs = {
        "crossed": (both0 & (bp0 > ap0)).sum(),
        "locked": (both0 & (bp0 == ap0)).sum(),
        "empty_valid": (bp0.is_null() | ap0.is_null()).sum(),
        "depth_full": pl.col(f"bid_p_{TOP_N - 1}").is_not_null().sum(),
    }
    for i in range(TOP_N):
        for side in ("bid", "ask"):
            p = pl.col(f"{side}_p_{i}")
            q = pl.col(f"{side}_q_{i}")
            exprs[f"_negp_{side}{i}"] = (p.is_not_null() & (p <= 0)).sum()
            exprs[f"_negq_{side}{i}"] = (q.is_not_null() & (q <= 0)).sum()
            exprs[f"_nan_{side}p{i}"] = (
                p.is_not_null() & (p.is_nan() | p.is_infinite())
            ).sum()
            exprs[f"_nan_{side}q{i}"] = (
                q.is_not_null() & (q.is_nan() | q.is_infinite())
            ).sum()
            exprs[f"_pair_{side}{i}"] = (p.is_null() != q.is_null()).sum()
        if i < TOP_N - 1:
            bi, bn = pl.col(f"bid_p_{i}"), pl.col(f"bid_p_{i + 1}")
            ai, an = pl.col(f"ask_p_{i}"), pl.col(f"ask_p_{i + 1}")
            exprs[f"_bu_{i}"] = (bi.is_not_null() & bn.is_not_null() & (bi <= bn)).sum()
            exprs[f"_au_{i}"] = (ai.is_not_null() & an.is_not_null() & (ai >= an)).sum()
            exprs[f"_bg_{i}"] = (bi.is_null() & bn.is_not_null()).sum()
            exprs[f"_ag_{i}"] = (ai.is_null() & an.is_not_null()).sum()
            bid_gap = bi - bn
            ask_gap = an - ai
            exprs[f"_ogb_{i}"] = (
                bi.is_not_null() & bn.is_not_null() & (bi > bn)
                & (((bid_gap / TICK_SIZE) - (bid_gap / TICK_SIZE).round()).abs() > 0.01)
            ).sum()
            exprs[f"_oga_{i}"] = (
                ai.is_not_null() & an.is_not_null() & (an > ai)
                & (((ask_gap / TICK_SIZE) - (ask_gap / TICK_SIZE).round()).abs() > 0.01)
            ).sum()

    agg = v.select(**{k: e for k, e in exprs.items()}).row(0, named=True)
    for k, val in agg.items():
        val = int(val or 0)
        if k in ("crossed", "locked", "empty_valid", "depth_full"):
            r[k] = val
        elif k.startswith("_negp_"):
            r["neg_price"] += val
        elif k.startswith("_negq_"):
            r["neg_qty"] += val
        elif k.startswith("_nan_"):
            r["nan_inf"] += val
        elif k.startswith("_pair_"):
            r["pair_bad"] += val
        elif k.startswith("_ogb_") or k.startswith("_oga_"):
            r["off_grid"] += val
        elif k.startswith("_bu_"):
            r["bid_unsorted"] += val
        elif k.startswith("_au_"):
            r["ask_unsorted"] += val
        elif k.startswith("_bg_"):
            r["bid_gap"] += val
        elif k.startswith("_ag_"):
            r["ask_gap"] += val

    spr = v.filter(both0).select((ap0 - bp0).alias("s"))["s"]
    if spr.len():
        r["spread_min"] = float(spr.min())
        r["spread_max"] = float(spr.max())
    return r


def main() -> None:
    dates = _dates()
    print(f"book20 audit | {dates[0]}..{dates[-1]}\n")

    recs: list[dict] = []
    anchor_seen: set[str] = set()
    for date in dates:
        ddir = BASE / date
        if not ddir.exists():
            print(f"  {date}: no date dir")
            continue
        d_rows = d_valid = 0
        d_anom = {k: 0 for k in ANOMALY_KEYS}
        d_files = 0
        spr_lo, spr_hi = None, None
        for h in range(24):
            f = ddir / f"book20_{h:02d}h.parquet"
            if not f.exists():
                continue
            d_files += 1
            rec = _audit_hour(pl.read_parquet(f))
            rec["date"], rec["hour"] = date, h
            recs.append(rec)
            d_rows += rec["rows"]
            d_valid += rec["valid"]
            for k in ANOMALY_KEYS:
                d_anom[k] += rec[k]
            anchor_seen.update(rec.get("anchor_kinds", []))
            if rec["spread_min"] is not None:
                spr_lo = rec["spread_min"] if spr_lo is None else min(spr_lo, rec["spread_min"])
                spr_hi = rec["spread_max"] if spr_hi is None else max(spr_hi, rec["spread_max"])

        flags = [k for k in ANOMALY_KEYS if d_anom[k]]
        vr = (d_valid / d_rows * 100) if d_rows else 0.0
        n_inv = d_rows - d_valid
        spread_txt = f"spread[{spr_lo:.2f},{spr_hi:.2f}]" if spr_lo is not None else "spread[--]"
        inv_txt = f"  invalid={n_inv}" if n_inv > 0 else ""
        flag_txt = ("   " + " ".join(f"{k}={d_anom[k]}" for k in flags)) if flags else ""
        print(
            f"  {date}  files={d_files:2d}  rows={d_rows:>7}  valid={vr:6.2f}%{inv_txt}  "
            f"{spread_txt}{flag_txt}"
        )

    if not recs:
        sys.exit("no book20 files found")

    rdf = pl.DataFrame(recs)
    tot_rows = int(rdf["rows"].sum())
    tot_valid = int(rdf["valid"].sum())
    print(f"\n{'=' * 70}")
    print(f"GLOBAL | {len(recs)} hour-files, {tot_rows} rows, "
          f"{tot_valid} valid ({tot_valid / tot_rows * 100:.2f}%)")
    print(f"anchor_kind values seen: {sorted(anchor_seen)}")
    sm = rdf["spread_min"].drop_nulls()
    sx = rdf["spread_max"].drop_nulls()
    if sm.len():
        print(f"spread range over all hours: min={sm.min():.4f}  max={sx.max():.4f}")

    # hours with a non-standard row count
    odd = rdf.filter(pl.col("rows") != ROWS_PER_HOUR)
    if odd.height:
        print(f"\nhours with rows != {ROWS_PER_HOUR}: {odd.height}")
        for row in odd.select("date", "hour", "rows").head(20).iter_rows():
            print(f"   {row[0]} {row[1]:02d}h  rows={row[2]}")

    print(f"\n{'-' * 70}\nANOMALY SUMMARY")
    any_anom = False
    for k in ANOMALY_KEYS:
        total = int(rdf[k].sum())
        if total == 0:
            print(f"  {k:<14} 0")
            continue
        any_anom = True
        hits = rdf.filter(pl.col(k) > 0)
        print(f"  {k:<14} {total}  in {hits.height} hour-file(s):")
        for row in hits.select("date", "hour", k).head(25).iter_rows():
            print(f"       {row[0]} {row[1]:02d}h  {k}={row[2]}")
        if hits.height > 25:
            print(f"       ... and {hits.height - 25} more")

    print(f"\n{'=' * 70}")
    if any_anom:
        print(" anomalies found, see breakdown above")
    else:
        print("no structural anomalies across the whole dataset")


if __name__ == "__main__":
    main()
