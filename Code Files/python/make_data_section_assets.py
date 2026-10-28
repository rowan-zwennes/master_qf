#!/usr/bin/env python3
"""Market-characterisation tables and figure for the data section."""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

DATA_ROOT = Path("/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt")
OUT_DIR = Path(__file__).resolve().parent / "thesis" / "reports"
CACHE = OUT_DIR / "daily_market_stats.csv"

TICK_SIZE = 0.1

ANNUALISATION_DAYS = 365

COLS = [
    "ts_ms",
    "valid",
    "mid_price",
    "spread",
    "spread_bps",
    "bid_depth_l1",
    "ask_depth_l1",
    "n_trades_1s",
    "buy_vol_1s",
    "sell_vol_1s",
    "funding_rate",
    "seconds_to_funding",
    "basis_bps",
]


def load_splits() -> dict[str, str]:
    """Return {funding_day -> 'Pre'|'OOS'} from the authoritative splits.json."""
    s = json.loads((DATA_ROOT / "splits.json").read_text())
    mapping: dict[str, str] = {}
    for d in s["splits"]["pre_analysis"]:
        mapping[d] = "Pre"
    for d in s["splits"]["sim"]:
        mapping[d] = "OOS"
    return mapping


def aggregate_day(day: str) -> dict | None:
    """Aggregate one funding day's label files to scalar market descriptors."""
    next_day = str(date.fromisoformat(day) + timedelta(days=1))
    files = sorted((DATA_ROOT / day).glob("labels_*.parquet"))
    next_dir = DATA_ROOT / next_day
    if next_dir.exists():
        files += sorted(next_dir.glob("labels_*.parquet"))
    if not files:
        return None

    lf = (
        pl.scan_parquet(files)
        .filter(pl.col("funding_day") == day)
        .select(COLS)
        .sort("ts_ms")
    )
    df = lf.collect()

    bk = df.filter(pl.col("valid") & pl.col("mid_price").is_not_null())

    rv = bk.select(
        ts="ts_ms",
        lmid=pl.col("mid_price").log(),
    ).with_columns(
        dt=(pl.col("ts") - pl.col("ts").shift(1)),
        r=(pl.col("lmid") - pl.col("lmid").shift(1)),
    )
    good = rv.filter((pl.col("dt") == 1000) & pl.col("r").is_not_null())
    rv_daily = float(math.sqrt((good["r"] ** 2).sum())) if good.height else float("nan")

    notional = df.filter(pl.col("mid_price").is_not_null()).select(
        n=((pl.col("buy_vol_1s") + pl.col("sell_vol_1s")) * pl.col("mid_price"))
    )["n"].sum()

    settle = df.filter(
        (pl.col("seconds_to_funding") == 0) & pl.col("funding_rate").is_not_null()
    )["funding_rate"]
    n_settle = settle.len()

    return {
        "day": day,
        "mid_open": float(bk["mid_price"].head(1).item()) if bk.height else float("nan"),
        "mid_close": float(bk["mid_price"].tail(1).item()) if bk.height else float("nan"),
        "mid_mean": float(bk["mid_price"].mean()) if bk.height else float("nan"),
        "mid_min": float(bk["mid_price"].min()) if bk.height else float("nan"),
        "mid_max": float(bk["mid_price"].max()) if bk.height else float("nan"),
        "rv_daily_pct": rv_daily * 100.0,
        "rv_ann_pct": rv_daily * 100.0 * math.sqrt(ANNUALISATION_DAYS),
        "spread_bps_mean": float(bk["spread_bps"].mean()) if bk.height else float("nan"),
        "spread_ticks_mean": float(bk["spread"].mean()) / TICK_SIZE if bk.height else float("nan"),
        "tob_depth_btc_mean": float((bk["bid_depth_l1"] + bk["ask_depth_l1"]).mean())
        if bk.height
        else float("nan"),
        "n_trades_day": int(df["n_trades_1s"].sum()),
        "volume_btc_day": float((df["buy_vol_1s"] + df["sell_vol_1s"]).sum()),
        "notional_usd_day": float(notional) if notional is not None else float("nan"),
        "funding_settle_n": int(n_settle),
        "funding_settle_mean": float(settle.mean()) if n_settle else float("nan"),
        "funding_settle_absmean": float(settle.abs().mean()) if n_settle else float("nan"),
        "funding_settle_min": float(settle.min()) if n_settle else float("nan"),
        "funding_settle_max": float(settle.max()) if n_settle else float("nan"),
        "funding_settle_sum": float(settle.sum()) if n_settle else 0.0,
        "funding_settle_abssum": float(settle.abs().sum()) if n_settle else 0.0,
        "funding_settle_npos": int((settle > 0).sum()) if n_settle else 0,
        "abs_basis_bps_mean": float(df["basis_bps"].abs().mean()),
    }


def build_cache(mapping: dict[str, str]) -> pl.DataFrame:
    rows = []
    for i, day in enumerate(sorted(mapping), 1):
        rec = aggregate_day(day)
        if rec is None:
            print(f"  [{i:2d}/94] {day}: NO FILES, skipped")
            continue
        rec["period"] = mapping[day]
        rows.append(rec)
        print(
            f"  [{i:2d}/94] {day} ({rec['period']}): "
            f"mid={rec['mid_mean']:,.0f} rv={rv_fmt(rec['rv_daily_pct'])} "
            f"trades={rec['n_trades_day']:,} "
            f"fund_settle={rec['funding_settle_mean']*1e4:+.3f}bps (n={rec['funding_settle_n']})"
        )
    df = pl.DataFrame(rows)
    df.write_csv(CACHE)
    print(f"\nWrote cache: {CACHE} ({df.height} days)")
    return df


def rv_fmt(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:.2f}%"


def make_figure(df: pl.DataFrame) -> None:
    d = df.sort("day")
    dates = [np.datetime64(x) for x in d["day"].to_list()]
    is_oos = np.array([p == "OOS" for p in d["period"].to_list()])
    # Split boundary = midpoint between last Pre day and first OOS day.
    split_x = None
    if is_oos.any() and (~is_oos).any():
        last_pre = max(np.array(dates)[~is_oos])
        first_oos = min(np.array(dates)[is_oos])
        split_x = last_pre + (first_oos - last_pre) / 2

    navy = "#1f3b63"
    steel = "#3b6ea5"

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, axes = plt.subplots(3, 1, figsize=(6.3, 3.4), sharex=True)

    # (a) Mid-price (thousands USDT), min-max band per day.
    ax = axes[0]
    ax.fill_between(
        dates,
        np.array(d["mid_min"]) / 1e3,
        np.array(d["mid_max"]) / 1e3,
        color=steel,
        alpha=0.25,
        linewidth=0,
    )
    ax.plot(dates, np.array(d["mid_mean"]) / 1e3, color=navy, linewidth=1.2)
    ax.set_ylabel("Mid-price\n(k USDT)")
    # Headroom so the period annotations clear the price curve.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.20 * (y1 - y0))

    # (b) Daily realised volatility (%).
    ax = axes[1]
    ax.plot(dates, d["rv_daily_pct"], color=navy, linewidth=1.2)
    ax.set_ylabel("Realised vol\n(% / day)")

    ax = axes[2]
    fr_bps = np.array(d["funding_settle_mean"]) * 1e4
    fr_lo = np.array(d["funding_settle_min"]) * 1e4
    fr_hi = np.array(d["funding_settle_max"]) * 1e4
    ax.fill_between(dates, fr_lo, fr_hi, color=steel, alpha=0.25, linewidth=0, zorder=1)
    ax.axhline(0.0, color="0.6", linewidth=0.7, zorder=2)
    ax.plot(dates, fr_bps, color=navy, linewidth=1.2, zorder=3)
    ax.set_ylabel("Settlement\nfunding (bps)")
    ax.yaxis.set_major_locator(mticker.FixedLocator([-1, 0, 1]))
    ax.set_xlabel("Recording date (2026)")

    for ax in axes:
        if split_x is not None:
            ax.axvline(split_x, color="#a02020", linestyle="--", linewidth=1.0)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        ax.margins(x=0.01)
    axes[0].annotate(
        "pre-analysis", xy=(0.01, 0.86), xycoords="axes fraction", fontsize=7.5, color="0.35"
    )
    axes[0].annotate(
        "out-of-sample", xy=(0.46, 0.86), xycoords="axes fraction", fontsize=7.5, color="0.35"
    )
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())

    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.6)
    out = OUT_DIR / "market_overview.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figure: {out}")


def _per_period(df: pl.DataFrame):
    pre = df.filter(pl.col("period") == "Pre")
    oos = df.filter(pl.col("period") == "OOS")
    return pre, oos, df


def make_market_table(df: pl.DataFrame) -> None:
    pre, oos, allp = _per_period(df)

    def col(frame: pl.DataFrame):
        return {
            "mid_lo": frame["mid_min"].min(),
            "mid_hi": frame["mid_max"].max(),
            "rv": frame["rv_daily_pct"].mean(),
            "rv_ann": frame["rv_ann_pct"].mean(),
            "spread_bps": frame["spread_bps_mean"].mean(),
            "spread_ticks": frame["spread_ticks_mean"].mean(),
            "depth": frame["tob_depth_btc_mean"].mean(),
            "trades": frame["n_trades_day"].mean() / 1e6,
            "notional": frame["notional_usd_day"].mean() / 1e9,
        }

    cp, co, ca = col(pre), col(oos), col(allp)
    rows = [
        ("Mid-price range (thousand USDT)", "{:.1f}--{:.1f}",
         (cp["mid_lo"] / 1e3, cp["mid_hi"] / 1e3),
         (co["mid_lo"] / 1e3, co["mid_hi"] / 1e3),
         (ca["mid_lo"] / 1e3, ca["mid_hi"] / 1e3)),
        ("Realised volatility (\\% / day)", "{:.2f}", (cp["rv"],), (co["rv"],), (ca["rv"],)),
        ("Realised volatility (\\% ann.)", "{:.1f}", (cp["rv_ann"],), (co["rv_ann"],), (ca["rv_ann"],)),
        ("Mean spread (bps)", "{:.3f}", (cp["spread_bps"],), (co["spread_bps"],), (ca["spread_bps"],)),
        ("Mean spread (ticks)", "{:.2f}", (cp["spread_ticks"],), (co["spread_ticks"],), (ca["spread_ticks"],)),
        ("Mean top-of-book depth (BTC)", "{:.2f}", (cp["depth"],), (co["depth"],), (ca["depth"],)),
        ("Mean daily trades (millions)", "{:.2f}", (cp["trades"],), (co["trades"],), (ca["trades"],)),
        ("Mean daily traded notional (billion USDT)", "{:.2f}", (cp["notional"],), (co["notional"],), (ca["notional"],)),
    ]

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Descriptive market characteristics of the BTCUSDT perpetual "
        r"over the study window (04:00 UTC 19 January to 04:00 UTC 23 April 2026), "
        r"by period (Pre\,=\,pre-analysis days 1--30; "
        r"OOS\,=\,out-of-sample days 31--94; All\,=\,full 94-day window). All "
        r"quantities are computed on the 1\,s grid over valid book ticks; the "
        r"mid-price range is the daily low--high envelope, all other entries are "
        r"period means of the per-day statistic. Realised volatility is the "
        r"daily root of summed squared 1\,s mid-price log-returns, also reported "
        r"annualised, i.e.\ multiplied by $\sqrt{365}$. Spread in basis points is "
        r"$10^4 \times \text{spread}/S_t$ (relative to mid-price); spread in "
        r"ticks is $\text{spread}/\$0.10$ (absolute tick units). Depth is the "
        r"resting quantity displayed at the best bid and best ask combined; "
        r"traded notional is price "
        r"times quantity summed over the day's trades. The near-identical "
        r"top-of-book conditions (spread, depth) and the overlapping volatility ranges "
        r"across the two periods support the walk-forward premise of "
        r"Section~\ref{sec:methodology}; the difference in price level reflects "
        r"the directional move documented in Figure~\ref{fig:market_overview}.}",
        r"\label{tab:market_characteristics}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Characteristic & Pre & OOS & All \\",
        r"\hline",
    ]
    for label, fmt, vp, vo, va in rows:
        lines.append(f"{label} & {fmt.format(*vp)} & {fmt.format(*vo)} & {fmt.format(*va)} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]

    out = OUT_DIR / "tab_market_characteristics.tex"
    out.write_text("\n".join(lines))
    print(f"Wrote table:  {out}")


def make_funding_table(df: pl.DataFrame) -> None:
    pre, oos, allp = _per_period(df)

    def col(frame: pl.DataFrame):
        n = int(frame["funding_settle_n"].sum() or 0)
        return {
            "mean": (frame["funding_settle_sum"].sum() / n * 1e4) if n else float("nan"),
            "absmean": (frame["funding_settle_abssum"].sum() / n * 1e4) if n else float("nan"),
            "min": frame["funding_settle_min"].min() * 1e4,
            "max": frame["funding_settle_max"].max() * 1e4,
            "frac_pos": (frame["funding_settle_npos"].sum() / n * 100.0) if n else float("nan"),
            "n": n,
            "abs_basis": frame["abs_basis_bps_mean"].mean(),
        }

    cp, co, ca = col(pre), col(oos), col(allp)
    rows = [
        ("Mean settlement funding (bps)", "{:+.3f}", cp["mean"], co["mean"], ca["mean"]),
        ("Mean abs.\\ settlement funding (bps)", "{:.3f}", cp["absmean"], co["absmean"], ca["absmean"]),
        ("Min settlement funding (bps)", "{:+.3f}", cp["min"], co["min"], ca["min"]),
        ("Max settlement funding (bps)", "{:+.3f}", cp["max"], co["max"], ca["max"]),
        ("Settlements with funding $>0$ (\\%)", "{:.1f}", cp["frac_pos"], co["frac_pos"], ca["frac_pos"]),
        ("Number of settlements", "{:.0f}", float(cp["n"]), float(co["n"]), float(ca["n"])),
        ("Mean abs.\\ premium basis (bps)", "{:.3f}", cp["abs_basis"], co["abs_basis"], ca["abs_basis"]),
    ]

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Funding and carry environment of the BTCUSDT perpetual over "
        r"the study window (04:00 UTC 19 January to 04:00 UTC 23 April 2026), "
        r"by period (Pre / OOS / All as in "
        r"Table~\ref{tab:market_characteristics}). Statistics are computed over "
        r"the funding rate \emph{as realised at each 8-hour settlement} (the rate "
        r"actually charged on position notional, recovered as the last published "
        r"estimate immediately before settlement), not the time-average of the "
        r"continuously-published estimate, which mean-reverts within the funding "
        r"window and would understate the realised carry; all entries are in basis "
        r"points (1\,bp\,$=\,0.01\%$). The mean pools over individual settlements "
        r"across the period. The premium basis is "
        r"$(\text{mark}-\text{mid})/\text{mid}$ in basis points. Settlement "
        r"funding is small but rarely zero, and its sign turns over between the "
        r"two periods, positive at two settlements in three before the split and "
        r"at fewer than one in two after it. That sign-switching carry is the "
        r"friction the Hamilton-Jacobi-Bellman extension of "
        r"Section~\ref{sec:math_foundation} is constructed to internalise: the "
        r"correction responds to the rate standing at each individual "
        r"settlement, never to a period average, and the gap between the mean "
        r"and the mean absolute rate measures how often that sign turns over.}",
        r"\label{tab:funding_characteristics}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Characteristic & Pre & OOS & All \\",
        r"\hline",
    ]
    for label, fmt, vp, vo, va in rows:
        lines.append(f"{label} & {fmt.format(vp)} & {fmt.format(vo)} & {fmt.format(va)} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]

    out = OUT_DIR / "tab_funding_characteristics.tex"
    out.write_text("\n".join(lines))
    print(f"Wrote table:  {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-cache", action="store_true", help="reuse daily_market_stats.csv")
    args = ap.parse_args()

    mapping = load_splits()
    if args.use_cache and CACHE.exists():
        print(f"Reading cache: {CACHE}")
        df = pl.read_csv(CACHE)
    else:
        print("Aggregating 94 funding days from the 1s feature grid")
        df = build_cache(mapping)

    make_figure(df)
    make_market_table(df)
    make_funding_table(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
