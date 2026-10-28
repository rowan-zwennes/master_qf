"""The combined results figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from analysis_markout import U_BUCKET_LABELS, U_BUCKETS
from analysis_pnl import STRATEGY_LABELS, daily_pnl_matrix, load_daily_panel
from statistical_tests import sharpe_ratio

STRAT_COLORS = {1: "tab:gray", 2: "tab:blue", 3: "tab:olive",
                4: "tab:green", 5: "tab:orange", 6: "tab:red"}
EVENT_WINDOW_S = 1_800
QUOTE_SIZE = 0.005


def load_snaps(run_dir: Path) -> pl.DataFrame:
    """All 1 s snapshots across strategies/days, pause rows dropped (their
    mid/mtm are NaN by construction in run_simulation)."""
    frames = []
    for sdir in sorted(run_dir.glob("s[0-9]")):
        sid = int(sdir.name[1:])
        for fp in sorted(sdir.glob("pnl_*.parquet")):
            df = pl.read_parquet(fp)
            if not df.height:
                continue
            frames.append(df.with_columns(
                pl.lit(sid).alias("strategy"),
                pl.lit(fp.stem[4:]).alias("funding_day"),
            ))
    if not frames:
        raise FileNotFoundError(f"no pnl_*.parquet under {run_dir}")
    out = pl.concat(frames, how="vertical_relaxed")
    return out.filter(~pl.col("paused") & pl.col("mtm").is_finite())


_BUCKET_ENUM = pl.Enum(U_BUCKET_LABELS)


def _proj(snaps: pl.DataFrame, cols: tuple[str, ...]) -> pl.DataFrame:
    """Narrow `snaps` to the columns one panel needs before any with_columns."""
    return snaps.select([c for c in cols if c in snaps.columns])


def _bucket_expr() -> pl.Expr:
    """Map u_s onto the shared U_BUCKETS labels."""
    n = len(U_BUCKET_LABELS)
    expr = pl.lit(None, dtype=pl.Int8)
    for i, (lo, hi) in reversed(list(enumerate(U_BUCKETS))):
        expr = (pl.when((pl.col("u_s") >= lo) & (pl.col("u_s") < hi))
                .then(pl.lit(i, dtype=pl.Int8)).otherwise(expr))
    # u_s == 28_800 exactly (snapshot at the settlement tick) -> last bucket
    expr = (pl.when(pl.col("u_s") >= U_BUCKETS[-1][1])
            .then(pl.lit(n - 1, dtype=pl.Int8)).otherwise(expr))
    return expr.replace_strict(list(range(n)), U_BUCKET_LABELS, default=None,
                               return_dtype=_BUCKET_ENUM).alias("bucket")


def inventory_by_u(snaps: pl.DataFrame) -> pl.DataFrame:
    aggs = [pl.col("inv").abs().mean().alias("mean_abs_inv"),
            pl.len().alias("n")]
    if "f_rate" in snaps.columns:
        aggs.append((pl.col("inv") * -pl.col("f_rate").sign())
                    .mean().alias("mean_aligned_inv"))
    return (_proj(snaps, ("strategy", "inv", "u_s", "f_rate"))
            .with_columns(_bucket_expr())
            .group_by("strategy", "bucket")
            .agg(*aggs)
            .sort("strategy", "bucket"))


def skew_by_u(snaps: pl.DataFrame, min_abs_inv: float = 0.5) -> pl.DataFrame:
    """Mean reservation skew (bps of mid) per strategy x u-bucket x inventory sign."""
    df = (_proj(snaps, ("strategy", "inv", "u_s", "delta_a", "delta_b", "mid"))
          .filter(pl.col("delta_b").is_finite()
                  & pl.col("delta_a").is_finite()
                  & (pl.col("inv").abs() >= min_abs_inv))
          .with_columns(
              _bucket_expr(),
              ((pl.col("delta_a") - pl.col("delta_b")) / 2.0
               / pl.col("mid") * 1e4).alias("skew_bp"),
              pl.when(pl.col("inv") > 0).then(pl.lit("q>0"))
              .otherwise(pl.lit("q<0")).alias("inv_sign"),
          ))
    return (df.group_by("strategy", "bucket", "inv_sign")
            .agg(pl.col("skew_bp").mean().alias("mean_skew_bp"),
                 (pl.col("skew_bp").std() / pl.len().sqrt()).alias("se"),
                 pl.len().alias("n"))
            .sort("strategy", "bucket", "inv_sign"))


def skew_fund_by_u(snaps: pl.DataFrame) -> pl.DataFrame:
    """Funding-directed reservation skew per strategy x u-bucket."""
    if "f_rate" not in snaps.columns:
        return pl.DataFrame()
    df = (_proj(snaps, ("strategy", "u_s", "delta_a", "delta_b", "mid",
                        "f_rate"))
          .filter(pl.col("delta_b").is_finite()
                  & pl.col("delta_a").is_finite()
                  & (pl.col("f_rate") != 0.0))
          .with_columns(
              _bucket_expr(),
              (-pl.col("f_rate").sign()
               * (pl.col("delta_a") - pl.col("delta_b")) / 2.0
               / pl.col("mid") * 1e4).alias("skew_fund_bp"),
          ))
    return (df.group_by("strategy", "bucket")
            .agg(pl.col("skew_fund_bp").mean().alias("mean_skew_fund_bp"),
                 (pl.col("skew_fund_bp").std() / pl.len().sqrt()).alias("se"),
                 pl.len().alias("n"))
            .sort("strategy", "bucket"))


def settlement_funding(snaps: pl.DataFrame) -> pl.DataFrame:
    """Per-settlement funding outcome per strategy: how often the book ends a
    settlement on the collecting side, and the mean cash per settlement.

    A settlement is a wrap of u_s (jump back toward the epoch length); the
    funding cash it books is the step of funding_cum across the wrap tick.
    |step| <= 1e-3 USD is counted as flat (inventory ~ 0 at the snapshot)."""
    recs = []
    for sid in sorted(snaps["strategy"].unique().to_list()):
        steps: list[float] = []
        sub = snaps.filter(pl.col("strategy") == sid)
        for fday in sub["funding_day"].unique().to_list():
            day = sub.filter(pl.col("funding_day") == fday).sort("ts_ms")
            u = day["u_s"].to_numpy()
            fc = day["funding_cum"].to_numpy()
            for w in np.where(np.diff(u) > 1000)[0]:
                steps.append(float(fc[w + 1] - fc[w]))
        s = np.asarray(steps)
        thr = 1e-3
        recs.append({
            "strategy": sid, "n_settlements": int(s.size),
            "collect_pct": float(100.0 * (s > thr).mean()),
            "pay_pct": float(100.0 * (s < -thr).mean()),
            "flat_pct": float(100.0 * (np.abs(s) <= thr).mean()),
            "mean_usd": float(s.mean()), "sum_usd": float(s.sum()),
        })
    return pl.DataFrame(recs)


def sharpe_by_u(snaps: pl.DataFrame) -> pl.DataFrame:
    """Annualised Sharpe per strategy x u-bucket from per-day bucket P&L.

    The MTM increment between consecutive surviving snapshots of one
    (strategy, day) is assigned to the bucket of the LATER snapshot; per-day
    sums across each bucket then feed the cross-day Sharpe.
    """
    df = (_proj(snaps, ("strategy", "funding_day", "ts_ms", "mtm", "u_s"))
          .sort("strategy", "funding_day", "ts_ms")
          .with_columns(
              (pl.col("mtm").diff().over("strategy", "funding_day"))
              .alias("dpnl"))
          .filter(pl.col("dpnl").is_finite())
          .with_columns(_bucket_expr()))
    daily = (df.group_by("strategy", "funding_day", "bucket")
             .agg(pl.col("dpnl").sum().alias("bucket_pnl")))
    recs = []
    for sid in sorted(daily["strategy"].unique().to_list()):
        for lab in U_BUCKET_LABELS:
            x = (daily.filter((pl.col("strategy") == sid)
                              & (pl.col("bucket") == lab))
                 ["bucket_pnl"].to_numpy())
            recs.append({
                "strategy": sid, "bucket": lab, "n_days": int(x.size),
                "sharpe": sharpe_ratio(x) if x.size >= 3 else float("nan"),
                "mean_pnl": float(x.mean()) if x.size else float("nan"),
            })
    return pl.DataFrame(recs)


def fig_money(path: Path, days: list[str], daily: dict[int, np.ndarray],
              inv_tab: pl.DataFrame, skew_tab: pl.DataFrame,
              sharpe_tab: pl.DataFrame,
              fund_tab: pl.DataFrame | None = None,
              skewf_tab: pl.DataFrame | None = None,
              event_tab: pl.DataFrame | None = None) -> None:
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 8.5,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7.5, "axes.titlesize": 9.5})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
    ax_b, ax_c = axes
    xpos = np.arange(len(U_BUCKET_LABELS))
    sids = sorted(daily.keys()) or sorted(inv_tab["strategy"].unique().to_list())

    have_aligned = "mean_aligned_inv" in inv_tab.columns
    for sid in (s for s in (2, 5, 6) if s in sids):
        sub = inv_tab.filter(pl.col("strategy") == sid)
        y = [float(sub.filter(pl.col("bucket") == lab)["mean_abs_inv"][0])
             if sub.filter(pl.col("bucket") == lab).height else np.nan
             for lab in U_BUCKET_LABELS]
        ax_b.plot(xpos, y, "o-", ms=3,
                  label=STRATEGY_LABELS.get(sid, str(sid)),
                  color=STRAT_COLORS.get(sid), lw=1.8 if sid == 6 else 1.0)
        if have_aligned:
            ya = [float(sub.filter(pl.col("bucket") == lab)
                        ["mean_aligned_inv"][0])
                  if sub.filter(pl.col("bucket") == lab).height else np.nan
                  for lab in U_BUCKET_LABELS]
            ax_b.plot(xpos, ya, "s--", ms=3, alpha=0.7,
                      color=STRAT_COLORS.get(sid),
                      lw=1.4 if sid == 6 else 0.8)
    if have_aligned:
        ax_b.axhline(0, color="k", lw=0.5)
        from matplotlib.lines import Line2D
        ax_b.legend(handles=[
            Line2D([], [], color="0.3", ls="-", marker="o", ms=3,
                   label=r"held size $\overline{|q_t|}$"),
            Line2D([], [], color="0.3", ls="--", marker="s", ms=3,
                   label=r"collecting-side $-\overline{q_t\,\mathrm{sgn}(f_t)}$"),
        ] + [Line2D([], [], color=STRAT_COLORS[s], lw=1.6,
                    label=STRATEGY_LABELS.get(s, str(s)))
             for s in (2, 5, 6) if s in sids],
            loc="upper left", fontsize=6.5, frameon=False,
            handlelength=1.6, labelspacing=0.3)
    ax_b.set_xticks(xpos, U_BUCKET_LABELS)
    ax_b.set_xlabel(r"time to funding settlement $u$")
    ax_b.set_ylabel("lots")
    ax_b.set_ylim(top=ax_b.get_ylim()[1] * 1.52)  # legend headroom
    ax_b.invert_xaxis()
    ax_b.set_title("(a) Inventory into the settlement")
    ax_b.grid(alpha=0.3)

    n_set = 0
    if event_tab is not None and event_tab.height:
        sub = event_tab.sort("min_to")
        x = sub["min_to"].to_numpy()
        if {"q25_bp", "q75_bp"} <= set(sub.columns):
            ax_c.fill_between(x, sub["q25_bp"].to_numpy(),
                              sub["q75_bp"].to_numpy(),
                              color=STRAT_COLORS.get(5), alpha=0.18, lw=0,
                              label="inter-quartile range")
            n_set = int(sub["n_settlements"].max())
        ax_c.plot(x, sub["skew_fund_diff_bp"].to_numpy(),
                  color=STRAT_COLORS.get(5), lw=1.2,
                  label="funding minus GLT skew (median)")
        u_star = float(sub["u_star_min"][0])
        if np.isfinite(u_star):
            ax_c.axvline(u_star, color="tab:purple", lw=0.9, ls=":",
                         label="Regime II latch (median)")
        ax_c.axvline(0, color="k", lw=0.8)
        ax_c.set_xlim(float(x.min()), float(x.max()))
    ax_c.axhline(0, color="k", lw=0.6)
    ax_c.set_xlabel("minutes to settlement"
                    + (f" ({n_set} settlements)" if n_set else ""))
    ax_c.set_ylabel(r"$\Delta$ collecting-side skew (bp of mid price)")
    ax_c.set_title("(b) The correction's quote footprint")
    ax_c.legend(loc="upper left", fontsize=7, frameon=False,
                handlelength=1.6, labelspacing=0.3)
    ax_c.grid(alpha=0.3)


    fig.tight_layout()
    fig.savefig(path)
    if path.suffix == ".pdf":  # png twin for quick visual checks
        fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def pick_event(run_dir: Path, snaps: pl.DataFrame) -> tuple[str, int]:
    """(median-quality funding day of s6 by terminal P&L, settlement ts_ms).

    Within the median day, the settlement with the largest s6 mean |inventory|
    in the 5 minutes before it is shown (strategy-observable, not
    outcome-based).
    """
    pnls = []
    for fp in sorted((run_dir / "s6").glob("summary_*.json")):
        d = json.loads(fp.read_text())
        pnls.append((float(d["terminal_pnl"]), d["funding_day"]))
    if not pnls:
        raise FileNotFoundError(f"no s6 summaries under {run_dir}")
    pnls.sort()
    fday = pnls[(len(pnls) - 1) // 2][1]

    day = snaps.filter((pl.col("strategy") == 6)
                       & (pl.col("funding_day") == fday)).sort("ts_ms")
    # settlement ticks: u_s wraps from ~0 back up to the epoch length
    u = day["u_s"].to_numpy()
    ts = day["ts_ms"].to_numpy()
    wrap = np.where(np.diff(u) > 0)[0]
    best_ts, best_q = None, -1.0
    for w in wrap:
        t_set = int(ts[w]) + int(round(float(u[w]) * 1000.0))
        pre = day.filter((pl.col("ts_ms") >= t_set - 300_000)
                         & (pl.col("ts_ms") < t_set))
        q = float(pre["inv"].abs().mean()) if pre.height else 0.0
        if q > best_q:
            best_q, best_ts = q, t_set
    if best_ts is None:  # day with a single epoch and no wrap in-window
        best_ts = int(ts[-1]) + int(round(float(u[-1]) * 1000.0))
    return fday, best_ts


def settlement_times(snaps: pl.DataFrame) -> list[tuple[str, int, float]]:
    """(funding day, settlement ts_ms, f at settle) for EVERY settlement wrap."""
    has_f = "f_rate" in snaps.columns
    sub = _proj(snaps, ("strategy", "funding_day", "ts_ms", "u_s", "f_rate")
                ).filter(pl.col("strategy") == 2)
    out: list[tuple[str, int, float]] = []
    for fday in sub["funding_day"].unique().to_list():
        day = sub.filter(pl.col("funding_day") == fday).sort("ts_ms")
        u = day["u_s"].to_numpy()
        ts = day["ts_ms"].to_numpy()
        fr = (day["f_rate"].to_numpy() if has_f
              else np.zeros(day.height, dtype=float))
        for w in np.where(np.diff(u) > 1000)[0]:
            out.append((str(fday),
                        int(ts[w]) + int(round(float(u[w]) * 1000.0)),
                        float(fr[w])))
    return out


def event_skew_pooled(snaps: pl.DataFrame, post_s: int = 300) -> pl.DataFrame:
    """Cross-settlement funding-directed skew profile, s5 minus s2."""
    if "f_rate" not in snaps.columns:
        return pl.DataFrame()
    sets = settlement_times(snaps)
    if not sets:
        return pl.DataFrame()
    st = pl.DataFrame({"t_set": sorted({t for _, t, _ in sets})}).with_columns(
        pl.col("t_set").cast(pl.Int64)).sort("t_set")
    base = (snaps.filter(pl.col("strategy").is_in([2, 5])
                         & pl.col("delta_a").is_finite()
                         & pl.col("delta_b").is_finite()
                         & pl.col("mid").is_finite() & (pl.col("mid") > 0)
                         & (pl.col("f_rate") != 0.0))
            .select("ts_ms", "strategy", "delta_a", "delta_b", "mid",
                    "f_rate", "regime")
            .with_columns(pl.col("ts_ms").cast(pl.Int64))
            .sort("ts_ms"))
    if not base.height:
        return pl.DataFrame()
    ev = (base.join_asof(st, left_on="ts_ms", right_on="t_set",
                         strategy="nearest")
          .with_columns(((pl.col("ts_ms") - pl.col("t_set")) / 1000.0)
                        .round(0).cast(pl.Int64).alias("sec_to"))
          .filter((pl.col("sec_to") >= -EVENT_WINDOW_S)
                  & (pl.col("sec_to") <= post_s))
          .with_columns((-pl.col("f_rate").sign()
                         * (pl.col("delta_a") - pl.col("delta_b")) / 2.0
                         / pl.col("mid") * 1e4).alias("skew_fund_bp")))
    if not ev.height:
        return pl.DataFrame()
    r2 = (ev.filter((pl.col("strategy") == 5) & (pl.col("regime") == 2)
                    & (pl.col("sec_to") < 0))
          .group_by("t_set").agg(pl.col("sec_to").min().alias("latch")))
    lat = float(r2["latch"].median()) / 60.0 if r2.height else float("nan")
    agg = (ev.group_by("strategy", "t_set", "sec_to")
           .agg(pl.col("skew_fund_bp").mean()))
    s2 = agg.filter(pl.col("strategy") == 2).select(
        "t_set", "sec_to", pl.col("skew_fund_bp").alias("skew2"))
    s5 = agg.filter(pl.col("strategy") == 5).select(
        "t_set", "sec_to", pl.col("skew_fund_bp").alias("skew5"))
    d = s2.join(s5, on=["t_set", "sec_to"], how="inner")
    if not d.height:
        return pl.DataFrame()
    return (d.with_columns((pl.col("skew5") - pl.col("skew2")).alias("diff_bp"))
            .group_by("sec_to")
            .agg(pl.col("diff_bp").median().alias("skew_fund_diff_bp"),
                 pl.col("diff_bp").quantile(0.25).alias("q25_bp"),
                 pl.col("diff_bp").quantile(0.75).alias("q75_bp"),
                 pl.col("t_set").n_unique().alias("n_settlements"))
            .with_columns((pl.col("sec_to") / 60.0).alias("min_to"),
                          pl.lit(lat).alias("u_star_min"))
            .sort("min_to"))


def pick_event_by_carry(snaps: pl.DataFrame) -> tuple[str, int, float]:
    """(funding day, settlement ts_ms, f at settle) of the out-of-sample
    settlement with the largest |funding rate| as-of the settlement tick.

    The funding rate is exogenous to every strategy, so selecting on it is
    not outcome selection. Scanned on the s2 snapshots (present in every
    run set)."""
    sub = snaps.filter(pl.col("strategy") == 2)
    best = ("", 0, 0.0)
    for fday in sub["funding_day"].unique().to_list():
        day = sub.filter(pl.col("funding_day") == fday).sort("ts_ms")
        u = day["u_s"].to_numpy()
        ts = day["ts_ms"].to_numpy()
        fr = day["f_rate"].to_numpy()
        for w in np.where(np.diff(u) > 1000)[0]:
            if abs(float(fr[w])) > abs(best[2]):
                t_set = int(ts[w]) + int(round(float(u[w]) * 1000.0))
                best = (str(fday), t_set, float(fr[w]))
    if not best[0]:
        raise ValueError("no settlement wrap found in s2 snapshots")
    return best


def fig_money_event(path: Path, run_dir: Path, snaps: pl.DataFrame,
                    fday: str, t_set: int, q_cap: int | None = None) -> dict:
    """Stacked event-window view (P&L / inventory / skew) around t_set."""
    lo, hi = t_set - EVENT_WINDOW_S * 1000, t_set + EVENT_WINDOW_S * 1000
    win = snaps.filter((pl.col("funding_day") == fday)
                       & (pl.col("ts_ms") >= lo) & (pl.col("ts_ms") <= hi))
    sids = [s for s in (2, 5, 6) if s in win["strategy"].unique().to_list()]

    fig, (ax_p, ax_q, ax_s) = plt.subplots(
        3, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]})

    u_star_min = None
    for sid in sids:
        sub = win.filter(pl.col("strategy") == sid).sort("ts_ms")
        if not sub.height:
            continue
        x = (sub["ts_ms"].to_numpy() - t_set) / 60_000.0
        mtm = sub["mtm"].to_numpy()
        ax_p.plot(x, mtm - mtm[0], color=STRAT_COLORS.get(sid),
                  lw=1.8 if sid == 6 else 1.0,
                  label=STRATEGY_LABELS.get(sid, str(sid)))
        ax_q.plot(x, sub["inv"].to_numpy(), color=STRAT_COLORS.get(sid),
                  lw=1.8 if sid == 6 else 1.0)
        skew_bp = ((sub["delta_a"] - sub["delta_b"]) / 2.0
                   / sub["mid"] * 1e4).to_numpy()
        ax_s.plot(x, skew_bp, color=STRAT_COLORS.get(sid),
                  lw=1.8 if sid == 6 else 1.0)
        if sid == 6:  # regime-II boundary: first pre-settlement regime-2 snap
            r2 = sub.filter((pl.col("regime") == 2) & (pl.col("ts_ms") < t_set))
            if r2.height:
                u_star_min = float((r2["ts_ms"].min() - t_set) / 60_000.0)

    # s6 fill markers on the inventory panel
    fp = run_dir / "s6" / f"fills_{fday}.parquet"
    if fp.exists():
        fills = pl.read_parquet(fp).filter(
            (pl.col("ts_ms") >= lo) & (pl.col("ts_ms") <= hi))
        for side, mk, col in ((1, "^", "tab:green"), (-1, "v", "tab:red")):
            f = fills.filter(pl.col("side") == side)
            if f.height:
                ax_q.scatter((f["ts_ms"].to_numpy() - t_set) / 60_000.0,
                             f["inv_after"].to_numpy() / QUOTE_SIZE,  # lots
                             marker=mk, s=18,
                             color=col, zorder=5,
                             label=f"s6 {'buy' if side > 0 else 'sell'} fill")

    for ax in (ax_p, ax_q, ax_s):
        ax.axvline(0, color="k", lw=1.0)
        if u_star_min is not None:
            ax.axvline(u_star_min, color="tab:purple", lw=0.9, ls=":")
        ax.grid(alpha=0.3)
    if q_cap is not None:
        ax_q.axhspan(q_cap, q_cap * 1.15, color="0.85")
        ax_q.axhspan(-q_cap * 1.15, -q_cap, color="0.85")
        ax_q.set_ylim(-q_cap * 1.15, q_cap * 1.15)
    ax_p.set_ylabel("P&L since window start (USDT)")
    ax_p.set_title(
        f"Settlement event, {fday} (discrete fee at $u=0$"
        + (", regime switch at dotted line)" if u_star_min else ")"),
        fontsize=10)
    ax_p.legend(fontsize=8)
    ax_q.set_ylabel(r"inventory $q_t$ (lots)")
    if ax_q.get_legend_handles_labels()[0]:
        ax_q.legend(fontsize=7, loc="upper right")
    ax_s.set_ylabel("reservation skew (bp)")
    ax_s.set_xlabel("minutes to funding settlement")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"funding_day": fday, "t_settle_ms": int(t_set),
            "window_s": EVENT_WINDOW_S, "u_star_min": u_star_min}


def run(run_dir: Path, report_dir: Path, q_cap: int | None = None,
        quote_size: float = QUOTE_SIZE) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    figs = report_dir / "figures"
    figs.mkdir(exist_ok=True)

    panel = load_daily_panel(run_dir)
    days, daily = daily_pnl_matrix(panel)
    snaps = load_snaps(run_dir)
    snaps = snaps.with_columns((pl.col("inv") / quote_size).alias("inv"))

    inv_tab = inventory_by_u(snaps)
    skew_tab = skew_by_u(snaps)
    shp_tab = sharpe_by_u(snaps)
    fund_tab = settlement_funding(snaps)
    skewf_tab = skew_fund_by_u(snaps)
    inv_tab.write_parquet(report_dir / "money_inventory_by_u.parquet")
    skew_tab.write_parquet(report_dir / "money_skew_by_u.parquet")
    shp_tab.write_parquet(report_dir / "money_sharpe_by_u.parquet")
    fund_tab.write_parquet(report_dir / "money_funding_settlements.parquet")
    if skewf_tab.height:
        skewf_tab.write_parquet(report_dir / "money_skew_fund_by_u.parquet")

    event_tab = event_skew_pooled(snaps)
    if event_tab.height:
        event_tab.write_parquet(report_dir / "money_event_skew.parquet")
    fday, t_set = pick_event(run_dir, snaps)

    fig_money(figs / "fig_money.pdf", days, daily, inv_tab, skew_tab, shp_tab,
              fund_tab=fund_tab, skewf_tab=skewf_tab, event_tab=event_tab)

    manifest = fig_money_event(figs / "fig_money_event.pdf", run_dir, snaps,
                               fday, t_set, q_cap)
    (report_dir / "money_event_manifest.json").write_text(
        json.dumps(manifest, indent=2))

    return {"days": len(days), "strategies": sorted(daily.keys()),
            "snap_rows": snaps.height, "event": manifest}


def main() -> None:
    ap = argparse.ArgumentParser(description="Money figure.")
    ap.add_argument("--run-dir", type=Path, default=Path("runs/ablation"))
    ap.add_argument("--report-dir", type=Path, default=Path("reports/ablation"))
    ap.add_argument("--q-cap", type=int, default=10,
                    help="inventory bound drawn on the event figure")
    args = ap.parse_args()
    out = run(args.run_dir, args.report_dir, args.q_cap)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
