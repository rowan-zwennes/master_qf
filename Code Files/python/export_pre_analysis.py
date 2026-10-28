#!/usr/bin/env python3
"""Export the pre-analysis results to CSV and LaTeX."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import numpy as np


HORIZONS = [1, 5, 10, 30]
METRICS = ["drift_micro", "return_micro", "drift_mid", "return_mid"]

TARGET_METRIC = "drift_mid"

CLASS_STATIONARY = "S"
CLASS_DIFFERENCE = "D"
CLASS_NONSTAT    = "N"
CLASS_INSUFF     = "I"
CLASS_ORDER = (CLASS_STATIONARY, CLASS_DIFFERENCE, CLASS_NONSTAT, CLASS_INSUFF)
CLASS_DISPLAY = {
    CLASS_STATIONARY: "Stat.",
    CLASS_DIFFERENCE: "Mixed",   # ADF and KPSS disagree (not difference-stationary)
    CLASS_NONSTAT:    "Non.",
    CLASS_INSUFF:     "Insf.",
}

FEATURE_GROUPS: dict[str, list[str]] = {
    "Book shape": [
        "spread", "spread_bps",
        "imbalance_l1", "imbalance_l2_5", "imbalance_l6_10", "imbalance_l11_20",
        "bid_slope", "ask_slope", "slope_imbalance",
        "rel_bid_depth_l1", "rel_bid_depth_l2_5",
        "rel_bid_depth_l6_10", "rel_bid_depth_l11_20",
        "rel_ask_depth_l1", "rel_ask_depth_l2_5",
        "rel_ask_depth_l6_10", "rel_ask_depth_l11_20",
        "book_age_ms",
    ],
    "OFI": ["ofi_l1", "ofi_l2_5", "ofi_l6_10", "ofi_l11_20"],
    "Trade flow": [
        "n_trades_1s", "buy_vol_1s", "sell_vol_1s", "signed_vol_1s",
        "trade_imbalance_1s", "largest_trade_1s", "return_1s",
        "n_trades_5s", "buy_vol_5s", "sell_vol_5s", "signed_vol_5s",
        "trade_imbalance_5s", "largest_trade_5s", "return_5s",
        "n_trades_30s", "buy_vol_30s", "sell_vol_30s", "signed_vol_30s",
        "trade_imbalance_30s", "largest_trade_30s", "return_30s",
    ],
    "VPIN": [
        "vpin_micro", "vpin_window_micro_s",
        "vpin_base",  "vpin_window_base_s",
        "vpin_macro", "vpin_window_macro_s",
    ],
    "Price dynamics": [
        "micro_return_1s", "micro_return_5s", "micro_return_30s",
        "realized_vol_60s",
    ],
    "Funding context": ["seconds_to_funding", "basis_bps", "mark_stale_age_ms"],
}

PERIOD_PRE = "pre_analysis"
PERIOD_SIM = "sim"
PERIOD_ORDER = (PERIOD_PRE, PERIOD_SIM)
PERIOD_DISPLAY = {
    PERIOD_PRE: "Pre-analysis (days 1--30)",
    PERIOD_SIM: "Out-of-sample (days 31--94)",
}


def load_json(path: Path) -> list | dict:
    if not path.exists():
        print(f"  Warning: {path} not found.")
        return []
    return json.loads(path.read_text())


def load_splits(path: Path) -> dict[str, str]:
    """Return {funding_day: period} from splits.json."""
    if not path.exists():
        raise SystemExit(
            f"splits.json not found at {path}. Run `label_split.py "
            f"--build-splits` before exporting pre-analysis tables."
        )
    sp = json.loads(path.read_text())
    splits = sp.get("splits", {})
    if PERIOD_PRE not in splits or PERIOD_SIM not in splits:
        raise SystemExit(
            f"splits.json at {path} is missing the '{PERIOD_PRE}' / "
            f"'{PERIOD_SIM}' keys (got: {sorted(splits)})."
        )
    mapping: dict[str, str] = {}
    for d in splits[PERIOD_PRE]:
        mapping[d] = PERIOD_PRE
    for d in splits[PERIOD_SIM]:
        mapping[d] = PERIOD_SIM
    return mapping


def attach_period(df: pd.DataFrame, period_map: dict[str, str]) -> pd.DataFrame:
    """Add a `period` column to a per-funding-day DataFrame.

    Any day not listed in splits.json gets period = NaN; those rows are
    excluded from every period-aware aggregate so an unexpected day cannot
    silently pollute either panel.
    """
    if df.empty:
        return df
    out = df.copy()
    out["period"] = out["funding_day"].map(period_map)
    return out


def fmt_float(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{decimals}f}"


def fmt_sci(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if v == 0.0:
        return "$<10^{-300}$"
    exp = int(np.floor(np.log10(abs(v))))
    return f"$<10^{{{exp}}}$" if v < 0.001 else f"{v:.4f}"


def latex_texttt(name: str) -> str:
    """Wrap an identifier in `\texttt{}` with underscores escaped."""
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def fmt_p(v, *, capped_low: bool = False, capped_high: bool = False) -> str:
    """Format a p-value for thesis tables."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    if capped_low:
        return "$<0.01$"
    if capped_high:
        return "$>0.10$"
    if v == 0.0:
        return "$<10^{-300}$"
    if v < 1e-3:
        exp = int(np.floor(np.log10(v)))
        return f"$<10^{{{exp}}}$"
    return f"{v:.3f}"


def _safe_mean(s: pd.Series) -> float:
    """Mean ignoring NaN / None; returns NaN if the series is empty."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if not s.empty else float("nan")


def _safe_median(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.median()) if not s.empty else float("nan")


def load_battery(path: Path) -> list[dict]:
    """Load one of pre_analysis_{adf,kpss,arch}.json."""
    if not path.exists():
        print(f"  Warning: {path} not found, skipping its table.")
        return []
    return json.loads(path.read_text())


def load_target_distribution(
    validation_dir: Path, metric: str = TARGET_METRIC
) -> pd.DataFrame:
    """Aggregate per-day distribution moments for ONE label metric."""
    if not validation_dir.exists():
        print(f"  Warning: {validation_dir} not found, skipping target distribution table.")
        return pd.DataFrame()
    files = sorted(validation_dir.glob("funding_day_*.json"))
    if not files:
        print(f"  Warning: no funding_day_*.json under {validation_dir}, skipping target distribution.")
        return pd.DataFrame()
    acc: dict[int, dict[str, list[float]]] = {
        h: {"mean": [], "std": [], "skew": [], "kurt": [], "jb": [], "jb_stat": []}
        for h in HORIZONS
    }
    for f in files:
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        ld = ((d.get("stats") or {}).get("label_distributions")) or {}
        for h in HORIZONS:
            rec = (ld.get(str(h)) or {}).get(metric)
            if not isinstance(rec, dict) or "mean" not in rec:
                continue
            acc[h]["mean"].append(rec["mean"])
            acc[h]["std"].append(rec["std"])
            acc[h]["skew"].append(rec["skew"])
            acc[h]["kurt"].append(rec["kurtosis"])
            acc[h]["jb"].append(1.0 if rec.get("jarque_bera_rejects_normality") else 0.0)
            if rec.get("jarque_bera") is not None:
                acc[h]["jb_stat"].append(rec["jarque_bera"])
    rows = []
    for h in HORIZONS:
        a = acc[h]
        if not a["mean"]:
            continue
        rows.append({
            "horizon": h,
            "n_days": len(a["mean"]),
            "mean": float(np.mean(a["mean"])),
            "std": float(np.mean(a["std"])),
            "skew": float(np.mean(a["skew"])),
            "kurt": float(np.mean(a["kurt"])),
            "jb_reject_pct": 100.0 * float(np.mean(a["jb"])),
            "jb_stat_median": float(np.median(a["jb_stat"])) if a["jb_stat"] else float("nan"),
        })
    return pd.DataFrame(rows)


def index_battery_by_series(records: list[dict]) -> dict[str, list[dict]]:
    """Reshape [{funding_day, results: {series: {...}}}] into {series: [per_day_result, ...]}."""
    out: dict[str, list[dict]] = {}
    for rec in records:
        for series, res in (rec.get("results") or {}).items():
            if not isinstance(res, dict):
                continue
            out.setdefault(series, []).append(res)
    return out


def classify_one(adf_res: dict | None, kpss_res: dict | None) -> str | None:
    """Joint ADF + KPSS classification for ONE (series, day).

    Returns one of CLASS_STATIONARY / CLASS_DIFFERENCE / CLASS_NONSTAT /
    CLASS_INSUFF, or None if either test was skipped on this day
    (a missing data point is honest; a fabricated class is not).
    """
    if not adf_res or not kpss_res:
        return None
    if "stationary" not in adf_res or "stationary" not in kpss_res:
        return None
    adf_rejects = bool(adf_res["stationary"])     # ADF p<0.05 -> rejects unit-root null
    kpss_stat   = bool(kpss_res["stationary"])    # KPSS p>=0.05 -> fails to reject stationary null
    if adf_rejects and kpss_stat:
        return CLASS_STATIONARY
    if adf_rejects and not kpss_stat:
        return CLASS_DIFFERENCE
    if (not adf_rejects) and (not kpss_stat):
        return CLASS_NONSTAT
    return CLASS_INSUFF


def aggregate_stationarity(
    adf_by_series: dict[str, list[dict]],
    kpss_by_series: dict[str, list[dict]],
    series_list: list[str],
) -> pd.DataFrame:
    """Per-series aggregate of the joint ADF + KPSS battery."""
    rows = []
    for s in series_list:
        adf_list  = adf_by_series.get(s, [])
        kpss_list = kpss_by_series.get(s, [])
        classes: list[str] = []
        for a, k in zip(adf_list, kpss_list):
            c = classify_one(a, k)
            if c is not None:
                classes.append(c)
        counts = {k: classes.count(k) for k in CLASS_ORDER}
        n_tested = sum(counts.values())
        adf_ps  = [r["p"] for r in adf_list  if "p" in r]
        kpss_ps = [r["p"] for r in kpss_list if "p" in r]
        adf_stationary_days  = sum(1 for r in adf_list  if r.get("stationary") is True)
        kpss_stationary_days = sum(1 for r in kpss_list if r.get("stationary") is True)
        n_adf_total  = sum(1 for r in adf_list  if "stationary" in r)
        n_kpss_total = sum(1 for r in kpss_list if "stationary" in r)
        rows.append({
            "series": s,
            "n_tested": n_tested,
            "adf_median_p":  float(np.median(adf_ps))  if adf_ps  else float("nan"),
            "adf_reject_pct":  100.0 * adf_stationary_days  / n_adf_total  if n_adf_total  else float("nan"),
            "kpss_median_p": float(np.median(kpss_ps)) if kpss_ps else float("nan"),
            "kpss_reject_pct": 100.0 * (n_kpss_total - kpss_stationary_days) / n_kpss_total
                                if n_kpss_total else float("nan"),
            "n_S": counts[CLASS_STATIONARY],
            "n_D": counts[CLASS_DIFFERENCE],
            "n_N": counts[CLASS_NONSTAT],
            "n_I": counts[CLASS_INSUFF],
            "modal_class": max(CLASS_ORDER, key=lambda c: (counts[c], -CLASS_ORDER.index(c)))
                            if n_tested else "--",
        })
    return pd.DataFrame(rows)


def aggregate_arch(
    arch_by_series: dict[str, list[dict]],
    series_list: list[str],
) -> pd.DataFrame:
    """Per-series aggregate of the ARCH battery.

    Columns: series, n_tested, lm_median, lm_median_p, het_pct.
    A skipped day (constant / too few obs / error) contributes to neither
    the median nor the heteroskedastic count, only to `n_tested`'s
    denominator-via-omission story (see n_runs).
    """
    rows = []
    for s in series_list:
        records = arch_by_series.get(s, [])
        lm_stats = [r["lm_stat"] for r in records if "lm_stat" in r]
        lm_ps    = [r["lm_p"]    for r in records if "lm_p" in r]
        het_days = sum(1 for r in records if r.get("heteroskedastic") is True)
        n_runs   = sum(1 for r in records if "heteroskedastic" in r)
        rows.append({
            "series": s,
            "n_tested":    n_runs,
            "lm_median":   float(np.median(lm_stats)) if lm_stats else float("nan"),
            "lm_median_p": float(np.median(lm_ps))    if lm_ps    else float("nan"),
            "het_pct":     100.0 * het_days / n_runs   if n_runs   else float("nan"),
        })
    return pd.DataFrame(rows)


def _label_series_list() -> list[str]:
    """The 16 label series in display order: metric-major, horizon-minor."""
    return [f"{m}_{h}" for m in METRICS for h in HORIZONS]


def _feature_series_with_groups() -> list[tuple[str, str]]:
    """Yield (group_label, feature_name) pairs in display order."""
    out: list[tuple[str, str]] = []
    for group, feats in FEATURE_GROUPS.items():
        for f in feats:
            out.append((group, f))
    return out


def to_latex(df: pd.DataFrame, caption: str, label: str, col_fmt: str, placement: str = "H") -> str:
    lines = [
        f"\\begin{{table}}[{placement}]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\hline",
        " & ".join(df.columns) + " \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def build_target_distribution_latex(df_dist: pd.DataFrame) -> str:
    """Distribution of the adopted mid-price drift target, one row per horizon."""
    if df_dist.empty:
        return ("% (per-day validation label_distributions absent, "
                "run validate_book.py before exporting the target distribution table)")

    def _fmt_jb(v: float) -> str:
        """Render a large Jarque-Bera statistic as a tidy power of ten."""
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "--"
        exp = int(np.floor(np.log10(v)))
        mant = v / (10 ** exp)
        return f"${mant:.1f}\\times 10^{{{exp}}}$"

    n_days = int(df_dist["n_days"].iloc[0])
    rows = []
    for _, r in df_dist.iterrows():
        rows.append({
            "$h$ (s)": int(r["horizon"]),
            "Mean (USDT/s)": f"{r['mean']:.4f}",
            "Std (USDT/s)": f"{r['std']:.3f}",
            "Skewness": f"{r['skew']:.3f}",
            "Excess kurtosis": f"{r['kurt']:.1f}",
            "JB stat (med.)": _fmt_jb(r["jb_stat_median"]),
        })
    return to_latex(
        pd.DataFrame(rows),
        caption=(
            "Distribution of the adopted learning target, the mid-price drift "
            "$\\operatorname{drift}^{\\mathrm{mid}}_h(t) = (S_{t+h} - S_t)/h$ in "
            "price units per second, the direct $\\alpha_{ML}$ target, at each "
            f"label horizon. Each statistic is the mean of the {n_days} "
            "per-funding-day values (the Jarque-Bera column is the daily "
            "median). The excess kurtosis far exceeds the Gaussian value of "
            "zero at every horizon and the Jarque-Bera test rejects normality "
            f"on all {n_days} days, confirming a heavy-tailed target. The "
            "target is also stationary (joint ADF $+$ KPSS class S, i.e.\\ ADF "
            "rejects the unit root and KPSS does not reject stationarity, on "
            "every day at every horizon) and conditionally heteroskedastic "
            "(Engle ARCH-LM rejects homoskedasticity on every day); both "
            "verdicts are identical across horizons and so are stated here "
            "rather than tabulated. It is the conditional heteroskedasticity, "
            "not the marginal non-Gaussianity, that motivates the non-linear "
            "function class for $\\alpha_{ML}$."
        ),
        label="tab:target_distribution",
        col_fmt="rrrrrr",
        placement="ht",
    )


def build_adf_arch_latex(df_stat: pd.DataFrame) -> str:
    """One row per horizon for the adopted drift_mid target."""
    rows = []
    for metric in [TARGET_METRIC]:
        for h in HORIZONS:
            col_adf = f"adf_stat_{metric}_{h}s"
            col_s   = f"stationary_{metric}_{h}s"
            col_arc = f"arch_stat_{metric}_{h}s"
            col_cl  = f"clustering_{metric}_{h}s"

            if col_adf not in df_stat.columns:
                continue

            tested  = int(df_stat[col_adf].notna().sum())
            n_stat  = int(df_stat[col_s].sum())
            n_arch  = int(df_stat[col_cl].sum())
            all_s   = n_stat == tested
            all_cl  = n_arch == tested
            med_adf = df_stat[col_adf].median()
            med_arc = df_stat[col_arc].median()

            rows.append({
                "Horizon": f"{h}\\,s",
                "Days": tested,
                "All stationary": "Yes" if all_s else f"No ({n_stat})",
                "Median ADF": fmt_float(med_adf, 2),
                "All ARCH": "Yes" if all_cl else f"No ({n_arch})",
                "Median LM": fmt_float(med_arc, 0),
            })

    tbl = pd.DataFrame(rows)
    return to_latex(
        tbl,
        caption=(
            "ADF stationarity and Engle ARCH volatility-clustering tests on the "
            "adopted mid-price drift target across all 94 funding days "
            "(full-corpus diagnostic, extending the 30-day pre-analysis battery "
            "to the held-out period). "
            "ADF $H_0$: unit root; ARCH $H_0$: homoskedastic residuals. "
            "All $p$-values $< 10^{-10}$ (not shown individually). "
            "The target is stationary and volatility-clustered on every day at "
            "every horizon; the three other metric families "
            "(\\texttt{return\\_mid}, \\texttt{drift\\_micro}, "
            "\\texttt{return\\_micro}) behave identically and are omitted."
        ),
        label="tab:adf_arch_summary",
        col_fmt="lrccrc",
    )


def build_vif_latex() -> str:
    """Table of the features with VIF $> 10$ on a representative day."""
    high = _compute_vif_high()
    if high is None or len(high) == 0:
        return "% (VIF computation unavailable, see _compute_vif_high)"
    inf_feats = [f for f, v in high.items() if math.isinf(v)]
    finite = [(f, v) for f, v in high.items() if not math.isinf(v)]

    rows = [{"Feature": latex_texttt(f), "VIF": f"{v:.1f}"} for f, v in finite]
    inf_label = None
    if inf_feats:
        inf_label = (
            f"{len(inf_feats)} exactly-collinear flow columns "
            "(\\texttt{\\{buy,sell,signed\\}\\_vol\\_\\{1,5,30\\}s})"
        )
        rows.append({"Feature": inf_label, "VIF": "$\\infty$"})

    latex = to_latex(
        pd.DataFrame(rows),
        caption=(
            "Features with a Variance Inflation Factor above the conventional "
            "threshold of 10, computed on the full feature matrix on "
            "2026-02-15 (all 56 features tested). Collinearity is localised: "
            "the nine \\texttt{\\{buy,sell,signed\\}\\_vol} columns are exactly "
            "collinear by construction (\\texttt{signed\\_vol} $=$ "
            "\\texttt{buy\\_vol} $-$ \\texttt{sell\\_vol}, so each trade-window "
            "triple is rank 2) and have infinite VIF, the two spread terms are "
            "large but finite, and every remaining feature is below about 22. "
            "The gradient-boosted model is invariant to this; the Ridge "
            "baseline drops only the three \\texttt{signed\\_vol\\_*} columns, "
            "which suffices to break the exact dependency, and retains the rest "
            "under its $L_2$ penalty."
        ),
        label="tab:vif_summary",
        col_fmt="lr",
    )
    # Separator rule before the collapsed infinite-VIF row.
    if inf_label:
        latex = latex.replace(f"\n{inf_label}", f"\n\\hline\n{inf_label}")
    return latex


def build_data_quality_latex(df_val: pd.DataFrame) -> str:
    """Compact data-quality table over a small set of representative sample days."""
    sample_days = [
        "2026-01-19", "2026-01-25", "2026-02-15",   # pre-analysis
        "2026-03-26", "2026-04-22",                 # OOS
    ]
    rows = []
    for d in sample_days:
        r = df_val[df_val["funding_day"] == d]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append({
            "Date": _fmt_date(d),
            "Per.": _PERIOD_SHORT.get(r["period"], "--") if pd.notna(r["period"]) else "--",
            "Rows": f"{int(r['n_rows']):,}",
            "\\% invalid": fmt_float(r["pct_invalid"], 4),
            "\\% stale": fmt_float(r["pct_stale"], 4),
        })

    agg_rows = []
    for period in PERIOD_ORDER:
        sub = df_val[df_val["period"] == period]
        if sub.empty:
            continue
        agg_rows.append({
            "Date": f"\\textit{{All {len(sub)} days}}",
            "Per.": _PERIOD_SHORT[period],
            "Rows": f"{int(sub['n_rows'].sum()):,}",
            "\\% invalid": fmt_float(_safe_mean(sub["pct_invalid"]), 4),
            "\\% stale": fmt_float(_safe_mean(sub["pct_stale"]), 4),
        })

    tbl = pd.DataFrame(rows + agg_rows)
    latex = to_latex(
        tbl,
        caption=(
            "Reconstructed-book data quality per study period, validated by "
            "\\texttt{validate\\_book.py} "
            "(Pre\\,=\\,pre-analysis days 1--30; OOS\\,=\\,out-of-sample days 31--94). "
            "\\textit{\\% invalid:} fraction of recording seconds inside a "
            "feed-gap interval, where the 100\\,ms book tick is flagged "
            "\\texttt{valid\\,=\\,False}. "
            "\\textit{\\% stale:} fraction of seconds where the mark price has "
            "not been refreshed by the exchange within the preceding 5 seconds "
            "(see Section~\\ref{sec:data_gap_handling}). "
            "Period-aggregate rows show arithmetic means."
        ),
        label="tab:data_quality",
        col_fmt="llrrr",
    )
    # Separator rule before aggregate rows
    first_agg = agg_rows[0]["Date"] if agg_rows else None
    if first_agg:
        latex = latex.replace(f"\n{first_agg}", f"\n\\hline\n{first_agg}")
    return latex


_MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

def _fmt_date(iso: str) -> str:
    parts = iso.split("-")
    return f"{_MONTH_ABBR[int(parts[1]) - 1]} {int(parts[2])}"

_PERIOD_SHORT = {
    PERIOD_PRE: "Pre",
    PERIOD_SIM: "OOS",
}


def build_raw_feed_quality_latex(df_raw: pd.DataFrame) -> str:
    """Raw WebSocket feed quality table for Appendix A (Table tab:raw_feed_quality)."""
    PARTIAL_DAY = "2026-04-23"

    df = df_raw.copy()
    df["gap_s"] = pd.to_numeric(df["Diffs_TotalGapTime_ms"], errors="coerce").fillna(0) / 1000.0
    df["book_cap"] = pd.to_numeric(df["Book_Capture_pct"], errors="coerce")
    df["diffs_q"] = pd.to_numeric(df["Diffs_Quality_pct"], errors="coerce")
    df["mark_rows"] = pd.to_numeric(df["Mark_Rows"], errors="coerce").fillna(0)
    df["mark_cov"] = df.apply(
        lambda r: r["mark_rows"] / (18000.0 if r["funding_day"] == PARTIAL_DAY else 86400.0) * 100.0,
        axis=1,
    )

    gap_days  = set(df.loc[df["gap_s"] > 0, "funding_day"])
    low_cap   = set(df.loc[df["book_cap"] < 85.0, "funding_day"])
    notable   = gap_days | low_cap

    # One representative clean day per period (not already notable)
    clean_picks = {
        PERIOD_PRE: ["2026-02-15", "2026-01-30", "2026-01-26"],
        PERIOD_SIM: ["2026-04-07", "2026-03-10", "2026-04-01"],
    }
    sample_clean: set[str] = set()
    for candidates in clean_picks.values():
        for d in candidates:
            if d not in notable and d in df["funding_day"].values:
                sample_clean.add(d)
                break

    show_days = sorted(notable | sample_clean)

    rows = []
    for d in show_days:
        mask = df["funding_day"] == d
        if not mask.any():
            continue
        r = df.loc[mask].iloc[0]
        period_val = r["period"]
        period_str = _PERIOD_SHORT.get(period_val, "--") if pd.notna(period_val) else "--"
        rows.append({
            "Date": _fmt_date(d),
            "Per.": period_str,
            "Diffs (\\%)": fmt_float(r["diffs_q"], 4),
            "Gap (s)": fmt_float(r["gap_s"], 1),
            "Book cap.\\ (\\%)": fmt_float(r["book_cap"], 2),
            "Mark cov.\\ (\\%)": fmt_float(r["mark_cov"], 2),
        })

    agg_rows = []
    for period in PERIOD_ORDER:
        sub = df[(df["period"] == period) & (df["funding_day"] != PARTIAL_DAY)]
        if sub.empty:
            continue
        n = len(sub)
        agg_rows.append({
            "Date": f"\\textit{{All {n} days}}",
            "Per.": _PERIOD_SHORT[period],
            "Diffs (\\%)": fmt_float(_safe_mean(sub["diffs_q"]), 4),
            "Gap (s)": fmt_float(sub["gap_s"].sum(), 1) + "$^\\dag$",
            "Book cap.\\ (\\%)": fmt_float(_safe_mean(sub["book_cap"]), 2),
            "Mark cov.\\ (\\%)": fmt_float(_safe_mean(sub["mark_cov"]), 2),
        })

    tbl = pd.DataFrame(rows + agg_rows)
    latex = to_latex(
        tbl,
        caption=(
            "Raw WebSocket feed quality over the 94-day recording window "
            "(00:00 UTC on 19 January 2026 to 04:00 UTC on 23 April 2026). "
            "Individual rows are shown for every day with a non-zero L2 diff gap, "
            "every day with a book-snapshot capture rate below 85\\,\\%, "
            "and one representative clean day per study period (Pre\\,=\\,pre-analysis "
            "days 1--30; OOS\\,=\\,out-of-sample days 31--94). "
            "Aggregate rows span all days in each period; percentage columns "
            "aggregate as the unweighted mean of daily values, Gap (s) as the "
            "period sum ($^\\dag$). "
            "\\textit{Diffs (\\%):} fraction of L2 update identifiers received "
            "without sequence gaps. "
            "\\textit{Gap (s):} total L2 feed downtime in seconds "
            "(individual days) or $^\\dag$period total (aggregate rows); "
            "the two-period sum of 345\\,s equals the figure cited in "
            "Section~\\ref{sec:data_gap_handling}. "
            "\\textit{Book cap.\\ (\\%):} fraction of the theoretical $100$\\,ms "
            "book snapshots delivered by Binance; shortfalls reflect the "
            "exchange's periodic snapshot scheduling, not irrecoverable data loss. "
            "\\textit{Mark cov.\\ (\\%):} mark-price rows as a fraction of "
            "expected seconds in the day (86\\,400 for full days). "
            "Trade update-identifier integrity was 100.00\\,\\% on all 94 days "
            "and is omitted."
        ),
        label="tab:raw_feed_quality",
        col_fmt="llrrrr",
    )
    # Insert a separator rule before the aggregate rows
    first_agg_date = agg_rows[0]["Date"] if agg_rows else None
    if first_agg_date:
        latex = latex.replace(
            f"\n{first_agg_date}",
            f"\n\\hline\n{first_agg_date}",
        )
    return latex


def _stationarity_row(s: dict, *, latex_series: str) -> dict:
    """Common formatting for one row of a stationarity LaTeX table.

    Used by both the labels and the features tables so the per-series cell
    formatting (median p, % reject, S/D/N/I counts, modal class) is
    identical across the two tables.
    """
    def _pct(v: float) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "--"
        return f"{v:.0f}\\%"
    n = int(s["n_tested"]) if not math.isnan(s["n_tested"]) else 0
    counts = f"{int(s['n_S'])}/{int(s['n_D'])}/{int(s['n_N'])}/{int(s['n_I'])}"
    modal = CLASS_DISPLAY.get(s["modal_class"], s["modal_class"])
    return {
        "Series": latex_series,
        "$n$": n,
        "ADF med.\\,$p$": fmt_p(s["adf_median_p"]),
        "ADF rej.\\,\\%": _pct(s["adf_reject_pct"]),
        "KPSS med.\\,$p$": fmt_p(s["kpss_median_p"]),
        "KPSS rej.\\,\\%": _pct(s["kpss_reject_pct"]),
        "S/D/N/I": counts,
        "Modal": modal,
    }


def build_labels_stationarity_latex(df_agg: pd.DataFrame) -> str:
    by_name = {r["series"]: r for _, r in df_agg.iterrows()}
    rows = []
    for metric in [TARGET_METRIC]:
        for h in HORIZONS:
            s = by_name.get(f"{metric}_{h}")
            if s is None:
                continue
            latex_name = latex_texttt(f"{metric}_{h}")
            rows.append(_stationarity_row(s, latex_series=latex_name))
    if not rows:
        return "% (pre_analysis_adf.json / pre_analysis_kpss.json absent)"
    return to_latex(
        pd.DataFrame(rows),
        caption=(
            "Joint Augmented Dickey-Fuller and KPSS stationarity "
            "classification of the adopted mid-price drift target at all four "
            "label horizons, per funding day, on the 30 pre-analysis days. "
            "ADF $H_0$: unit root; KPSS $H_0$: stationary. Median $p$ is "
            "taken across the 30 daily tests; rej.\\,\\% is the fraction "
            "of days the test rejects its null. The S/D/N/I column counts "
            "(stationary, difference-stationary, non-stationary, "
            "insufficient) the day-by-day joint classification; Modal is "
            "the most common class across the 30 days. The target is class S "
            "(stationary in both senses) at every horizon."
        ),
        label="tab:preanalysis_labels_stationarity",
        col_fmt="lrrrrrrl",
    )


def build_features_stationarity_latex(df_agg: pd.DataFrame) -> str:
    """Joint ADF + KPSS classification for every feature, first 30 days only."""
    by_name = {r["series"]: r for _, r in df_agg.iterrows()}
    grouped = _feature_series_with_groups()
    seen = {f for _, f in grouped}
    other = [f for f in by_name if f not in seen and f not in _label_series_list()]
    if other:
        grouped += [("Other", f) for f in sorted(other)]

    def _pct(v: float) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "--"
        return f"{v:.0f}\\%"

    header_cells = [
        "Group", "Series", "ADF rej.\\,\\%", "KPSS rej.\\,\\%", "S/D/N", "Modal",
    ]
    header = " & ".join(header_cells) + " \\\\"
    caption = (
        "Joint ADF $+$ KPSS stationarity classification of every model feature "
        "in \\texttt{FEATURE\\_COLS}, per funding day, on the 30 pre-analysis "
        "days; this table also serves as the complete enumeration of the "
        "feature set. Every series was tested on all $n=30$ days. ADF $H_0$: "
        "unit root; KPSS $H_0$: stationary. ADF rej.\\,\\% and KPSS rej.\\,\\% "
        "are the fraction of days each test rejects its null. The S/D/N column "
        "counts the day-by-day joint classification: S, both tests indicate "
        "stationarity; D (Mixed), the tests disagree, ADF finding no unit root "
        "while KPSS rejects stationarity; N (Non.), both indicate a unit root. "
        "The fourth, insufficient-evidence, class is empty throughout and "
        "omitted. Modal is the most common class across the 30 days. The four "
        "VPIN scale and timing series "
        "(\\texttt{vpin\\_window\\_\\{micro,base,macro\\}\\_s} and "
        "\\texttt{vpin\\_macro}) are first-differenced, three falling in class N "
        "(a unit root by both tests) and the fast-bucket fill time borderline; "
        "all other series, where ADF rejects the unit root, are left "
        "untransformed (see Section~\\ref{sec:data_stat_char})."
    )

    lines = [
        "{\\small",
        "\\begin{longtable}{llrrcl}",
        f"\\caption{{{caption}}}",
        "\\label{tab:preanalysis_features_stationarity}\\\\",
        "\\hline",
        header,
        "\\hline",
        "\\endfirsthead",
        "\\hline",
        header,
        "\\hline",
        "\\endhead",
        "\\hline",
        "\\endfoot",
    ]

    last_group: str | None = None
    for group, feat in grouped:
        s = by_name.get(feat)
        if s is None:
            continue
        if last_group is not None and group != last_group:
            lines.append("\\hline")
        counts = f"{int(s['n_S'])}/{int(s['n_D'])}/{int(s['n_N'])}"
        modal = CLASS_DISPLAY.get(s["modal_class"], s["modal_class"])
        cells = [
            group if group != last_group else "",
            latex_texttt(feat),
            _pct(s["adf_reject_pct"]),
            _pct(s["kpss_reject_pct"]),
            counts,
            modal,
        ]
        lines.append(" & ".join(cells) + " \\\\")
        last_group = group

    if last_group is None:
        return "% (pre_analysis_adf.json / pre_analysis_kpss.json absent)"
    lines.append("\\end{longtable}")
    lines.append("}")
    return "\n".join(lines)


def build_purge_window_latex(purge: dict) -> str:
    """Operational summary of the empirical purge window."""
    if not purge:
        return "% (purge_window.json absent, run compute_purge_window.py)"

    per_feat = purge.get("per_feature_tau") or {}
    feat_rows = []
    for feat, info in per_feat.items():
        if not isinstance(info, dict):
            continue
        tau = info.get("tau_seconds")
        tau_str = f"{tau}" if tau is not None else "$> \\text{max\\_lag}$"
        feat_rows.append({
            "Feature": feat.replace("_", "\\_"),
            "$\\tau$ (s)": tau_str,
            "median ACF @ 60s":  fmt_float(info.get("median_acf_at_60s"), 3),
            "median ACF @ 300s": fmt_float(info.get("median_acf_at_300s"), 3),
        })

    diag = purge.get("diagnostic_per_day_max_distribution") or {}
    driver = purge.get("tau_driver_feature", "--")
    driver_val = purge.get("tau_driver_value_sec", "--")

    summary_rows = [
        {"Quantity": "$\\tau_{\\text{purge}}$ (s, operational)",
         "Value": f"\\textbf{{{purge.get('tau_purge_sec', '--')}}}"},
        {"Quantity": "Driver feature (slowest)", "Value": str(driver).replace("_", "\\_")},
        {"Quantity": "Driver $\\tau$ (s)",       "Value": f"{driver_val}"},
        {"Quantity": "$n$ days used",            "Value": f"{purge.get('n_days', '--')}"},
        {"Quantity": "$n$ features",             "Value": f"{purge.get('n_features', '--')}"},
        {"Quantity": "ACF max lag (s)",          "Value": f"{purge.get('max_lag_sec', '--')}"},
        {"Quantity": "Confirmation window (s)",  "Value": f"{purge.get('confirm_lags_sec', '--')}"},
        {"Quantity": "White-noise CI $z$",       "Value": fmt_float(purge.get("ci_z"), 2)},
        {"Quantity": "diagnostic median per-day-max (s)",  "Value": fmt_float(diag.get("median_sec"), 0)},
        {"Quantity": "diagnostic p75 per-day-max (s)",     "Value": fmt_float(diag.get("p75_sec"), 0)},
        {"Quantity": "diagnostic p90 per-day-max (s)",     "Value": fmt_float(diag.get("p90_sec"), 0)},
        {"Quantity": "diagnostic p95 per-day-max (s)",     "Value": fmt_float(diag.get("p95_sec"), 0)},
    ]

    feat_tab = to_latex(
        pd.DataFrame(feat_rows),
        caption=(
            "Per-feature decay timescales from the median ACF across "
            f"{purge.get('n_days', '?')} pre-analysis funding days. For "
            "each microstructure feature (level-1 to level-11–20 OFI and OBI, "
            "trade imbalance, signed volume, micro return), the daily ACFs "
            "are stacked and the median curve is taken across days; "
            "$\\tau$ is the smallest lag at which $|\\widetilde{\\mathrm{ACF}}(k)| "
            "< z/\\sqrt{n_{\\text{median}}}$ holds for the next "
            "$\\text{confirm\\_lags}$ consecutive lags. The slowest-decaying "
            "feature drives the operational purge window. Regime-indicator "
            "features (realized volatility, VPIN scales) are excluded because "
            "their slow ACF reflects regime persistence, not seam-specific "
            "contamination. Lopez de Prado (2018), ch. 7."
        ),
        label="tab:purge_window_per_feature",
        col_fmt="lrrr",
    )

    summary_tab = to_latex(
        pd.DataFrame(summary_rows),
        caption=(
            "Operational summary of the median-ACF purge window. "
            "$\\tau_{\\text{purge}}$ is the maximum per-feature $\\tau$ "
            "rounded up to the nearest minute. The diagnostic per-day-max "
            "distribution is reported for comparison: it would be the "
            "aggregation base of an alternative per-day-max method, but is "
            "dominated by 1–3 anomalous days at $n_{\\text{days}}=30$ and "
            "is not used to set $\\tau_{\\text{purge}}$."
        ),
        label="tab:purge_window",
        col_fmt="lr",
    )

    return feat_tab + "\n\n" + summary_tab


def build_purge_window_methodology(purge: dict) -> str:
    """Prose paragraph to cite the empirical purge window in the
    methodology section. Not a LaTeX table, a single \\paragraph block
    the user pastes verbatim."""
    if not purge:
        return ""
    tp = purge.get("tau_purge_sec", "??")
    driver = purge.get("tau_driver_feature", "??")
    driver_val = purge.get("tau_driver_value_sec", "??")
    n = purge.get("n_days", "??")
    nf = purge.get("n_features", "??")
    cl = purge.get("confirm_lags_sec", "??")
    return (
        "\\paragraph{Empirical purge window (median-ACF).}\n"
        f"To prevent leakage at the walk-forward train/test seam, we drop "
        f"the last $\\tau_{{\\text{{purge}}}} = {tp}$ seconds of every "
        f"training window before the held-out day. The window size from which "
        f"$\\tau$ is estimated is the full 30-day rolling training window, "
        f"not a short local window. This is justified by the stationarity "
        f"tests reported in Table~\\ref{{tab:stationarity_joint}}: the "
        f"event-indicator microstructure features used here are class S or "
        f"D, so the autocorrelation function is time-invariant in expectation, "
        f"and statistical efficiency dictates using as much data as is "
        f"available. Within the 30-day window, we stack the daily ACFs of "
        f"each event-indicator feature across the {n} days, take the median "
        f"ACF across days for each feature (robust to anomalous days), and "
        f"locate the smallest lag $\\tau$ at which "
        f"$|\\widetilde{{\\mathrm{{ACF}}}}(k)| < z/\\sqrt{{n_{{\\text{{median}}}}}}$ "
        f"holds for {cl} consecutive lags. The operational constant is the "
        f"maximum across the {nf} features (the slowest-decaying feature "
        f"drives the simultaneous purge); here {str(driver).replace('_', '\\_')} "
        f"= {driver_val}\\,s. Regime-indicator features (realized volatility, "
        f"VPIN scales) are excluded because their slow ACF reflects volatility "
        f"clustering and informed-flow regime persistence, not seam-specific "
        f"leakage; a purge window cannot fix regime persistence. In the "
        f"production protocol, $\\tau$ is recomputed per fold on each "
        f"30-day rolling training window, under stationarity this is "
        f"statistically equivalent to the fixed pre-analysis $\\tau$, but it "
        f"aligns with the walk-forward principle that every model-relevant "
        f"parameter is recalibrated on the training window and is robust to "
        f"subtle non-stationarity. See "
        f"Table~\\ref{{tab:purge_window_per_feature}} for the per-feature "
        f"$\\tau$ and Lopez de Prado (2018, ch. 7) for the broader methodology.\n"
    )


VIF_DAY = "2026-02-15"


def _compute_vif_high():
    """Per-feature VIF on `VIF_DAY`, restricted to VIF $>10$, sorted desc.

    Returns a pandas Series (feature -> VIF) or None on import/data failure.
    Shared by `build_vif_latex` (the thesis table) and `_print_vif_detail`
    (the console roll-up) so both report identical numbers from one fit.
    """
    import os
    import warnings
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    warnings.filterwarnings("ignore")

    try:
        from pipeline_utils import load_labels_for_funding_day, select_training_rows
        from statistical_tests import vif_check
        from ml_train import FEATURE_COLS
    except ImportError as e:
        print(f"  (VIF skipped, import error: {e})")
        return None

    base = Path("/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt")
    raw = load_labels_for_funding_day(base, VIF_DAY)
    df = select_training_rows(raw, 10)
    book_cols = [c for c in FEATURE_COLS if c in df.columns]
    vif = vif_check(df.select(book_cols).to_pandas())
    return vif[vif > 10].sort_values(ascending=False)


def _print_vif_detail() -> None:
    """Print the per-feature VIF roll-up. Not exported, purely for reference."""
    high = _compute_vif_high()
    if high is None:
        return
    print(f"  {'Feature':<35} {'VIF':>10}")
    print(f"  {'-'*35} {'-'*10}")
    for feat, val in high.items():
        val_str = "inf" if val == float("inf") else f"{val:>10.1f}"
        print(f"  {feat:<35} {val_str:>10}")
    print(f"\n  VIF > 10: {len(high)},  "
          f"infinite: {sum(v == float('inf') for v in high)}")
    print(f"\n  Ridge baseline: drop signed_vol_* (3, exact inf-VIF collinearity) "
          f"before fitting; spread/spread_bps (high-but-finite VIF) are retained "
          f"(see ml_baseline_ridge.RIDGE_DROP_FEATURES).")


def _print_pass_rates_by_period(df_stat: pd.DataFrame) -> None:
    """Per-period pass rates for the full-corpus ADF / ARCH diagnostic."""
    if "period" not in df_stat.columns:
        print("  (period column missing, falling back to all-days summary)")
        groups = [("all 94 days", df_stat)]
    else:
        groups = [
            (PERIOD_DISPLAY[p], df_stat[df_stat["period"] == p])
            for p in PERIOD_ORDER
        ]

    for label, sub in groups:
        if sub.empty:
            continue
        print(f"\n  {label}:")
        for metric in METRICS:
            print(f"    {metric}:")
            for h in HORIZONS:
                col_s  = f"stationary_{metric}_{h}s"
                col_cl = f"clustering_{metric}_{h}s"
                if col_s not in sub.columns:
                    continue
                n  = int(sub[col_s].notna().sum())
                ns = int(sub[col_s].sum())
                na = int(sub[col_cl].sum())
                print(f"      {h:2d}s  stationary {ns}/{n}   ARCH {na}/{n}")


def _safe_filename(label: str) -> str:
    """Sanitise a LaTeX `\\label{...}` value into a safe filename stem.

    `tab:preanalysis_labels_stationarity` -> `tab_preanalysis_labels_stationarity`.
    Only `:` and `/` are translated; everything else in LaTeX label syntax
    is already filename-safe.
    """
    return label.replace(":", "_").replace("/", "_")


def _emit_table(
    label: str,
    latex_text: str,
    *,
    reports_dir: Path,
    accumulator: list[tuple[str, str]],
    print_to_stdout: bool = True,
) -> None:
    """Triple-purpose emit: print to stdout, write `reports/<label>.tex`,
    append `(label, latex_text)` to `accumulator` for the eventual
    concatenated `all_tables.tex`. The stdout copy preserves the existing
    interactive workflow; the file copies are what the user comes back to
    in the morning after an overnight run.
    """
    if print_to_stdout:
        sep = "\n" + "=" * 70 + "\n"
        print(f"{sep}LaTeX: {label}")
        print(latex_text)
    (reports_dir / f"{_safe_filename(label)}.tex").write_text(latex_text + "\n")
    accumulator.append((label, latex_text))


def _write_all_tables(accumulator: list[tuple[str, str]], reports_dir: Path) -> None:
    """Concatenate every emitted table into a single `all_tables.tex` so
    the user can `\\input{all_tables}` in the thesis source for a one-shot
    paste of the whole appendix."""
    if not accumulator:
        return
    chunks = []
    for label, txt in accumulator:
        chunks.append(f"% ---- {label} ----\n{txt}\n")
    (reports_dir / "all_tables.tex").write_text("\n".join(chunks))


def _write_per_series_csvs(
    df_stationarity: pd.DataFrame,
    df_arch_agg: pd.DataFrame,
    purge: dict,
    *,
    reports_dir: Path,
) -> list[Path]:
    """Per-series CSV roll-ups for sortable/filterable review in Excel."""
    written: list[Path] = []
    if not df_stationarity.empty:
        p = reports_dir / "stationarity_joint.csv"
        df_stationarity.to_csv(p, index=False)
        written.append(p)
    if not df_arch_agg.empty:
        p = reports_dir / "arch_summary.csv"
        df_arch_agg.to_csv(p, index=False)
        written.append(p)
    if purge:
        # Summary CSV: one row, the operational stats (median-ACF method).
        diag = purge.get("diagnostic_per_day_max_distribution") or {}
        summary_row = {
            "n_days":                  purge.get("n_days"),
            "n_features":              purge.get("n_features"),
            "n_skipped":               purge.get("n_skipped"),
            "tau_purge_sec":           purge.get("tau_purge_sec"),
            "tau_aggregation_method":  purge.get("tau_aggregation_method"),
            "tau_driver_feature":      purge.get("tau_driver_feature"),
            "tau_driver_value_sec":    purge.get("tau_driver_value_sec"),
            "diag_per_day_max_median_sec": diag.get("median_sec"),
            "diag_per_day_max_p75_sec":    diag.get("p75_sec"),
            "diag_per_day_max_p90_sec":    diag.get("p90_sec"),
            "diag_per_day_max_p95_sec":    diag.get("p95_sec"),
            "diag_per_day_max_max_sec":    diag.get("max_sec"),
            "ci_z":                    purge.get("ci_z"),
            "max_lag_sec":             purge.get("max_lag_sec"),
            "confirm_lags_sec":        purge.get("confirm_lags_sec"),
            "rounding_step_sec":       purge.get("rounding_step_sec"),
        }
        p = reports_dir / "purge_window_summary.csv"
        pd.DataFrame([summary_row]).to_csv(p, index=False)
        written.append(p)
        # Per-feature τ CSV (one row per feature, the median-ACF values).
        per_feat = purge.get("per_feature_tau") or {}
        feat_rows: list[dict] = []
        for feat, info in per_feat.items():
            if not isinstance(info, dict):
                continue
            feat_rows.append({"feature": feat, **{k: v for k, v in info.items()}})
        if feat_rows:
            p = reports_dir / "purge_window_per_feature_tau.csv"
            pd.DataFrame(feat_rows).to_csv(p, index=False)
            written.append(p)
        long_rows: list[dict] = []
        for rec in purge.get("per_day", []):
            day = rec.get("funding_day")
            for feat, info in (rec.get("features") or {}).items():
                row = {"funding_day": day, "feature": feat}
                row.update({k: v for k, v in info.items() if k != "_acf_array"})
                long_rows.append(row)
        if long_rows:
            p = reports_dir / "purge_window_per_feature.csv"
            pd.DataFrame(long_rows).to_csv(p, index=False)
            written.append(p)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pre-analysis results to CSV and LaTeX.")
    base_default = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
    parser.add_argument(
        "--raw-feed-report",
        default=str(Path(__file__).parent / "daily_validation_report.csv"),
        help="Per-day raw WebSocket feed quality CSV (daily_validation_report.csv).",
    )
    parser.add_argument(
        "--validation-summary",
        default=f"{base_default}/validation/summary.json",
    )
    parser.add_argument(
        "--validation-dir",
        default=f"{base_default}/validation",
        help="Directory of per-day funding_day_*.json (validate_book.py); "
             "source of the target drift distribution table.",
    )
    parser.add_argument(
        "--stat-tests-summary",
        default=f"{base_default}/validation/statistical_tests.json",
        help="Full-corpus 94-day labels-only ADF/ARCH from run_statistical_tests.py.",
    )
    parser.add_argument(
        "--pre-analysis-adf",
        default=f"{base_default}/validation/pre_analysis_adf.json",
        help="Per-day ADF on first 30 days (pre_analysis_tests.py --adf).",
    )
    parser.add_argument(
        "--pre-analysis-kpss",
        default=f"{base_default}/validation/pre_analysis_kpss.json",
        help="Per-day KPSS on first 30 days (pre_analysis_tests.py --kpss).",
    )
    parser.add_argument(
        "--pre-analysis-arch",
        default=f"{base_default}/validation/pre_analysis_arch.json",
        help="Per-day ARCH on first 30 days (pre_analysis_tests.py --arch).",
    )
    parser.add_argument(
        "--purge-window",
        default=f"{base_default}/validation/purge_window.json",
        help="Empirical purge window summary (compute_purge_window.py).",
    )
    parser.add_argument(
        "--splits",
        default=f"{base_default}/splits.json",
        help="Path to splits.json. Required: every period-split table reads it.",
    )
    parser.add_argument("--output", default="pre_analysis.csv")
    parser.add_argument(
        "--reports-dir",
        default=str(Path(__file__).parent / "thesis" / "reports"),
        help=(
            "Directory for per-table .tex files, per-series CSV roll-ups, "
            "and the concatenated all_tables.tex. Created if absent. "
            "Defaults to thesis/reports/ next to this script so LaTeX can "
            "\\input them directly without a manual copy step."
        ),
    )
    args = parser.parse_args()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    val_data  = load_json(Path(args.validation_summary))
    stat_data = load_json(Path(args.stat_tests_summary))
    period_map = load_splits(Path(args.splits))
    df_target_dist = load_target_distribution(Path(args.validation_dir))

    raw_feed_path = Path(args.raw_feed_report)
    if raw_feed_path.exists():
        df_raw = pd.read_csv(raw_feed_path).rename(columns={"Date": "funding_day"})
        df_raw = attach_period(df_raw, period_map)
    else:
        print(f"  Warning: --raw-feed-report {raw_feed_path} not found; skipping raw feed table.")
        df_raw = pd.DataFrame()

    pre_adf  = load_battery(Path(args.pre_analysis_adf))
    pre_kpss = load_battery(Path(args.pre_analysis_kpss))
    pre_arch = load_battery(Path(args.pre_analysis_arch))
    purge    = load_json(Path(args.purge_window)) or {}

    df_val  = pd.DataFrame(val_data)  if val_data  else pd.DataFrame()
    df_stat = pd.DataFrame(stat_data) if stat_data else pd.DataFrame()

    df_val  = attach_period(df_val, period_map)
    df_stat = attach_period(df_stat, period_map)

    if not df_val.empty:
        unmapped = df_val[df_val["period"].isna()]["funding_day"].tolist()
        if unmapped:
            print(
                f"  Warning: {len(unmapped)} day(s) not in splits.json "
                f"(excluded from period aggregates): {unmapped[:5]}"
                f"{'…' if len(unmapped) > 5 else ''}"
            )

    # Merged CSV
    if not df_val.empty and not df_stat.empty:
        right = df_stat.drop(columns=["period"], errors="ignore")
        df = pd.merge(df_val, right, on="funding_day", how="outer")
    elif not df_val.empty:
        df = df_val
    else:
        df = df_stat

    if "period" not in df.columns:
        df = attach_period(df, period_map)

    df = df.sort_values("funding_day").reset_index(drop=True)
    df.to_csv(args.output, index=False)
    n_pre = int((df["period"] == PERIOD_PRE).sum()) if "period" in df.columns else 0
    n_sim = int((df["period"] == PERIOD_SIM).sum()) if "period" in df.columns else 0
    print(
        f"Wrote {len(df)} rows to {args.output} "
        f"(pre-analysis={n_pre}, OOS={n_sim})"
    )

    # Console summary
    if not df_stat.empty:
        print("\nADF / ARCH pass rates, per study period (full-corpus diagnostic):")
        _print_pass_rates_by_period(df_stat)

        print(f"\nVIF  max={df_stat['vif_max'].iloc[0]:.1f}  "
              f"n_high={int(df_stat['vif_n_high'].iloc[0])}")

    adf_idx  = index_battery_by_series(pre_adf)
    kpss_idx = index_battery_by_series(pre_kpss)
    arch_idx = index_battery_by_series(pre_arch)
    all_series = list(dict.fromkeys(
        list(adf_idx.keys()) + list(kpss_idx.keys()) + list(arch_idx.keys())
    ))
    df_stationarity = aggregate_stationarity(adf_idx, kpss_idx, all_series) \
        if (adf_idx and kpss_idx) else pd.DataFrame()
    df_arch_agg = aggregate_arch(arch_idx, all_series) if arch_idx else pd.DataFrame()

    if not df_stationarity.empty:
        spotlight = (
            _label_series_list()
            + ["ofi_l1", "ofi_l2_5", "vpin_micro", "vpin_base",
               "micro_return_1s", "micro_return_30s", "realized_vol_60s",
               "imbalance_l1", "trade_imbalance_1s"]
        )
        sub = df_stationarity[df_stationarity["series"].isin(spotlight)]
        if not sub.empty:
            print("\nPre-analysis joint ADF + KPSS modal class (first 30 days):")
            for _, r in sub.iterrows():
                print(f"  {r['series']:<22}  n={r['n_tested']:>2}  "
                      f"S/D/N/I={r['n_S']}/{r['n_D']}/{r['n_N']}/{r['n_I']}  "
                      f"modal={CLASS_DISPLAY.get(r['modal_class'], r['modal_class'])}")

    if purge:
        diag = purge.get("diagnostic_per_day_max_distribution") or {}
        driver = purge.get("tau_driver_feature", "??")
        driver_val = purge.get("tau_driver_value_sec", "??")
        print(
            f"\nEmpirical purge window (median-ACF, max across features): "
            f"tau_purge={purge.get('tau_purge_sec')}s "
            f"(driver={driver}={driver_val}s; "
            f"diag per-day-max median={diag.get('median_sec'):.0f}s, "
            f"p90={diag.get('p90_sec'):.0f}s, "
            f"max={diag.get('max_sec'):.0f}s "
            f"over {purge.get('n_days')} days)"
        )

    tables_emitted: list[tuple[str, str]] = []

    if not df_raw.empty:
        _emit_table(
            "tab:raw_feed_quality",
            build_raw_feed_quality_latex(df_raw),
            reports_dir=reports_dir, accumulator=tables_emitted,
        )

    if not df_val.empty:
        _emit_table(
            "tab:data_quality",
            build_data_quality_latex(df_val),
            reports_dir=reports_dir, accumulator=tables_emitted,
        )
    # Target distribution (drift_mid) across all available funding days.
    if not df_target_dist.empty:
        _emit_table(
            "tab:target_distribution",
            build_target_distribution_latex(df_target_dist),
            reports_dir=reports_dir, accumulator=tables_emitted,
        )

    if not df_stationarity.empty:
        _emit_table(
            "tab:preanalysis_features_stationarity",
            build_features_stationarity_latex(df_stationarity),
            reports_dir=reports_dir, accumulator=tables_emitted,
        )

    if purge:
        _emit_table(
            "tab:purge_window",
            build_purge_window_latex(purge),
            reports_dir=reports_dir, accumulator=tables_emitted,
        )
        _emit_table(
            "purge_window_methodology",
            build_purge_window_methodology(purge),
            reports_dir=reports_dir, accumulator=tables_emitted,
        )

    _emit_table(
        "tab:vif_summary",
        build_vif_latex(),
        reports_dir=reports_dir, accumulator=tables_emitted,
    )
    print("\n" + "=" * 70 + "\nVIF detail (all features with VIF > 10, for reference)")
    _print_vif_detail()

    # All-in-one concat + per-series CSV roll-ups.
    _write_all_tables(tables_emitted, reports_dir)
    written_csvs = _write_per_series_csvs(
        df_stationarity, df_arch_agg, purge, reports_dir=reports_dir,
    )
    print(
        f"\nWrote {len(tables_emitted)} .tex artefact(s), "
        f"all_tables.tex, and {len(written_csvs)} CSV roll-up(s) "
        f"to {reports_dir}/"
    )


if __name__ == "__main__":
    main()
