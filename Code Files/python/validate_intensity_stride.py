"""Validation of the rolling-intensity stride."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

try:
    import polars as pl
    _HAVE_PL = True
except Exception:  # pragma: no cover
    _HAVE_PL = False

try:
    import pipeline_utils as pu
    _HAVE_PU = True
except Exception:  # pragma: no cover
    _HAVE_PU = False

from calibrate_intensity import (
    DEFAULT_BASE, joined_market_orders, extract_queue_fill_records,
)
from calibrate_intensity_rolling import (
    RollConfig,
    _load_funding_day_with_warmup,
    rolling_mle,
    rolling_queue_aware,
    _load_qa_day_with_warmup,
    WINDOW_S_QA_DEFAULT,
)


DEFAULT_OUT_DIR = Path("reports")
DEFAULT_OUT_STEM = "validate_intensity_stride"
DEFAULT_LAGS_S: tuple[int, ...] = (60, 300, 900)
DEFAULT_S_FINE = 5
DEFAULT_S_DEFAULT = 60
DEFAULT_WINDOW_S = 15 * 60
DEFAULT_S_FINE_QA = 60


@dataclass
class StrideStats:
    lag_s: int
    n_pairs: int
    median: float
    p95: float
    p99: float


@dataclass
class StrideResult:
    metric: str        # "k" or "A"
    s_fine_s: int
    window_s: int
    n_rows_total: int
    median_n_orders_per_window: float
    noise_floor_pct: float  # closed-form Fisher SE on the chosen metric, in %
    lags: list[StrideStats]


def relative_changes(values: np.ndarray, ts_ms: np.ndarray,
                     lag_s: int, fine_stride_s: int) -> np.ndarray:
    """|Delta v / v| between rows exactly lag_s seconds apart on the SAME funding day."""
    step = int(round(lag_s / fine_stride_s))
    if step <= 0 or values.size <= step:
        return np.empty(0, dtype=float)
    v0 = values[:-step]
    v1 = values[step:]
    dt = ts_ms[step:] - ts_ms[:-step]
    expected_ms = lag_s * 1000
    tol_ms = (fine_stride_s * 1000) // 2 + 1
    ok = (
        np.isfinite(v0) & np.isfinite(v1)
        & (v0 > 0) & (v1 > 0)   # both endpoints must be a valid positive (A, k);
                                # v0 also guards the denominator of the ratio
        & (np.abs(dt - expected_ms) <= tol_ms)
    )
    v0, v1 = v0[ok], v1[ok]
    if v0.size == 0:
        return np.empty(0, dtype=float)
    return np.abs((v1 - v0) / v0)


def stats_per_lag(values: np.ndarray, ts_ms: np.ndarray,
                  lags_s: tuple[int, ...], fine_stride_s: int
                  ) -> list[StrideStats]:
    out: list[StrideStats] = []
    for lag in lags_s:
        diffs = relative_changes(values, ts_ms, lag, fine_stride_s)
        if diffs.size == 0:
            out.append(StrideStats(lag_s=lag, n_pairs=0,
                                    median=float("nan"),
                                    p95=float("nan"),
                                    p99=float("nan")))
            continue
        out.append(StrideStats(
            lag_s=lag, n_pairs=int(diffs.size),
            median=float(np.median(diffs)),
            p95=float(np.quantile(diffs, 0.95)),
            p99=float(np.quantile(diffs, 0.99)),
        ))
    return out


def pool_per_day_diffs(per_day_rows: list["pl.DataFrame"], col: str,
                        lags_s: tuple[int, ...], fine_stride_s: int
                        ) -> dict[int, np.ndarray]:
    """For each lag, concatenate per-day diffs (so a 04:00 UTC boundary
    never produces a synthetic same-day pair across days)."""
    bag: dict[int, list[np.ndarray]] = {lag: [] for lag in lags_s}
    for df in per_day_rows:
        if df.is_empty():
            continue
        d = df.filter(pl.col("valid"))
        if d.height < 2:
            continue
        v = d[col].to_numpy()
        t = d["ts_ms"].to_numpy()
        for lag in lags_s:
            bag[lag].append(relative_changes(v, t, lag, fine_stride_s))
    return {lag: (np.concatenate(parts) if parts else np.empty(0))
            for lag, parts in bag.items()}


def run_real(base: Path, fdays: list[str], out_dir: Path, *,
             window_s: int, s_fine: int, s_default: int,
             lags_s: tuple[int, ...], queue_aware: bool = False) -> dict:
    if not (_HAVE_PL and _HAVE_PU):
        raise SystemExit("polars/pipeline_utils unavailable")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = DEFAULT_OUT_STEM + ("_qa" if queue_aware else "")
    if queue_aware:
        cfg = RollConfig(window_s=window_s, stride_s=s_fine, queue_aware=True)
        from data_gap_handler import load_pause_intervals, merge_intervals
    else:
        cfg = RollConfig(window_s=window_s, stride_s=s_fine,
                         min_orders_per_window=100)
    per_day_rows: list[pl.DataFrame] = []
    per_day_meta: list[dict] = []
    for fday in fdays:
        if queue_aware:
            trades, book, start_ms, end_ms = _load_qa_day_with_warmup(
                base, fday, cfg)
            if trades.is_empty() or book.is_empty():
                per_day_meta.append({"fday": fday, "status": "no_data"})
                print(f"  {fday}: no data, skipped")
                continue
            pauses = merge_intervals(load_pause_intervals(base, fday))
            recs = extract_queue_fill_records(
                trades, book, horizon_s=cfg.qa_extract_horizon_s,
                repost_s=cfg.qa_repost_s, pause_intervals=pauses)
            rolled = rolling_queue_aware(recs, start_ms, end_ms, cfg)
        else:
            trades, book, start_ms, end_ms = _load_funding_day_with_warmup(
                base, fday, cfg)
            if trades.is_empty() or book.is_empty():
                per_day_meta.append({"fday": fday, "status": "no_data"})
                print(f"  {fday}: no data, skipped")
                continue
            joined = joined_market_orders(trades, book)
            rolled = rolling_mle(joined, start_ms, end_ms, cfg)
        per_day_rows.append(rolled)
        valid = rolled.filter(pl.col("valid"))
        per_day_meta.append({
            "fday": fday, "status": "ok",
            "rows": rolled.height, "valid_rows": valid.height,
            "n_orders_median": int(valid["n_orders"].median())
                if valid.height else None,
        })
        print(f"  {fday}: rolled rows={rolled.height} (valid={valid.height})")

    diffs_k = pool_per_day_diffs(per_day_rows, "k", lags_s, s_fine)
    diffs_A = pool_per_day_diffs(per_day_rows, "A", lags_s, s_fine)

    valid_all = pl.concat([df.filter(pl.col("valid")) for df in per_day_rows],
                          how="vertical_relaxed") if per_day_rows else pl.DataFrame()

    def _se_floor(val_col: str, se_col: str) -> float:
        """sqrt(2) x median(SE/value) over valid windows, in percent."""
        if valid_all.is_empty() or se_col not in valid_all.columns:
            return float("nan")
        rel = (valid_all[se_col] / valid_all[val_col]).to_numpy()
        rel = rel[np.isfinite(rel) & (rel > 0)]
        return 100.0 * math.sqrt(2.0) * float(np.median(rel)) if rel.size else float("nan")

    if valid_all.is_empty():
        n_med = float("nan"); n_total = 0
        floor_k = floor_A = float("nan")
        noise_floor_method = "n/a (no valid rows)"
    else:
        n_med = float(valid_all["n_orders"].median())
        n_total = int(valid_all.height)
        floor_A = _se_floor("A", "A_se")
        if queue_aware:
            floor_k = _se_floor("k", "k_se")
            noise_floor_method = ("sqrt(2) x median(SE/value) from per-window fits "
                                  "(k: k_se/k, A: A_se/A)")
        else:
            floor_k = (100.0 * math.sqrt(2.0) / math.sqrt(n_med)
                       if n_med > 0 else float("nan"))
            noise_floor_method = ("k: sqrt(2)/sqrt(N) (exp-MLE exact); "
                                  "A: sqrt(2) x median(A_se/A)")

    def _stats_from_bag(bag: dict[int, np.ndarray]) -> list[StrideStats]:
        out: list[StrideStats] = []
        for lag in lags_s:
            d = bag[lag]
            if d.size == 0:
                out.append(StrideStats(lag, 0, float("nan"),
                                        float("nan"), float("nan")))
            else:
                out.append(StrideStats(
                    lag, int(d.size),
                    float(np.median(d)),
                    float(np.quantile(d, 0.95)),
                    float(np.quantile(d, 0.99)),
                ))
        return out

    res_k = StrideResult(
        metric="k", s_fine_s=s_fine, window_s=window_s,
        n_rows_total=n_total, median_n_orders_per_window=n_med,
        noise_floor_pct=floor_k,
        lags=_stats_from_bag(diffs_k),
    )
    res_A = StrideResult(
        metric="A", s_fine_s=s_fine, window_s=window_s,
        n_rows_total=n_total, median_n_orders_per_window=n_med,
        noise_floor_pct=floor_A,
        lags=_stats_from_bag(diffs_A),
    )
    summary = {
        "method": "queue_aware_rolling" if queue_aware else "trade_through_rolling",
        "fdays": [m for m in per_day_meta],
        "window_s": window_s, "s_fine_s": s_fine,
        "s_default_s": s_default,
        "noise_floor_method": noise_floor_method,
        "noise_floor_pct_k_indep": floor_k,   # lag>=L (independent) bound
        "noise_floor_pct_A_indep": floor_A,
        "noise_floor_lag_scaling": ("floor(lag) = noise_floor_pct_x_indep * "
                                    "sqrt(min(lag, window_s)/window_s); windows "
                                    "lag apart overlap by (L-lag)/L"),
        "k": _stride_result_to_dict(res_k),
        "A": _stride_result_to_dict(res_A),
        "verdict": _verdict(res_k, s_default),
    }
    if queue_aware:
        summary["qa"] = {"qa_horizon_s": cfg.qa_horizon_s,
                         "qa_extract_horizon_s": cfg.qa_extract_horizon_s,
                         "qa_repost_s": cfg.qa_repost_s,
                         "qa_min_fills_window": cfg.qa_min_fills_window}
    (out_dir / f"{stem}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    _write_latex(res_k, res_A,
                 out_dir / ("tab_intensity_stride_qa.tex" if queue_aware
                            else "tab_intensity_stride.tex"),
                 s_default=s_default)
    _print_report(res_k, res_A, summary["verdict"])
    if per_day_rows:
        _make_figure(per_day_rows, diffs_k, diffs_A, lags_s,
                      out_dir / f"{stem}.png")
    return summary


def _stride_result_to_dict(r: StrideResult) -> dict:
    lag_rows = []
    for s in r.lags:
        d = asdict(s)
        fl_pct = _lag_floor(r.noise_floor_pct, s.lag_s, r.window_s)
        d["floor"] = fl_pct / 100.0 if math.isfinite(fl_pct) else float("nan")
        d["p99_over_floor"] = (s.p99 / d["floor"]
                               if (math.isfinite(d["floor"]) and d["floor"] > 0
                                   and math.isfinite(s.p99)) else float("nan"))
        lag_rows.append(d)
    return {
        "metric": r.metric, "s_fine_s": r.s_fine_s, "window_s": r.window_s,
        "n_rows_total": r.n_rows_total,
        "median_n_orders_per_window": r.median_n_orders_per_window,
        "noise_floor_pct_indep_estimates": r.noise_floor_pct,
        "lags": lag_rows,
    }


def _lag_floor(base_floor_pct: float, lag_s: int, window_s: int) -> float:
    """Overlap-adjusted noise floor at `lag_s`. Two rolling windows lag apart share
    (L-lag)/L of their data, so the noise-only diff scales with the fresh fraction:
    floor(lag) = base * sqrt(min(lag, L)/L), where base = sqrt(2) x rel_se is the
    independent (lag>=L) bound. Equals base at lag>=L; falls below it for lag<L."""
    if not (window_s > 0) or not math.isfinite(base_floor_pct):
        return base_floor_pct
    return base_floor_pct * math.sqrt(min(lag_s, window_s) / window_s)


def _verdict(res_k: StrideResult, s_default: int) -> dict:
    """Headline verdict on whether S = s_default is well-chosen on k."""
    floor = _lag_floor(res_k.noise_floor_pct, s_default, res_k.window_s)
    p99_at_s = next((s for s in res_k.lags if s.lag_s == s_default), None)
    if p99_at_s is None or not math.isfinite(p99_at_s.p99):
        return {"verdict": "inconclusive (no diffs at default stride)",
                 "ratio_p99_to_floor": None}
    p99_pct = 100.0 * p99_at_s.p99
    ratio = p99_pct / floor if (floor and math.isfinite(floor)) else float("nan")
    if ratio > 3.0:
        text = (
            f"S = {s_default} s captures real variability: at lag {s_default}s "
            f"the p99 |Delta k / k| = {p99_pct:.2f}% is {ratio:.1f}x the "
            f"overlap-adjusted noise floor at this lag ({floor:.2f}%). "
            f"Keep S = {s_default}."
        )
    elif ratio > 1.5:
        text = (
            f"S = {s_default} s sits at the noise floor: p99 |Delta k / k| = "
            f"{p99_pct:.2f}% vs the lag-{s_default}s overlap-adjusted floor "
            f"{floor:.2f}% (ratio {ratio:.1f}x). "
            f"S = {s_default} is defensible but S = 300 s would lose little."
        )
    else:
        text = (
            f"S = {s_default} s is dominated by estimator noise: p99 |Delta k / k|"
            f" = {p99_pct:.2f}% vs the lag-{s_default}s overlap-adjusted floor "
            f"{floor:.2f}% (ratio {ratio:.1f}x). "
            f"S could safely be coarser (300-900 s)."
        )
    return {"verdict": text, "ratio_p99_to_floor": ratio,
             "p99_pct": p99_pct, "noise_floor_pct": floor}


def _print_report(res_k: StrideResult, res_A: StrideResult,
                  verdict: dict) -> None:
    print("\n=== Rolling-intensity stride validation ===")
    print(f"window L = {res_k.window_s}s   fine stride S_fine = {res_k.s_fine_s}s")
    print(f"valid rows total = {res_k.n_rows_total}   "
          f"median N per window = {res_k.median_n_orders_per_window:.0f}")
    print(f"lag>=L (independent) noise floor sqrt(2)xrel-SE:  "
          f"k = {res_k.noise_floor_pct:.2f}%   A = {res_A.noise_floor_pct:.2f}%")
    print(f"per-lag floor = that x sqrt(lag/L) "
          f"(L = {res_k.window_s}s; windows lag apart overlap by (L-lag)/L)")
    hdr = (f"{'metric':>6} {'lag(s)':>7} {'n_pairs':>9} {'median%':>9} "
           f"{'p95%':>8} {'p99%':>8} {'floor%':>8} {'p99/fl':>7}")
    print(hdr)
    for r in (res_k, res_A):
        for s in r.lags:
            fl = _lag_floor(r.noise_floor_pct, s.lag_s, r.window_s)
            ratio = (100 * s.p99 / fl
                     if (math.isfinite(fl) and fl > 0 and math.isfinite(s.p99))
                     else float("nan"))
            print(
                f"{r.metric:>6} {s.lag_s:>7} {s.n_pairs:>9} "
                f"{100*s.median:>9.3f} {100*s.p95:>8.3f} {100*s.p99:>8.3f} "
                f"{fl:>8.3f} {ratio:>7.2f}"
            )
    print()
    print("Headline:", verdict.get("verdict"))


def _write_latex(res_k: StrideResult, res_A: StrideResult,
                  path: Path, *, s_default: int) -> None:
    lines = [
        r"% Auto-generated by validate_intensity_stride.py",
        r"% Stride validation for rolling (A, k).",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"metric \& lag (s) & $n_{\text{pairs}}$ & median (\%) & p95 (\%) "
        r"& p99 (\%) & floor (\%) \\",
        r"\midrule",
    ]
    for r in (res_k, res_A):
        for s in r.lags:
            fl = _lag_floor(r.noise_floor_pct, s.lag_s, r.window_s)
            lines.append(
                f"${r.metric}$, {s.lag_s} & {s.n_pairs} & "
                f"{100*s.median:.3f} & {100*s.p95:.3f} & {100*s.p99:.3f} "
                f"& {fl:.3f} \\\\"
            )
    lines += [
        r"\midrule",
        r"\multicolumn{6}{l}{Per-lag noise floor "
        f"$= (\\sqrt{{2}}\\times$rel-SE$)\\times\\sqrt{{\\mathrm{{lag}}/L}}$, "
        f"$L = {res_k.window_s}$ s; lag$\\ge L$ bound "
        f"$k = {res_k.noise_floor_pct:.2f}\\%$, $A = {res_A.noise_floor_pct:.2f}\\%$. "
        f"Default stride $S = {s_default}$ s.}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines))


def _make_figure(per_day_rows: list["pl.DataFrame"],
                  diffs_k: dict[int, np.ndarray],
                  diffs_A: dict[int, np.ndarray],
                  lags_s: tuple[int, ...], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) Histograms of |Delta k / k| at each lag, in percent (log-x)
    bins = np.logspace(-3.5, 0.0, 60)  # 0.03% .. 100%
    for lag in lags_s:
        d = diffs_k[lag]
        if d.size == 0:
            continue
        ax0.hist(d, bins=bins, histtype="step", lw=1.5,
                  label=f"lag {lag}s (n={d.size})")
    ax0.set_xscale("log")
    ax0.set_xlabel(r"$|\Delta k / k|$")
    ax0.set_ylabel("count")
    ax0.set_title("Per-stride change in k (pre-analysis pool)")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3, which="both")

    # (b) k(t) for the longest valid day (sanity strip)
    longest = max(per_day_rows,
                   key=lambda df: df.filter(pl.col("valid")).height
                                  if not df.is_empty() else 0)
    long_valid = longest.filter(pl.col("valid"))
    if long_valid.height:
        t = (long_valid["ts_ms"].to_numpy() - long_valid["ts_ms"].min()) / 3600_000.0
        ax1.plot(t, long_valid["k"].to_numpy(), color="C0", lw=1.0,
                  label="k (from-mid)")
        ax1.plot(t, long_valid["k_touch"].to_numpy(), color="C3", lw=1.0,
                  alpha=0.7, label="k_touch (floor-free)")
        ax1.set_xlabel("hours since 04:00 UTC")
        ax1.set_ylabel(r"$k$ (1/USDT)")
        ax1.set_title("k(t) on the longest pre-analysis day")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--queue-aware", action="store_true",
                     help="validate the queue-aware rolling (A, k) instead "
                          "of trade-through")
    ap.add_argument("--window-s", type=int, default=None,
                     help=f"trailing window L (default {DEFAULT_WINDOW_S}s; "
                          f"{WINDOW_S_QA_DEFAULT}s under --queue-aware)")
    ap.add_argument("--s-fine", type=int, default=None,
                     help=f"fine stride for the validation run (default "
                          f"{DEFAULT_S_FINE}s; {DEFAULT_S_FINE_QA}s under --queue-aware)")
    ap.add_argument("--s-default", type=int, default=DEFAULT_S_DEFAULT,
                     help="the stride being defended (default 60)")
    ap.add_argument("--lags", type=int, nargs="+", default=list(DEFAULT_LAGS_S))
    ap.add_argument("--fdays", nargs="+", default=None,
                     help="explicit funding-day list "
                          "(default: splits.json -> pre_analysis)")
    args = ap.parse_args()


    if not (_HAVE_PL and _HAVE_PU):
        raise SystemExit("polars/pipeline_utils unavailable")

    window_s = args.window_s if args.window_s is not None else (
        WINDOW_S_QA_DEFAULT if args.queue_aware else DEFAULT_WINDOW_S)
    s_fine = args.s_fine if args.s_fine is not None else (
        DEFAULT_S_FINE_QA if args.queue_aware else DEFAULT_S_FINE)

    base = Path(args.base)
    out_dir = Path(args.out_dir)
    if args.fdays is not None:
        fdays = list(args.fdays)
    else:
        fdays = list(pu.load_splits(base)["splits"]["pre_analysis"])

    print(f"  stride validation ({'queue-aware' if args.queue_aware else 'trade-through'}): "
          f"window={window_s}s  s_fine={s_fine}s  s_default={args.s_default}s  "
          f"fdays={len(fdays)} (pre-analysis only)")
    run_real(base, fdays, out_dir,
             window_s=window_s, s_fine=s_fine,
             s_default=args.s_default, lags_s=tuple(args.lags),
             queue_aware=args.queue_aware)


if __name__ == "__main__":
    main()
