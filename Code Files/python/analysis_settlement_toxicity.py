"""Adverse selection near the funding settlement versus away from it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

RUN_ROOT_DEFAULT = Path(
    "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt/runs/rerun_20260807_ar")
BASE_CELL = "ablation_h10_qa"

NEAR_MAX_U_S = 900.0
FAR_MIN_U_S = 7200.0

METRICS = ("advsel_bps", "abs_move_bps", "swept_pct", "fills_per_hour")
METRIC_LABELS = {
    "advsel_bps": r"Adverse selection (bps)",
    "abs_move_bps": r"$|$mid move$|$ 5\,s (bps)",
    "swept_pct": r"Swept share (\%)",
    "fills_per_hour": r"Fills per hour",
}


def _days(sdir: Path) -> list[str]:
    return sorted(p.stem.replace("fills_", "") for p in sdir.glob("fills_*.parquet"))


def day_metrics(sdir: Path, day: str, live_h: dict[str, float]) -> dict:
    """Near/far metrics for one strategy-day. NaN if a window has no fills."""
    f = (pl.read_parquet(sdir / f"fills_{day}.parquet",
                         columns=["u_s", "side", "price", "swept",
                                  "is_liquidation", "mid_0s", "mid_5s"])
         .filter(~pl.col("is_liquidation")))
    px = f["price"].to_numpy()
    m0, m5 = f["mid_0s"].to_numpy(), f["mid_5s"].to_numpy()
    side = f["side"].to_numpy().astype(float)
    u = f["u_s"].to_numpy()
    swept = f["swept"].to_numpy()
    ok = np.isfinite(m0) & np.isfinite(m5) & np.isfinite(px) & (px > 0)

    out: dict = {"day": day}
    for name, mask in (("near", u < NEAR_MAX_U_S), ("far", u > FAR_MIN_U_S)):
        w = mask & ok
        n = int(w.sum())
        h = live_h.get((day, name), float("nan"))
        out[name] = {
            "n_fills": n,
            "advsel_bps": float((1e4 * side[w] * (m5[w] - m0[w]) / px[w]).mean())
            if n else float("nan"),
            "abs_move_bps": float((1e4 * np.abs(m5[w] - m0[w]) / px[w]).mean())
            if n else float("nan"),
            "swept_pct": float(100.0 * swept[mask].mean())
            if int(mask.sum()) else float("nan"),
            "fills_per_hour": (int(mask.sum()) / h) if h and np.isfinite(h)
            else float("nan"),
            "live_hours": h,
        }
    return out


def live_hours(sdir: Path) -> dict[tuple[str, str], float]:
    """Live (non-paused) hours per day in each window, from the 1 Hz grid."""
    out: dict[tuple[str, str], float] = {}
    for day in _days(sdir):
        s = (pl.read_parquet(sdir / f"pnl_{day}.parquet",
                             columns=["u_s", "paused"])
             .filter(~pl.col("paused")))
        u = s["u_s"].to_numpy()
        out[(day, "near")] = float((u < NEAR_MAX_U_S).sum()) / 3600.0
        out[(day, "far")] = float((u > FAR_MIN_U_S).sum()) / 3600.0
    return out


ALIGN_BINS: tuple[tuple[str, float, float], ...] = (
    ("<1m", 0.0, 60.0), ("1-5m", 60.0, 300.0), ("5-15m", 300.0, 900.0),
    ("15-30m", 900.0, 1800.0), ("30m-2h", 1800.0, 7200.0),
    (">2h", 7200.0, float("inf")))


def signal_alignment(cell: Path, sid: int) -> dict:
    """Is the drift signal aligned with the funding it never reads?"""
    out: dict = {}
    a_all, f_all = [], []
    acc = {b[0]: {"a": [], "s": [], "f": []} for b in ALIGN_BINS}
    for p in sorted((cell / f"s{sid}").glob("fills_*.parquet")):
        day = p.stem.replace("fills_", "")
        g = pl.read_parquet(cell / f"s{sid}" / f"pnl_{day}.parquet",
                            columns=["ts_ms", "f_rate"])
        f = (pl.read_parquet(p, columns=["ts_ms", "u_s", "side", "alpha_ml",
                                         "is_liquidation"])
             .filter(~pl.col("is_liquidation") & (pl.col("alpha_ml") != 0.0))
             .with_columns((pl.col("ts_ms") // 1000 * 1000).alias("sec"))
             .join(g.with_columns((pl.col("ts_ms") // 1000 * 1000).alias("sec"))
                   .select("sec", "f_rate"), on="sec", how="inner"))
        if not f.height:
            continue
        u = f["u_s"].to_numpy()
        a, s, fr = (f["alpha_ml"].to_numpy(), f["side"].to_numpy().astype(float),
                    f["f_rate"].to_numpy())
        a_all.append(a)
        f_all.append(fr)
        for label, lo, hi in ALIGN_BINS:
            m = (u >= lo) & (u < hi)
            acc[label]["a"].append(a[m])
            acc[label]["s"].append(s[m])
            acc[label]["f"].append(fr[m])
    A, F = np.concatenate(a_all), np.concatenate(f_all)
    out["pooled"] = {
        "n_fills": int(A.size),
        "corr_alpha_frate": float(np.corrcoef(A, F)[0, 1]),
        "pct_signal_toward_collecting": float(
            100.0 * np.mean(-np.sign(F) * np.sign(A) > 0))}
    out["by_u"] = []
    for label, lo, hi in ALIGN_BINS:
        a = np.concatenate(acc[label]["a"])
        s = np.concatenate(acc[label]["s"])
        fr = np.concatenate(acc[label]["f"])
        if a.size < 2:
            continue
        sig = (-np.sign(fr) * np.sign(a) > 0).astype(float)
        fil = (-np.sign(fr) * s > 0).astype(float)
        out["by_u"].append({
            "bin": label, "n_fills": int(a.size),
            "pct_signal_toward_collecting": float(100.0 * sig.mean()),
            "signal_se_pp": float(100.0 * sig.std(ddof=1) / np.sqrt(a.size)),
            "pct_fills_on_collecting_side": float(100.0 * fil.mean()),
            "fills_se_pp": float(100.0 * fil.std(ddof=1) / np.sqrt(a.size)),
            "corr_alpha_frate": float(np.corrcoef(a, fr)[0, 1]),
            "mean_abs_alpha": float(np.abs(a).mean()),
            "mean_abs_frate_bps": float(np.abs(fr).mean() * 1e4),
            "corr_abs_alpha_abs_frate": float(
                np.corrcoef(np.abs(a), np.abs(fr))[0, 1])})
    return out


def mde(sd: float, n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest paired difference this design would detect with `power`."""
    z_a = stats.norm.ppf(1.0 - alpha / 2.0)
    z_b = stats.norm.ppf(power)
    return (z_a + z_b) * sd / np.sqrt(n)


def paired(rows: list[dict], metric: str) -> dict:
    """Paired near-minus-far test across days; days with either side NaN drop."""
    near = np.array([r["near"][metric] for r in rows], dtype=float)
    far = np.array([r["far"][metric] for r in rows], dtype=float)
    ok = np.isfinite(near) & np.isfinite(far)
    near, far = near[ok], far[ok]
    diff = near - far
    sd = diff.std(ddof=1)
    t, p = stats.ttest_rel(near, far)
    return {"near": float(near.mean()), "far": float(far.mean()),
            "diff": float(diff.mean()),
            "se": float(sd / np.sqrt(len(diff))),
            "t": float(t), "p": float(p),
            "mde_80pct_power": float(mde(sd, len(diff))),
            "n_days": int(ok.sum())}


def strategy_panel(cell: Path, sid: int) -> dict:
    sdir = cell / f"s{sid}"
    lh = live_hours(sdir)
    rows = [day_metrics(sdir, day, lh) for day in _days(sdir)]
    res = {m: paired(rows, m) for m in METRICS}
    res["n_fills_near"] = int(sum(r["near"]["n_fills"] for r in rows))
    res["n_fills_far"] = int(sum(r["far"]["n_fills"] for r in rows))
    res["n_days"] = len(rows)
    return res


def write_tex(panel: dict, path: Path, sid: int) -> None:
    lines = [
        f"% generated by analysis_settlement_toxicity.py (strategy {sid})",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        (r"Statistic & Near & Far & Paired diff.\ & $t$ & $p$ \\"),
        (r" & ($u<15$\,min) & ($u>2$\,h) & & & \\"),
        r"\midrule",
    ]
    for m in METRICS:
        r = panel[m]
        if m == "fills_per_hour":
            lines.append(f"{METRIC_LABELS[m]} & {r['near']:,.1f} & "
                         f"{r['far']:,.1f} & & & \\\\")
            continue
        p = r["p"]
        ps = f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"
        lines.append(
            f"{METRIC_LABELS[m]} & {r['near']:.4f} & {r['far']:.4f} & "
            f"{r['diff']:+.4f} & {r['t']:+.2f} & {ps} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)
    ap.add_argument("--cell", type=str, default=BASE_CELL)
    ap.add_argument("--report-dir", type=Path,
                    default=Path("reports/rerun_20260807_ar/settlement_toxicity"))
    ap.add_argument("--strategies", type=str, default="2,4",
                    help="first is the headline panel written to .tex")
    args = ap.parse_args()

    cell = args.run_root / args.cell
    sids = [int(s) for s in args.strategies.split(",")]
    res = {"run": str(cell), "near_max_u_s": NEAR_MAX_U_S,
           "far_min_u_s": FAR_MIN_U_S,
           "panels": {f"s{s}": strategy_panel(cell, s) for s in sids},
           "signal_alignment": {f"s{s}": signal_alignment(cell, s)
                                for s in sids}}
    if "s2" in res["signal_alignment"]:
        base = {r["bin"]: r for r in res["signal_alignment"]["s2"]["by_u"]}
        for k, al in res["signal_alignment"].items():
            if k == "s2":
                continue
            al["vs_s2"] = []
            for r in al["by_u"]:
                b = base.get(r["bin"])
                if b is None:
                    continue
                se = float(np.hypot(r["fills_se_pp"], b["fills_se_pp"]))
                d = r["pct_fills_on_collecting_side"] - b["pct_fills_on_collecting_side"]
                al["vs_s2"].append({
                    "bin": r["bin"], "diff_pp": d, "se_pp": se,
                    "z": d / se if se > 0 else float("nan")})

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "settlement_toxicity.json").write_text(
        json.dumps(res, indent=1))
    write_tex(res["panels"][f"s{sids[0]}"],
              args.report_dir / "tab_settlement_toxicity.tex", sids[0])

    for s in sids:
        print(f"--- strategy {s} ---")
        for m in METRICS:
            r = res["panels"][f"s{s}"][m]
            print(f"  {m:>14}  near {r['near']:10.4f}  far {r['far']:10.4f}  "
                  f"diff {r['diff']:+9.4f}  t={r['t']:+6.2f}  p={r['p']:.4f}")
    for s, al in res["signal_alignment"].items():
        p = al["pooled"]
        print(f"--- {s} signal/funding alignment (n={p['n_fills']:,} fills) ---")
        print(f"  pooled: corr(alpha,f) = {p['corr_alpha_frate']:+.5f}   "
              f"signal toward collecting side {p['pct_signal_toward_collecting']:.2f}%")
        for r in al["by_u"]:
            print(f"  {r['bin']:>7}: signal {r['pct_signal_toward_collecting']:5.2f}%"
                  f" (+/-{r['signal_se_pp']:.2f})   fills on collecting side "
                  f"{r['pct_fills_on_collecting_side']:5.2f}% (+/-{r['fills_se_pp']:.2f})"
                  f"   corr {r['corr_alpha_frate']:+.4f}   n={r['n_fills']:,}")
        for r in al.get("vs_s2", []):
            print(f"    vs s2 {r['bin']:>7}: fills on collecting side "
                  f"{r['diff_pp']:+5.2f} pp (+/-{r['se_pp']:.2f}, z={r['z']:+5.2f})")
    print("written:", args.report_dir)


if __name__ == "__main__":
    main()
