#!/usr/bin/env python3
"""Ridge versus LightGBM Diebold-Mariano comparison."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from statistical_tests import diebold_mariano_test
from ml_train import DEFAULT_BASE, LABEL_HORIZONS_S, TARGET_FAMILIES


def _load_predictions(path: Path) -> pl.DataFrame:
    """Read one oos_predictions_h<h>.parquet, with a clear error message
    if the file is absent (most common cause: the corresponding training
    script has not been run yet for this (target, horizon) combination)."""
    if not path.exists():
        raise SystemExit(
            f"oos predictions parquet not found: {path}\n"
            f"  Run the corresponding training script first."
        )
    df = pl.read_parquet(path)
    required = {"ts_ms", "fold", "test_day", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")
    return df


def _join_aligned(ridge: pl.DataFrame, lgb: pl.DataFrame) -> pl.DataFrame:
    """Inner-join Ridge and LightGBM predictions on the (test_day, ts_ms)
    primary key, with a sanity check that the two `y_true` columns agree."""
    joined = ridge.rename({"y_pred": "y_pred_ridge", "y_true": "y_true_ridge"}).join(
        lgb.rename({"y_pred": "y_pred_lgb", "y_true": "y_true_lgb"}),
        on=["test_day", "ts_ms"],
        how="inner",
    )
    if joined.is_empty():
        raise SystemExit(
            "join produced zero rows, the two prediction parquets share "
            "no (test_day, ts_ms) keys. Re-run both training scripts on "
            "the same target / horizon / split"
        )
    mismatch = (
        (joined["y_true_ridge"] - joined["y_true_lgb"]).abs() > 1e-7
    ).sum()
    if mismatch:
        raise SystemExit(
            f"y_true disagrees on {mismatch} row(s) between the two "
            f"parquets, the test would be invalid. Check that Ridge and "
            f"LightGBM ran on identical fold layout and row filter."
        )
    return joined


def _per_fold_metrics(joined: pl.DataFrame) -> list[dict]:
    """MSE for both models, per outer fold. Useful for spotting wins
    driven by a single outlier fold vs consistent across all folds."""
    rows = []
    fold_col = "fold" if "fold" in joined.columns else "fold_right"
    for fold_id, sub in joined.group_by(fold_col, maintain_order=True):
        # group_by yields a tuple (key,) for a single grouping column
        fid = int(fold_id[0] if isinstance(fold_id, tuple) else fold_id)
        y = sub["y_true_lgb"].to_numpy()
        r = sub["y_pred_ridge"].to_numpy()
        l = sub["y_pred_lgb"].to_numpy()
        rows.append({
            "fold": fid,
            "n": int(sub.height),
            "mse_ridge": float(np.mean((y - r) ** 2)),
            "mse_lgb":   float(np.mean((y - l) ** 2)),
            "lgb_wins":  bool(np.mean((y - l) ** 2) < np.mean((y - r) ** 2)),
        })
    rows.sort(key=lambda r: r["fold"])
    return rows


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    """Out-of-sample R^2 against the realised-mean benchmark. Returns NaN on a
    degenerate (zero-variance) day so the caller can skip it cleanly."""
    sstot = float(np.sum((y - y.mean()) ** 2))
    if sstot <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y - p) ** 2)) / sstot


def _per_test_day_metrics(joined: pl.DataFrame) -> list[dict]:
    rows = []
    for key, sub in joined.group_by("test_day", maintain_order=True):
        day = key[0] if isinstance(key, tuple) else key
        y = sub["y_true_lgb"].to_numpy()
        r = sub["y_pred_ridge"].to_numpy()
        l = sub["y_pred_lgb"].to_numpy()
        r2r, r2l = _r2(y, r), _r2(y, l)
        rows.append({
            "test_day": str(day),
            "n": int(sub.height),
            "target_std": float(np.std(y)),
            "r2_ridge": r2r,
            "r2_lgb": r2l,
            "dr2_lgb_minus_ridge": r2l - r2r,
            "mse_ridge": float(np.mean((y - r) ** 2)),
            "mse_lgb": float(np.mean((y - l) ** 2)),
            "lgb_wins": bool(np.mean((y - l) ** 2) < np.mean((y - r) ** 2)),
        })
    rows.sort(key=lambda x: x["test_day"])
    return rows


def _regime_split(joined: pl.DataFrame, per_day: list[dict]) -> dict:
    stds = sorted(d["target_std"] for d in per_day)
    median_std = stds[len(stds) // 2]
    hi_days = [d["test_day"] for d in per_day if d["target_std"] >= median_std]
    out = {"median_target_std": median_std, "high_vol_days": hi_days}
    for label, day_set in (("high_vol", set(hi_days)),
                           ("calm", {d["test_day"] for d in per_day} - set(hi_days))):
        sub = joined.filter(pl.col("test_day").is_in(list(day_set)))
        y = sub["y_true_lgb"].to_numpy()
        out[label] = {
            "n_days": len(day_set),
            "n_rows": int(sub.height),
            "r2_ridge": _r2(y, sub["y_pred_ridge"].to_numpy()),
            "r2_lgb": _r2(y, sub["y_pred_lgb"].to_numpy()),
        }
        out[label]["dr2_lgb_minus_ridge"] = out[label]["r2_lgb"] - out[label]["r2_ridge"]
    return out


def _format_p_for_latex(p: float) -> str:
    if not math.isfinite(p):
        return "--"
    if p < 1e-10:
        return "$<10^{-10}$"
    if p < 1e-3:
        exp = int(np.ceil(np.log10(p)))
        return f"$<10^{{{exp}}}$"
    return f"{p:.4f}"


def _write_latex(
    summary: dict, per_fold: list[dict], out_path: Path, horizon: int, target: str
) -> None:
    """Single-table LaTeX summary for the thesis appendix."""
    n_lgb_wins = sum(1 for r in per_fold if r["lgb_wins"])
    n_folds = len(per_fold)
    decision = summary["decision"]
    rows = [
        ["Test", "Diebold-Mariano (squared-error loss)"],
        ["Sign", "$d_t = (y - \\hat{y}_{\\text{ridge}})^2 - (y - \\hat{y}_{\\text{lgb}})^2$;  $\\bar{d} > 0 \\Rightarrow$ LGB lower MSE"],
        ["DM statistic", f"{summary['dm_statistic']:+.3f}"],
        ["DM $p$-value", _format_p_for_latex(summary["p_value"])],
        ["HAC lag", f"{summary['hac_nlags']}"],
        ["$n$ observations", f"{summary['n_observations']:,}"],
        ["MSE (Ridge)", f"{summary['mse_ridge']:.3e}"],
        ["MSE (LightGBM)", f"{summary['mse_lgb']:.3e}"],
        ["Per-fold LGB wins", f"{n_lgb_wins} / {n_folds}"],
        ["Decision (5\\,\\%)", decision],
    ]
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        f"\\caption{{Diebold-Mariano comparison of Ridge vs LightGBM on the "
        f"pre-analysis pooled out-of-sample predictions, target = "
        f"\\texttt{{{target.replace('_', '\\_')}}}, horizon = {horizon}\\,s. "
        f"A significantly positive statistic establishes the non-linearity "
        f"claim; a non-significant result invalidates it.}}",
        f"\\label{{tab:ridge_vs_lgb_dm_h{horizon}}}",
        "\\begin{tabular}{lr}",
        "\\hline",
    ]
    for left, right in rows:
        lines.append(f"{left} & {right} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ridge vs LightGBM Diebold-Mariano comparison."
    )
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument(
        "--horizon", type=int, default=10, choices=LABEL_HORIZONS_S,
        help="label horizon in seconds (default 10)",
    )
    ap.add_argument(
        "--target", default="return_micro", choices=TARGET_FAMILIES,
        help="target family; column read is <target>_<horizon>",
    )
    ap.add_argument(
        "--lgb-out-dir", default=None,
        help="LightGBM run dir; defaults to <base>/models/lgbm_pre_analysis_<target>_h<h>",
    )
    ap.add_argument(
        "--ridge-out-dir", default=None,
        help="Ridge run dir; defaults to <base>/models/ridge_preanalysis_<target>_h<h>",
    )
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()

    base = Path(args.base)
    lgb_dir   = Path(args.lgb_out_dir   or base / "models" / f"lgbm_pre_analysis_{args.target}_h{args.horizon}")
    ridge_dir = Path(args.ridge_out_dir or base / "models" / f"ridge_preanalysis_{args.target}_h{args.horizon}")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    lgb_pred   = _load_predictions(lgb_dir   / f"oos_predictions_h{args.horizon}.parquet")
    ridge_pred = _load_predictions(ridge_dir / f"oos_predictions_h{args.horizon}.parquet")

    print(f"Ridge vs LightGBM DM | target={args.target} horizon={args.horizon}s")
    print(f"   Ridge:    {ridge_pred.height:,} rows  ({ridge_dir})")
    print(f"   LightGBM: {lgb_pred.height:,}   rows  ({lgb_dir})")

    joined = _join_aligned(ridge_pred, lgb_pred)
    n = joined.height
    print(f"   Joined:   {n:,} rows on (test_day, ts_ms)")

    y_true    = joined["y_true_lgb"].to_numpy()
    y_ridge   = joined["y_pred_ridge"].to_numpy()
    y_lgb     = joined["y_pred_lgb"].to_numpy()

    # a=ridge, b=lgb so positive stat means LGB wins (lower loss).
    res = diebold_mariano_test(y_true, y_ridge, y_lgb, horizon=args.horizon)

    mse_ridge = float(np.mean((y_true - y_ridge) ** 2))
    mse_lgb   = float(np.mean((y_true - y_lgb)   ** 2))

    if res.p_value < 0.05 and res.statistic > 0:
        decision = "LightGBM significantly outperforms Ridge"
    elif res.p_value < 0.05 and res.statistic < 0:
        decision = "Ridge significantly outperforms LightGBM"
    else:
        decision = "No significant difference"

    per_fold = _per_fold_metrics(joined)
    n_lgb_wins = sum(1 for r in per_fold if r["lgb_wins"])

    schwert = int(np.floor(4 * (n / 100) ** (2 / 9)))
    hac_nlags = max(args.horizon - 1, schwert)

    summary = {
        "target": args.target,
        "horizon_s": args.horizon,
        "n_observations": n,
        "n_folds": len(per_fold),
        "n_folds_lgb_wins": n_lgb_wins,
        "dm_statistic": res.statistic,
        "p_value":     res.p_value,
        "mean_squared_loss_diff": res.mean_diff,  # >0 = LGB has lower MSE
        "hac_std_err": res.hac_std_err,
        "hac_nlags":   hac_nlags,
        "mse_ridge":   mse_ridge,
        "mse_lgb":     mse_lgb,
        "lgb_superior_at_5pct": bool(res.p_value < 0.05 and res.statistic > 0),
        "decision":    decision,
        "per_fold":    per_fold,
        "ridge_out_dir": str(ridge_dir),
        "lgb_out_dir":   str(lgb_dir),
    }

    json_path  = reports_dir / f"ridge_vs_lgb_dm_h{args.horizon}.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    latex_path = reports_dir / f"tab_ridge_vs_lgb_dm_h{args.horizon}.tex"
    _write_latex(summary, per_fold, latex_path, args.horizon, args.target)

    per_day = _per_test_day_metrics(joined)
    regime = _regime_split(joined, per_day)
    perday_df = pl.DataFrame(per_day)
    perday_csv = reports_dir / f"preanalysis_perday_{args.target}_h{args.horizon}.csv"
    perday_df.write_csv(perday_csv)
    regime_json = reports_dir / f"preanalysis_regime_{args.target}_h{args.horizon}.json"
    regime_json.write_text(json.dumps(regime, indent=2, default=str))

    print()
    print(f"   DM statistic   = {res.statistic:+.3f}")
    print(f"   DM p-value     = {res.p_value:.3e}")
    print(f"   HAC lags       = {hac_nlags}")
    print(f"   MSE Ridge      = {mse_ridge:.3e}")
    print(f"   MSE LightGBM   = {mse_lgb:.3e}")
    print(f"   Per-fold wins  = LGB {n_lgb_wins} / {len(per_fold)}")
    print(f"\n   {decision}.")

    print("\n   Per-test-day R² (target_std flags the regime):")
    for d in per_day:
        spike = "  <-- 5 Feb spike" if d["test_day"] == "2026-02-05" else ""
        print(f"     {d['test_day']}  σ_tgt={d['target_std']:7.3f}  "
              f"R² ridge={d['r2_ridge']:+.4f}  lgb={d['r2_lgb']:+.4f}  "
              f"Δ={d['dr2_lgb_minus_ridge']:+.4f}{spike}")
    hv, cm = regime["high_vol"], regime["calm"]
    print(f"\n   Regime split at median σ_tgt={regime['median_target_std']:.3f}:")
    print(f"     high-vol ({hv['n_days']}d): R² ridge={hv['r2_ridge']:+.4f}  "
          f"lgb={hv['r2_lgb']:+.4f}  Δ={hv['dr2_lgb_minus_ridge']:+.4f}")
    print(f"     calm     ({cm['n_days']}d): R² ridge={cm['r2_ridge']:+.4f}  "
          f"lgb={cm['r2_lgb']:+.4f}  Δ={cm['dr2_lgb_minus_ridge']:+.4f}")

    print(f"\n   Wrote {json_path}")
    print(f"   Wrote {latex_path}")
    print(f"   Wrote {perday_csv}")
    print(f"   Wrote {regime_json}")


if __name__ == "__main__":
    main()
