#!/usr/bin/env python3
"""L2 versus Huber training-loss comparison."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from statistical_tests import diebold_mariano_test
from ml_train import DEFAULT_BASE, LABEL_HORIZONS_S, TARGET_FAMILIES


def _load_predictions(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"oos predictions parquet not found: {path}\n"
            f"  Run ml_train.py for this (objective, target, horizon) first."
        )
    df = pl.read_parquet(path)
    required = {"ts_ms", "fold", "test_day", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")
    return df


def _join_aligned(l2: pl.DataFrame, huber: pl.DataFrame) -> pl.DataFrame:
    """Inner-join on (test_day, ts_ms) and verify y_true agrees. A mismatch
    means the two runs scored different rows (different fold layout / row
    filter) and the comparison would be invalid."""
    joined = l2.rename({"y_pred": "y_pred_l2", "y_true": "y_true_l2"}).join(
        huber.rename({"y_pred": "y_pred_huber", "y_true": "y_true_huber"}),
        on=["test_day", "ts_ms"],
        how="inner",
    )
    if joined.is_empty():
        raise SystemExit(
            "join produced zero rows, the two prediction parquets share no "
            "(test_day, ts_ms) keys. Re-run both at the same target/horizon/split."
        )
    mismatch = (
        (joined["y_true_l2"] - joined["y_true_huber"]).abs() > 1e-7
    ).sum()
    if mismatch:
        raise SystemExit(
            f"y_true disagrees on {mismatch} row(s) between the two parquets "
            f"the comparison would be invalid. Check that the L2 and Huber "
            f"runs used identical fold layout / row filter (same target, "
            f"horizon, split, feature set)."
        )
    return joined


def _ic(y: np.ndarray, p: np.ndarray) -> float:
    """Pearson information coefficient (correlation of prediction and target)."""
    if np.std(p) <= 0.0 or np.std(y) <= 0.0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def _rank_ic(y: np.ndarray, p: np.ndarray) -> float:
    if np.std(p) <= 0.0 or np.std(y) <= 0.0:
        return float("nan")
    return float(sp_stats.spearmanr(y, p).statistic)


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    """Out-of-sample R^2 against the realised-mean benchmark."""
    sstot = float(np.sum((y - y.mean()) ** 2))
    if sstot <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y - p) ** 2)) / sstot


def _dir_acc(y: np.ndarray, p: np.ndarray) -> float:
    """Directional accuracy on rows with a non-zero realised move; a zero
    target has no direction to predict, so sign(0) rows are excluded (this
    matches the per-fold dir_acc reported in metrics_h<h>.csv)."""
    nz = y != 0.0
    if nz.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(p[nz]) == np.sign(y[nz])))


def _per_fold_metrics(joined: pl.DataFrame) -> list[dict]:
    rows = []
    fold_col = "fold" if "fold" in joined.columns else "fold_right"
    for fold_id, sub in joined.group_by(fold_col, maintain_order=True):
        fid = int(fold_id[0] if isinstance(fold_id, tuple) else fold_id)
        y = sub["y_true_l2"].to_numpy()
        a = sub["y_pred_l2"].to_numpy()
        b = sub["y_pred_huber"].to_numpy()
        mse_l2 = float(np.mean((y - a) ** 2))
        mse_huber = float(np.mean((y - b) ** 2))
        rows.append({
            "fold": fid,
            "n": int(sub.height),
            "mse_l2": mse_l2,
            "mse_huber": mse_huber,
            # "huber wins" if its per-fold MSE is strictly smaller than L2's.
            "huber_wins": bool(mse_huber < mse_l2),
        })
    rows.sort(key=lambda r: r["fold"])
    return rows


def _format_p_for_latex(p: float) -> str:
    if not math.isfinite(p):
        return "--"
    if p < 1e-10:
        return "$<10^{-10}$"
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        return f"$<10^{{{exp}}}$"
    return f"{p:.4f}"


def _write_latex(summary: dict, per_fold: list[dict], out_path: Path,
                 horizon: int, target: str) -> None:
    n_huber_wins = sum(1 for r in per_fold if r["huber_wins"])
    n_folds = len(per_fold)
    rows = [
        ["Test", "Diebold-Mariano (squared-error loss)"],
        ["Sign", "$d_t = (y - \\hat{y}_{\\text{L2}})^2 - (y - \\hat{y}_{\\text{Huber}})^2$;  $\\bar{d} > 0 \\Rightarrow$ Huber lower MSE"],
        ["DM statistic", f"{summary['dm_statistic']:+.3f}"],
        ["DM $p$-value", _format_p_for_latex(summary["p_value"])],
        ["HAC lag", f"{summary['hac_nlags']}"],
        ["$n$ observations", f"{summary['n_observations']:,}"],
        ["RMSE (L2)", f"{summary['rmse_l2']:.4f}"],
        ["RMSE (Huber)", f"{summary['rmse_huber']:.4f}"],
        ["$\\Delta$RMSE (L2 $-$ Huber)", f"{summary['rmse_delta_l2_minus_huber']:+.4f}"],
        ["$R^2$ (L2)", f"{summary['r2_l2']:+.5f}"],
        ["$R^2$ (Huber)", f"{summary['r2_huber']:+.5f}"],
        ["$\\Delta R^2$ (L2 $-$ Huber)", f"{summary['r2_delta_l2_minus_huber']:+.5f}"],
        ["Dir.\\ accuracy (L2)", f"{summary['dir_acc_l2']:.4f}"],
        ["Dir.\\ accuracy (Huber)", f"{summary['dir_acc_huber']:.4f}"],
        ["$\\Delta$Dir.\\ acc.\\ (L2 $-$ Huber)", f"{summary['dir_acc_delta_l2_minus_huber']:+.4f}"],
        ["IC (L2)", f"{summary['ic_l2']:.4f}"],
        ["IC (Huber)", f"{summary['ic_huber']:.4f}"],
        ["$\\Delta$IC (L2 $-$ Huber)", f"{summary['ic_delta_l2_minus_huber']:+.4f}"],
        ["Rank-IC (L2)", f"{summary['rank_ic_l2']:.4f}"],
        ["Rank-IC (Huber)", f"{summary['rank_ic_huber']:.4f}"],
        ["Per-fold Huber wins", f"{n_huber_wins} / {n_folds}"],
        ["Decision (5\\,\\%)", summary["decision"]],
    ]
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        f"\\caption{{Training-loss robustness check: L2 vs Huber on the "
        f"pre-analysis pooled out-of-sample predictions at the headline "
        f"horizon, target = \\texttt{{{target.replace('_', '\\_')}}}, "
        f"horizon = {horizon}\\,s. Both models are trained end-to-end under "
        f"the same compute budget; only the training objective differs "
        f"(Huber $\\delta$ set to the median RMSE of the L2 run). A "
        f"non-significant DM statistic and near-zero IC/RMSE deltas show the "
        f"headline result is robust to the loss choice.}}",
        f"\\label{{tab:l2_vs_huber_dm_h{horizon}}}",
        "\\begin{tabular}{lr}",
        "\\hline",
    ]
    for left, right in rows:
        lines.append(f"{left} & {right} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="L2 vs Huber training-loss robustness comparison."
    )
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--horizon", type=int, default=10, choices=LABEL_HORIZONS_S)
    ap.add_argument("--target", default="drift_mid", choices=TARGET_FAMILIES)
    ap.add_argument(
        "--l2-dir", default=None,
        help="L2 run dir; defaults to <base>/models/lgbm_pre_analysis_<target>_h<h>",
    )
    ap.add_argument(
        "--huber-dir", default=None,
        help="Huber run dir; defaults to <base>/models/lgbm_pre_analysis_<target>_h<h>_huber",
    )
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()

    base = Path(args.base)
    l2_dir    = Path(args.l2_dir    or base / "models" / f"lgbm_pre_analysis_{args.target}_h{args.horizon}")
    huber_dir = Path(args.huber_dir or base / "models" / f"lgbm_pre_analysis_{args.target}_h{args.horizon}_huber")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    l2_pred    = _load_predictions(l2_dir    / f"oos_predictions_h{args.horizon}.parquet")
    huber_pred = _load_predictions(huber_dir / f"oos_predictions_h{args.horizon}.parquet")

    print(f"L2 vs Huber DM | target={args.target} horizon={args.horizon}s")
    print(f"   L2:    {l2_pred.height:,} rows  ({l2_dir})")
    print(f"   Huber: {huber_pred.height:,} rows  ({huber_dir})")

    joined = _join_aligned(l2_pred, huber_pred)
    n = joined.height
    print(f"   Joined: {n:,} rows on (test_day, ts_ms)")

    y     = joined["y_true_l2"].to_numpy()
    p_l2  = joined["y_pred_l2"].to_numpy()
    p_hub = joined["y_pred_huber"].to_numpy()

    # a=L2, b=Huber so positive stat means Huber wins (lower loss).
    res = diebold_mariano_test(y, p_l2, p_hub, horizon=args.horizon)

    rmse_l2  = float(np.sqrt(np.mean((y - p_l2)  ** 2)))
    rmse_hub = float(np.sqrt(np.mean((y - p_hub) ** 2)))
    ic_l2,  ic_hub  = _ic(y, p_l2),      _ic(y, p_hub)
    ric_l2, ric_hub = _rank_ic(y, p_l2), _rank_ic(y, p_hub)
    r2_l2,  r2_hub  = _r2(y, p_l2),      _r2(y, p_hub)
    da_l2,  da_hub  = _dir_acc(y, p_l2), _dir_acc(y, p_hub)

    if res.p_value < 0.05 and res.statistic > 0:
        decision = "Huber significantly outperforms L2"
    elif res.p_value < 0.05 and res.statistic < 0:
        decision = "L2 significantly outperforms Huber"
    else:
        decision = "No significant difference (L2 specification robust)"

    per_fold = _per_fold_metrics(joined)
    n_huber_wins = sum(1 for r in per_fold if r["huber_wins"])

    schwert = int(np.floor(4 * (n / 100) ** (2 / 9)))
    hac_nlags = max(args.horizon - 1, schwert)

    summary = {
        "target": args.target,
        "horizon_s": args.horizon,
        "n_observations": n,
        "n_folds": len(per_fold),
        "n_folds_huber_wins": n_huber_wins,
        "dm_statistic": res.statistic,
        "p_value": res.p_value,
        "mean_squared_loss_diff": res.mean_diff,  # >0 = Huber lower MSE
        "hac_std_err": res.hac_std_err,
        "hac_nlags": hac_nlags,
        "rmse_l2": rmse_l2,
        "rmse_huber": rmse_hub,
        "rmse_delta_l2_minus_huber": rmse_l2 - rmse_hub,
        "r2_l2": r2_l2,
        "r2_huber": r2_hub,
        "r2_delta_l2_minus_huber": r2_l2 - r2_hub,
        "dir_acc_l2": da_l2,
        "dir_acc_huber": da_hub,
        "dir_acc_delta_l2_minus_huber": da_l2 - da_hub,
        "ic_l2": ic_l2,
        "ic_huber": ic_hub,
        "ic_delta_l2_minus_huber": ic_l2 - ic_hub,
        "rank_ic_l2": ric_l2,
        "rank_ic_huber": ric_hub,
        "rank_ic_delta_l2_minus_huber": ric_l2 - ric_hub,
        "huber_superior_at_5pct": bool(res.p_value < 0.05 and res.statistic > 0),
        "decision": decision,
        "per_fold": per_fold,
        "l2_dir": str(l2_dir),
        "huber_dir": str(huber_dir),
    }

    json_path  = reports_dir / f"l2_vs_huber_dm_h{args.horizon}.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    latex_path = reports_dir / f"tab_l2_vs_huber_dm_h{args.horizon}.tex"
    _write_latex(summary, per_fold, latex_path, args.horizon, args.target)

    print()
    print(f"   DM statistic = {res.statistic:+.3f}  (>0 => Huber lower MSE)")
    print(f"   DM p-value   = {res.p_value:.3e}")
    print(f"   HAC lags     = {hac_nlags}")
    print(f"   RMSE   L2 = {rmse_l2:.4f}   Huber = {rmse_hub:.4f}   "
          f"(L2-Huber = {rmse_l2 - rmse_hub:+.4f})")
    print(f"   R2     L2 = {r2_l2:+.5f}  Huber = {r2_hub:+.5f}  "
          f"(L2-Huber = {r2_l2 - r2_hub:+.5f})")
    print(f"   dirAcc L2 = {da_l2:.4f}   Huber = {da_hub:.4f}   "
          f"(L2-Huber = {da_l2 - da_hub:+.4f})")
    print(f"   IC     L2 = {ic_l2:.4f}   Huber = {ic_hub:.4f}   "
          f"(L2-Huber = {ic_l2 - ic_hub:+.4f})")
    print(f"   rankIC L2 = {ric_l2:.4f}   Huber = {ric_hub:.4f}   "
          f"(L2-Huber = {ric_l2 - ric_hub:+.4f})")
    print(f"   Per-fold Huber wins = {n_huber_wins} / {len(per_fold)}")
    print(f"   Decision = {decision}")
    print(f"   wrote {json_path}")
    print(f"   wrote {latex_path}")


if __name__ == "__main__":
    main()
