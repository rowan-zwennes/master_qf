"""Ridge linear baseline for alpha_ML."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from sklearn.linear_model import Ridge

from ml_train import (
    DEFAULT_BASE,
    FEATURE_COLS,
    LABEL_HORIZONS_S,
    PRE_ANALYSIS_WINDOW,
    TARGET_FAMILIES,
    DayCache,
    _effective_purge,
    _load_preanalysis_tau,
    assert_purge_invariant,
    build_outer_folds,
    load_split_days,
    make_inner_folds,
    measure_latency,
    regression_metrics,
    resolve_feature_cols,
    train_max_ts_ms,
)

DEFAULT_ALPHAS = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]


RIDGE_DROP_FEATURES: list[str] = [
    "signed_vol_1s",
    "signed_vol_5s",
    "signed_vol_30s",
    "dobi_l1_1s",
    "trade_intensity_accel",
    "micro_mom_accel",
]


def fit_preprocessor(
    X: np.ndarray,
    winsorize_pct: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive (optional) winsorisation bounds, imputation medians, and standardisation (mean."""
    X = np.where(np.isinf(X), np.nan, X)   # treat ±inf as missing, impute as NaN
    n_features = X.shape[1]
    if winsorize_pct is not None:
        if not (50.0 < winsorize_pct < 100.0):
            raise ValueError(
                f"winsorize_pct must be in (50, 100); got {winsorize_pct}"
            )
        lo = np.nanpercentile(X, 100.0 - winsorize_pct, axis=0)
        hi = np.nanpercentile(X, winsorize_pct, axis=0)
        lo = np.where(np.isfinite(lo), lo, -np.inf)
        hi = np.where(np.isfinite(hi), hi, np.inf)
        X = np.clip(X, lo, hi)  # NaN survives np.clip
    else:
        lo = np.full(n_features, -np.inf, dtype=np.float64)
        hi = np.full(n_features, np.inf, dtype=np.float64)

    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    Xi = np.where(np.isnan(X), medians, X)
    means = Xi.mean(axis=0)
    stds = Xi.std(axis=0)
    stds = np.where(stds > 0.0, stds, 1.0)
    return lo, hi, medians, means, stds


def apply_preprocessor(
    X: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Map ±inf to missing, clip to [lo, hi], impute with `medians`, then
    standardise. The inf->NaN step mirrors fit_preprocessor so train and
    inference treat degenerate values identically."""
    Xf = np.where(np.isinf(X), np.nan, X)  # treat ±inf as missing (Ridge cannot ingest inf)
    Xc = np.clip(Xf, lo, hi)  # NaN survives np.clip
    Xi = np.where(np.isnan(Xc), medians, Xc)
    return (Xi - means) / stds


def select_lambda_via_inner_cv(
    train_days: list[str],
    cache: DayCache,
    alphas: list[float],
    purge_sec: int,
    *,
    winsorize_pct: float | None = None,
) -> tuple[float, dict[float, float]]:
    """Run the inner expanding 3-fold CV inside `train_days` and pick the ridge penalty."""
    inner_folds = make_inner_folds(train_days)
    # per-alpha list of per-fold RMSEs
    per_alpha: dict[float, list[float]] = {a: [] for a in alphas}

    for ifold in inner_folds:
        seam_day = ifold["val_days"][0]
        seam_max_ts = train_max_ts_ms(seam_day, purge_sec)
        Xi_tr, yi_tr, ti_tr = cache.stack(ifold["train_days"], max_ts_ms=seam_max_ts)
        Xi_va, yi_va, _ = cache.stack(ifold["val_days"])
        assert_purge_invariant(ti_tr, seam_day, cache.horizon, purge_sec)
        if Xi_tr.shape[0] == 0 or Xi_va.shape[0] == 0:
            continue
        lo, hi, med, mean, std = fit_preprocessor(Xi_tr, winsorize_pct)
        Xi_tr_p = apply_preprocessor(Xi_tr, lo, hi, med, mean, std)
        Xi_va_p = apply_preprocessor(Xi_va, lo, hi, med, mean, std)
        for a in alphas:
            ridge = Ridge(alpha=a)
            ridge.fit(Xi_tr_p, yi_tr)
            v = ridge.predict(Xi_va_p)
            per_alpha[a].append(float(np.sqrt(np.mean((v - yi_va) ** 2))))

    mean_val_rmse = {
        a: (float(np.mean(rmses)) if rmses else float("inf"))
        for a, rmses in per_alpha.items()
    }
    best_alpha = min(mean_val_rmse, key=mean_val_rmse.get)  # type: ignore[arg-type]
    return best_alpha, mean_val_rmse


def train_ridge_fold(
    fold: dict,
    cache: DayCache,
    alphas: list[float],
    purge_sec: int,
    *,
    winsorize_pct: float | None = None,
) -> dict | None:
    """Train and score one outer walk-forward fold with a Ridge regressor.

    `purge_sec` is the pre-analysis tau loaded from purge_window.json.
    Ridge only runs in pre-analysis mode, so this is fixed for the whole run.
    """
    seam_day = fold["test_day"]
    seam_max_ts = train_max_ts_ms(seam_day, purge_sec)
    Xtr, ytr, ttr = cache.stack(fold["train_days"], max_ts_ms=seam_max_ts)
    Xte, yte, tste = cache.stack([seam_day])
    assert_purge_invariant(ttr, seam_day, cache.horizon, purge_sec)

    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        print(
            f"   fold {fold['fold']:02d}  skipped "
            f"(train={Xtr.shape[0]} test={Xte.shape[0]})"
        )
        return None

    t_start = time.perf_counter()

    # --- inner CV picks lambda ---
    best_alpha, per_alpha_rmse = select_lambda_via_inner_cv(
        fold["train_days"], cache, alphas, purge_sec=purge_sec,
        winsorize_pct=winsorize_pct,
    )
    if best_alpha in (alphas[0], alphas[-1]) and len(alphas) > 1:
        print(f"   fold {fold['fold']:02d}  selected lambda={best_alpha:g} "
              f"is at a grid boundary, consider widening --alphas")

    lo_f, hi_f, med_f, mean_f, std_f = fit_preprocessor(Xtr, winsorize_pct)
    Xtr_p = apply_preprocessor(Xtr, lo_f, hi_f, med_f, mean_f, std_f)
    Xte_p = apply_preprocessor(Xte, lo_f, hi_f, med_f, mean_f, std_f)

    model = Ridge(alpha=best_alpha)
    model.fit(Xtr_p, ytr)
    fit_seconds = time.perf_counter() - t_start

    y_pred = model.predict(Xte_p)
    metrics = regression_metrics(yte, y_pred)
    latency = measure_latency(model, Xte_p)

    return {
        "fold": fold["fold"],
        "best_alpha": best_alpha,
        "per_alpha_rmse": per_alpha_rmse,
        "fit_seconds": fit_seconds,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "test_day": fold["test_day"],
        "metrics": metrics,
        "latency": latency,
        "coef": model.coef_.astype(np.float64),
        "intercept": float(model.intercept_),
        "pred_ts_ms": tste,
        "y_true": yte,
        "y_pred": y_pred,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--horizon", type=int, default=10, choices=LABEL_HORIZONS_S,
                   help="label horizon in seconds (default 10)")
    p.add_argument("--target", default="return_micro", choices=TARGET_FAMILIES,
                   help="target family; column read is <target>_<horizon>")
    p.add_argument("--max-folds", type=int, default=None,
                   help="cap the number of outer folds (debugging)")
    p.add_argument("--out-dir", default=None,
                   help="output root (default <base>/models)")
    p.add_argument("--keep-stale", action="store_true",
                   help="keep rows where the mark price was stale "
                        "(default: drop them)")
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="ridge penalty grid (default: wide log grid)")
    p.add_argument("--winsorize-pct", type=float, default=None,
                   help="optional robustness check")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.winsorize_pct is not None and not (50.0 < args.winsorize_pct < 100.0):
        raise SystemExit(
            f"--winsorize-pct must be in (50, 100); got {args.winsorize_pct}"
        )

    base = Path(args.base)
    preanalysis_tau: int = _load_preanalysis_tau(base)
    out_root = Path(args.out_dir) if args.out_dir else (base / "models")
    target_col = f"{args.target}_{args.horizon}"
    winsor_tag = (
        f"_winsor{args.winsorize_pct:g}".replace(".", "p")
        if args.winsorize_pct is not None
        else ""
    )
    label_stem = "labels"
    feature_allowlist = FEATURE_COLS
    out_dir = (
        out_root
        / f"ridge_preanalysis_{args.target}_h{args.horizon}{winsor_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = sorted(args.alphas) if args.alphas else list(DEFAULT_ALPHAS)

    all_days, n_pre_analysis = load_split_days(base)
    lgbm_feature_cols = resolve_feature_cols(
        base, all_days[0], feature_cols=feature_allowlist, stem=label_stem
    )
    ridge_dropped = [c for c in RIDGE_DROP_FEATURES if c in lgbm_feature_cols]
    feature_cols = [c for c in lgbm_feature_cols if c not in RIDGE_DROP_FEATURES]
    folds = build_outer_folds(
        all_days,
        mode="pre_analysis",
        n_pre_analysis=n_pre_analysis,
        max_folds=args.max_folds,
    )

    print(f"Ridge baseline (pre-analysis) | target={target_col}")
    print(f"   {len(all_days)} funding days "
          f"(pre_analysis={n_pre_analysis}), {len(folds)} outer folds, "
          f"window={PRE_ANALYSIS_WINDOW}")
    print(f"   {len(feature_cols)} features "
          f"(dropped {len(ridge_dropped)} exact-collinearity columns: "
          f"{ridge_dropped if ridge_dropped else 'none'}), "
          f"lambda grid={alphas}")
    print(
        f"   purge: fixed {_effective_purge(preanalysis_tau)}s at every "
        f"train->test / train->val seam "
        f"(pre-analysis tau={preanalysis_tau}s from purge_window.json, "
        f"label-horizon floor={max(LABEL_HORIZONS_S)}s)"
    )
    if args.winsorize_pct is not None:
        print(f"   winsorisation: pct={args.winsorize_pct:g} "
              f"(robustness run)")
    else:
        print("   winsorisation: off (headline DM run)")
    print(f"   out -> {out_dir}")

    cache = DayCache(
        base, feature_cols, target_col, args.horizon,
        drop_stale=not args.keep_stale, label_stem=label_stem,
    )

    metrics_rows: list[dict] = []
    coef_rows: list[dict] = []
    inner_cv_rows: list[dict] = []
    pred_frames: list[pl.DataFrame] = []
    coef_acc = np.zeros(len(feature_cols), dtype=np.float64)
    coef_sq_acc = np.zeros(len(feature_cols), dtype=np.float64)
    n_scored = 0

    wall0 = time.perf_counter()
    for fold in folds:
        res = train_ridge_fold(
            fold, cache, alphas, purge_sec=preanalysis_tau,
            winsorize_pct=args.winsorize_pct,
        )
        if res is None:
            continue
        n_scored += 1
        f = fold["fold"]
        first_test = fold["test_day"]

        m, lat = res["metrics"], res["latency"]
        metrics_rows.append({
            "fold": f,
            "test_day": first_test,
            "n_train": res["n_train"],
            "n_test": res["n_test"],
            "best_alpha": res["best_alpha"],
            "intercept": res["intercept"],
            "rmse": m["rmse"],
            "r2": m["r2"],
            "dir_acc": m["dir_acc"],
            "y_true_mean": m["y_true_mean"],
            "y_true_std": m["y_true_std"],
            "y_pred_mean": m["y_pred_mean"],
            "y_pred_std": m["y_pred_std"],
            "predict_batch_ns_per_row": lat["batch_ns_per_row"],
            "predict_single_p50_ns": lat["single_p50_ns"],
            "predict_single_p99_ns": lat["single_p99_ns"],
            "fit_seconds": res["fit_seconds"],
        })

        for feat, c in zip(feature_cols, res["coef"]):
            coef_rows.append({
                "fold": f,
                "test_day": first_test,
                "feature": feat,
                "coef_standardized": float(c),
            })

        for a, rmse in res["per_alpha_rmse"].items():
            inner_cv_rows.append({
                "fold": f,
                "test_day": first_test,
                "alpha": a,
                "mean_val_rmse": rmse,
                "selected": (a == res["best_alpha"]),
            })

        pred_frames.append(pl.DataFrame({
            "ts_ms": res["pred_ts_ms"],
            "fold": np.full(res["n_test"], f, dtype=np.int32),
            "test_day": np.full(res["n_test"], first_test),
            "y_true": res["y_true"],
            "y_pred": res["y_pred"],
        }))

        coef_acc += res["coef"]
        coef_sq_acc += res["coef"] ** 2

        print(
            f"   fold {f:02d} [{first_test}] R2={m['r2']:+.5f} "
            f"dir={m['dir_acc']:.4f} rmse={m['rmse']:.3e} "
            f"lambda={res['best_alpha']:g} fit={res['fit_seconds']:.1f}s"
        )

    if n_scored == 0:
        raise SystemExit("no fold produced a model, check data availability")

    # --- write metrics CSV ---
    pl.DataFrame(metrics_rows).write_csv(out_dir / f"metrics_h{args.horizon}.csv")

    # --- write per-fold coefficients (long format) ---
    pl.DataFrame(coef_rows).write_csv(out_dir / f"coefficients_h{args.horizon}.csv")

    # --- write fold-averaged coefficient summary ---
    coef_mean = coef_acc / n_scored
    coef_std = np.sqrt(np.maximum(coef_sq_acc / n_scored - coef_mean**2, 0.0))
    coef_summary = pl.DataFrame({
        "feature": feature_cols,
        "coef_mean": coef_mean,
        "coef_std": coef_std,
        "abs_coef_mean": np.abs(coef_mean),
    }).sort("abs_coef_mean", descending=True)
    coef_summary.write_csv(out_dir / f"coefficient_summary_h{args.horizon}.csv")

    # --- write inner-CV diagnostics ---
    pl.DataFrame(inner_cv_rows).write_csv(
        out_dir / f"inner_cv_h{args.horizon}.csv"
    )

    # --- write concatenated out-of-sample predictions ---
    oos = pl.concat(pred_frames, how="vertical").sort("ts_ms")
    oos.write_parquet(
        out_dir / f"oos_predictions_h{args.horizon}.parquet", compression="zstd"
    )

    # --- write run config ---
    import sklearn
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "model": "sklearn.linear_model.Ridge",
        "args": vars(args),
        "target_col": target_col,
        "mode": "pre_analysis",
        "pre_analysis_window": PRE_ANALYSIS_WINDOW,
        "n_folds": n_scored,
        "alpha_grid": alphas,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "ridge_dropped_features": ridge_dropped,
        "lgbm_feature_cols": lgbm_feature_cols,
        "purge_strategy": "fixed_preanalysis",
        "preanalysis_tau_sec": preanalysis_tau,
        "effective_purge_sec": _effective_purge(preanalysis_tau),
        "label_horizon_floor_sec": max(LABEL_HORIZONS_S),
        "winsorize_pct": args.winsorize_pct,
    }
    (out_dir / f"run_config_h{args.horizon}.json").write_text(
        json.dumps(config, indent=2, default=str)
    )

    # --- pooled out-of-sample summary ---
    pooled = regression_metrics(
        oos["y_true"].to_numpy(), oos["y_pred"].to_numpy()
    )
    r2_vals = [r["r2"] for r in metrics_rows if r["r2"] is not None]
    da_vals = [r["dir_acc"] for r in metrics_rows if r["dir_acc"] is not None]
    wall = time.perf_counter() - wall0

    print(f"\n{n_scored} Ridge folds trained in {wall:.0f}s")
    print(f"   pooled OOS: R2={pooled['r2']:+.5f} "
          f"dir_acc={pooled['dir_acc']:.4f} rmse={pooled['rmse']:.3e} "
          f"n={pooled['n']}")
    if r2_vals:
        print(f"   per-fold R2: mean={np.mean(r2_vals):+.5f} "
              f"median={np.median(r2_vals):+.5f} "
              f"min={np.min(r2_vals):+.5f} max={np.max(r2_vals):+.5f}")
    if da_vals:
        print(f"   per-fold dir_acc: mean={np.mean(da_vals):.4f}")
    top5 = coef_summary.head(5)["feature"].to_list()
    print(f"   top-5 features by |coef|: {top5}")
    print(f"   artifacts -> {out_dir}")
    print(f"   compare against `ml_train.py --mode pre_analysis` "
          f"oos_predictions_h{args.horizon}.parquet via statistical_tests.py "
          f"(Diebold-Mariano on squared-error losses).")


if __name__ == "__main__":
    main()
