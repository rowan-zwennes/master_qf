"""Pre-analysis data-quality diagnostics."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
TARGET = "drift_mid"
ALL_HORIZONS = [1, 5, 10, 30]

TUNED_PARAMS = [
    "tuned_num_leaves",
    "tuned_learning_rate",
    "tuned_min_data_in_leaf",
    "tuned_feature_fraction",
    "tuned_bagging_fraction",
    "tuned_lambda_l2",
]

BOUNDARY_ZONE_FRAC = 0.10
BOUNDARY_HUG_THRESHOLD = 0.50


@dataclass
class HorizonData:
    h: int
    metrics: pl.DataFrame
    inner_cv: pl.DataFrame
    search_space: dict[str, dict]   # key = bare param name, e.g. "num_leaves"
    max_boost_rounds: int
    lgb_params_base: dict


def load_horizon(base: Path, h: int) -> HorizonData | None:
    model_dir = base / "models" / f"lgbm_pre_analysis_{TARGET}_h{h}"
    if not model_dir.exists():
        print(f"  h={h:2d}s   directory not found: {model_dir}, skipping")
        return None

    metrics  = pl.read_csv(model_dir / f"metrics_h{h}.csv")
    inner_cv = pl.read_csv(model_dir / f"inner_cv_h{h}.csv")
    config   = json.loads((model_dir / f"run_config_h{h}.json").read_text())

    return HorizonData(
        h=h,
        metrics=metrics,
        inner_cv=inner_cv,
        search_space=config["optuna"]["search_space"],
        max_boost_rounds=config["max_boost_rounds"],
        lgb_params_base=config["lgb_params_base"],
    )


def _inner_bounds(lo: float, hi: float, is_log: bool) -> tuple[float, float]:
    """Return the inner boundary thresholds (outside = near-boundary zone)."""
    if is_log:
        log_lo, log_hi = math.log(lo), math.log(hi)
        margin = BOUNDARY_ZONE_FRAC * (log_hi - log_lo)
        return math.exp(log_lo + margin), math.exp(log_hi - margin)
    margin = BOUNDARY_ZONE_FRAC * (hi - lo)
    return lo + margin, hi - margin


def _is_near_boundary(val: float, lo: float, hi: float, is_log: bool) -> bool:
    inner_lo, inner_hi = _inner_bounds(lo, hi, is_log)
    return val < inner_lo or val > inner_hi


def check_fold_metrics(data: list[HorizonData], out_dir: Path) -> list[dict]:
    print("\n" + "=" * 72)
    print("CHECK 1: Fold-level metrics time series")
    print("=" * 72)

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    fig.suptitle(f"Pre-analysis fold-level metrics ({TARGET})", fontsize=12)

    summary: list[dict] = []

    for idx, hd in enumerate(data):
        m     = hd.metrics
        folds = m["fold"].to_numpy()
        r2    = m["r2"].to_numpy()
        dacc  = m["dir_acc"].to_numpy()
        crds  = m["calibrated_rounds"].to_numpy()
        c     = colours[idx % len(colours)]
        lbl   = f"h={hd.h}s"

        axes[0].plot(folds, r2,   marker="o", ms=4, lw=1.2, label=lbl, color=c)
        axes[1].plot(folds, dacc, marker="o", ms=4, lw=1.2, label=lbl, color=c)
        axes[2].plot(folds, crds, marker="o", ms=4, lw=1.2, label=lbl, color=c)

        print(f"\n  h={hd.h}s  (n_folds={len(m)})")
        print(f"  {'fold':>5} {'test_day':>12} {'R²':>9} {'RMSE':>9} "
              f"{'dir_acc':>8} {'cal_rds':>8}")
        for row in m.iter_rows(named=True):
            print(f"  {row['fold']:>5d} {str(row['test_day']):>12} "
                  f"{row['r2']:>9.5f} {row['rmse']:>9.4f} "
                  f"{row['dir_acc']:>8.4f} {row['calibrated_rounds']:>8d}")

        r2_mean = float(np.mean(r2))
        r2_std  = float(np.std(r2))
        print(f"  R2  mean={r2_mean:.5f}  std={r2_std:.5f}  "
              f"min={r2.min():.5f}  max={r2.max():.5f}")

        summary.append({
            "horizon_s": hd.h,
            "r2_mean": r2_mean,
            "r2_std": r2_std,
            "r2_min": float(r2.min()),
            "r2_max": float(r2.max()),
            "r2_cv_pct": r2_std / abs(r2_mean) * 100.0 if r2_mean != 0 else float("nan"),
            "dir_acc_mean": float(np.mean(dacc)),
            "dir_acc_std": float(np.std(dacc)),
            "cal_rounds_mean": float(np.mean(crds)),
            "cal_rounds_max": int(crds.max()),
            "cap_hit_frac": float(np.mean(crds >= hd.max_boost_rounds)),
        })

    axes[0].set_ylabel("R²")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("dir_acc")
    axes[1].axhline(0.5, ls="--", c="k", lw=0.8, alpha=0.7)
    axes[1].grid(alpha=0.3)
    axes[2].set_ylabel("calibrated rounds")
    axes[2].axhline(data[0].max_boost_rounds, ls=":", c="red", lw=1.0, alpha=0.7,
                    label=f"cap ({data[0].max_boost_rounds})")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    for ax in axes:
        ax.set_xlabel("fold index")

    plt.tight_layout()
    out = out_dir / "fig_fold_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  figure saved: {out}")
    return summary


def check_hyperparams(data: list[HorizonData], out_dir: Path) -> tuple[list[dict], list[dict]]:
    print("\n" + "=" * 72)
    print("CHECK 2: Hyperparameter stability  |  CHECK 3: Boundary audit")
    print("=" * 72)
    print(f"  Boundary zone = outer {BOUNDARY_ZONE_FRAC:.0%} of each param's search range "
          f"(log-scale for log-sampled params).")
    print(f"  Flag: ≥{BOUNDARY_HUG_THRESHOLD:.0%} of folds land in the boundary zone "
          "on ONE side (low or high).")

    stability_rows: list[dict] = []
    boundary_rows:  list[dict] = []

    for hd in data:
        cv = hd.inner_cv

        print(f"\n  h={hd.h}s  ({len(cv)} folds, all are retune folds in pre_analysis mode)")
        print(f"  {'param':<25} {'mean':>10} {'std':>10} {'CV%':>7} "
              f"{'min':>10} {'max':>10}  {'max_hug':>8}  {'flag'}")

        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        fig.suptitle(f"Hyperparameter trajectories, h={hd.h}s ({TARGET})", fontsize=11)
        axes_flat = axes.flatten()

        for pi, col in enumerate(TUNED_PARAMS):
            bare = col[len("tuned_"):]       # strip "tuned_" prefix to match search_space key
            label = bare                      # human label = bare name

            if col not in cv.columns:
                print(f"   {col} not in inner_cv, skipping")
                continue

            vals = cv[col].to_numpy().astype(float)
            folds = cv["fold"].to_numpy()

            mean_ = float(np.mean(vals))
            std_  = float(np.std(vals))
            cv_   = (std_ / abs(mean_) * 100.0) if mean_ != 0 else float("nan")
            min_  = float(vals.min())
            max_  = float(vals.max())

            ss = hd.search_space.get(bare)
            bound_frac = float("nan")
            flag = "n/a"
            n_low = n_high = None
            if ss is not None:
                lo, hi, is_log = ss["low"], ss["high"], ss.get("log", False)
                inner_lo, inner_hi = _inner_bounds(lo, hi, is_log)
                n_low = int(sum(v < inner_lo for v in vals))
                n_high = int(sum(v > inner_hi for v in vals))
                # Flag if EITHER side independently exceeds the threshold
                frac_low, frac_high = n_low / len(vals), n_high / len(vals)
                bound_frac = max(frac_low, frac_high)
                flag = " HUGGING" if bound_frac >= BOUNDARY_HUG_THRESHOLD else "ok"

            print(f"  {label:<25} {mean_:>10.4g} {std_:>10.4g} {cv_:>6.1f}% "
                  f"{min_:>10.4g} {max_:>10.4g}  {bound_frac:>7.1%}   {flag}")

            stability_rows.append({
                "horizon_s": hd.h, "param": label,
                "mean": mean_, "std": std_, "cv_pct": cv_,
                "min": min_, "max": max_,
            })

            boundary_rows.append({
                "horizon_s": hd.h, "param": label,
                "n_folds": len(vals),
                "n_near_low": n_low,
                "n_near_high": n_high,
                "max_side_frac": bound_frac,
                "search_low": ss["low"] if ss else None,
                "search_high": ss["high"] if ss else None,
                "log_scale": ss.get("log", False) if ss else None,
                "flag": flag,
            })

            # trajectory subplot
            ax = axes_flat[pi]
            ax.plot(folds, vals, color="steelblue", lw=1.2, marker="o", ms=3)
            if ss is not None:
                lo, hi, is_log = ss["low"], ss["high"], ss.get("log", False)
                inner_lo, inner_hi = _inner_bounds(lo, hi, is_log)
                ax.axhline(lo,       ls="--", c="red",    lw=0.9, alpha=0.8, label="search bounds")
                ax.axhline(hi,       ls="--", c="red",    lw=0.9, alpha=0.8)
                ax.axhline(inner_lo, ls=":",  c="orange", lw=0.8, alpha=0.7, label="inner 80% zone")
                ax.axhline(inner_hi, ls=":",  c="orange", lw=0.8, alpha=0.7)
                if is_log:
                    ax.set_yscale("log")
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("fold", fontsize=8)
            ax.grid(alpha=0.3)
            if pi == 0:
                ax.legend(fontsize=7)

        plt.tight_layout()
        out = out_dir / f"fig_hyperparams_h{hd.h}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  figure saved: {out}")

    return stability_rows, boundary_rows


def check_fixed_params(data: list[HorizonData]) -> list[dict]:
    print("\n" + "=" * 72)
    print("CHECK 4: Fixed hyperparameter sanity")
    print("=" * 72)

    rows: list[dict] = []
    for hd in data:
        m  = hd.metrics
        cv = hd.inner_cv

        # round cap
        cap_hits = int((m["calibrated_rounds"] >= hd.max_boost_rounds).sum())
        cap_frac = cap_hits / len(m)
        cap_msg  = (" cap is binding, model may be underfitting"
                    if cap_frac > 0.25 else "ok")

        # min_data_in_leaf / n_train
        if "tuned_min_data_in_leaf" in cv.columns and "n_train" in m.columns:
            merged = cv.join(m.select(["fold", "n_train"]), on="fold", how="left")
            leaf_frac = (merged["tuned_min_data_in_leaf"].cast(pl.Float64) /
                         merged["n_train"].cast(pl.Float64)).to_numpy()
            leaf_mean = float(np.mean(leaf_frac))
            leaf_max  = float(np.max(leaf_frac))
            leaf_msg  = ("typical range" if leaf_mean < 0.01
                         else " > 1% of training set, may over-regularise")
        else:
            leaf_mean = leaf_max = float("nan")
            leaf_msg  = "n/a"

        # lambda_l1 (fixed) vs tuned lambda_l2
        lam_l1  = hd.lgb_params_base.get("lambda_l1", 0.0)
        lam_l2v = (cv["tuned_lambda_l2"].to_numpy()
                   if "tuned_lambda_l2" in cv.columns else np.array([float("nan")]))
        lam_l2_mean = float(np.mean(lam_l2v))
        lam_msg = ("l1=0 looks fine given moderate l2 regularisation"
                   if lam_l2_mean < 20
                   else "l2 is large; l1=0 was not explored, unlikely to matter but worth noting")

        # max_depth vs num_leaves
        max_depth = hd.lgb_params_base.get("max_depth", -1)
        nl_vals   = (cv["tuned_num_leaves"].to_numpy()
                     if "tuned_num_leaves" in cv.columns else np.array([float("nan")]))
        nl_mean   = float(np.mean(nl_vals))
        nl_ss_hi  = hd.search_space.get("num_leaves", {}).get("high", 127)
        depth_msg = ("num_leaves well below cap, max_depth=-1 is not binding"
                     if nl_mean < 0.7 * nl_ss_hi
                     else " num_leaves approaching upper bound, consider widening search space")

        # bagging_freq (fixed, untested)
        bf = hd.lgb_params_base.get("bagging_freq", 5)
        bf_frac_vals = (cv["tuned_bagging_fraction"].to_numpy()
                        if "tuned_bagging_fraction" in cv.columns else np.array([1.0]))
        bf_msg = ("standard" if np.mean(bf_frac_vals) < 0.85
                  else "bagging_fraction consistently high, bagging_freq may not matter much")

        print(f"\n  h={hd.h}s")
        print(f"    calibrated_rounds cap ({hd.max_boost_rounds}): "
              f"{cap_hits}/{len(m)} folds ({cap_frac:.0%})  {cap_msg}")
        print(f"    min_data_in_leaf / n_train: "
              f"mean={leaf_mean:.4f}  max={leaf_max:.4f}  {leaf_msg}")
        print(f"    lambda_l1 (fixed={lam_l1}): "
              f"lambda_l2 range [{lam_l2v.min():.2g}, {lam_l2v.max():.2g}]  {lam_msg}")
        print(f"    max_depth={max_depth}: "
              f"num_leaves mean={nl_mean:.1f} (cap={nl_ss_hi})  {depth_msg}")
        print(f"    bagging_freq={bf} (fixed, untested): "
              f"bagging_fraction mean={np.mean(bf_frac_vals):.3f}  {bf_msg}")

        rows.append({
            "horizon_s": hd.h,
            "cap_hits": cap_hits,
            "cap_frac": cap_frac,
            "leaf_frac_mean": leaf_mean,
            "leaf_frac_max": leaf_max,
            "lambda_l1_fixed": lam_l1,
            "lambda_l2_mean": lam_l2_mean,
            "lambda_l2_min": float(lam_l2v.min()),
            "lambda_l2_max": float(lam_l2v.max()),
            "max_depth_fixed": max_depth,
            "num_leaves_mean": nl_mean,
            "num_leaves_ss_high": nl_ss_hi,
            "bagging_freq_fixed": bf,
            "bagging_fraction_mean": float(np.mean(bf_frac_vals)),
        })
    return rows


def check_optuna_budget(data: list[HorizonData]) -> list[dict]:
    print("\n" + "=" * 72)
    print("CHECK 5: Optuna trial budget")
    print("=" * 72)

    rows: list[dict] = []
    need = {"n_optuna_trials", "n_optuna_complete", "n_optuna_pruned", "cv_seconds"}

    for hd in data:
        cv = hd.inner_cv
        if not need.issubset(set(cv.columns)):
            print(f"\n  h={hd.h}s   Optuna columns missing, skipping")
            continue

        print(f"\n  h={hd.h}s")
        print(f"  {'fold':>5} {'complete':>10} {'pruned':>8} {'prune%':>8} {'cv_sec':>9}")

        prune_pcts: list[float] = []
        for row in cv.iter_rows(named=True):
            nc  = int(row["n_optuna_complete"])
            np_ = int(row["n_optuna_pruned"])
            tot = nc + np_
            pp  = np_ / tot * 100.0 if tot > 0 else 0.0
            prune_pcts.append(pp)
            print(f"  {row['fold']:>5d} {nc:>10d} {np_:>8d} {pp:>7.1f}% "
                  f"{row['cv_seconds']:>9.1f}")
            rows.append({
                "horizon_s": hd.h, "fold": row["fold"],
                "n_complete": nc, "n_pruned": np_,
                "prune_pct": pp,
                "cv_seconds": float(row["cv_seconds"]),
            })

        mean_pp = float(np.mean(prune_pcts))
        verdict = ("pruning is effective, 40 trials is sufficient"
                   if mean_pp >= 20 else
                   "low pruning rate, more trials might improve hyperparameter search")
        print(f"  mean prune rate: {mean_pp:.1f}%  ({verdict})")

    return rows


def check_latency(data: list[HorizonData]) -> list[dict]:
    print("\n" + "=" * 72)
    print("CHECK 6: Inference latency")
    print("=" * 72)
    print(f"\n  {'h':>4}  {'batch ns/row (mean)':>21}  {'single p50 ns (mean)':>21}  "
          f"{'single p99 ns (mean)':>21}  {'p99/p50':>8}")

    rows: list[dict] = []
    need = {"predict_batch_ns_per_row", "predict_single_p50_ns", "predict_single_p99_ns"}

    for hd in data:
        m = hd.metrics
        if not need.issubset(set(m.columns)):
            print(f"  h={hd.h}s   latency columns missing, skipping")
            continue
        batch = float(m["predict_batch_ns_per_row"].mean())
        p50   = float(m["predict_single_p50_ns"].mean())
        p99   = float(m["predict_single_p99_ns"].mean())
        ratio = p99 / p50 if p50 > 0 else float("nan")
        print(f"  {hd.h:>4d}  {batch:>21.0f}  {p50:>21.0f}  {p99:>21.0f}  {ratio:>8.2f}")
        rows.append({
            "horizon_s": hd.h,
            "batch_ns_per_row_mean": batch,
            "single_p50_ns_mean": p50,
            "single_p99_ns_mean": p99,
            "p99_p50_ratio": ratio,
        })

    if rows:
        print("\n  Reading: batch = throughput mode (rows evaluated together); "
              "single = one-at-a-time production latency.")
        print("  The single p99 is the figure to cite.")

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--horizons", type=int, nargs="+", default=ALL_HORIZONS,
                   choices=ALL_HORIZONS,
                   help="horizons to audit (default: all four)")
    p.add_argument("--out-dir", default=None,
                   help="output directory (default: <base>/validation/preanalysis_diagnostics)")
    args = p.parse_args()

    base    = Path(args.base)
    out_dir = (Path(args.out_dir) if args.out_dir
               else base / "validation" / "preanalysis_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nml_preanalysis_diagnostics.py")
    print(f"   base     : {base}")
    print(f"   horizons : {sorted(args.horizons)}")
    print(f"   out_dir  : {out_dir}")

    data: list[HorizonData] = []
    for h in sorted(args.horizons):
        hd = load_horizon(base, h)
        if hd is not None:
            data.append(hd)

    if not data:
        raise SystemExit("No horizon data found, check --base and model directories.")

    metrics_summary              = check_fold_metrics(data, out_dir)
    stability_rows, boundary_rows = check_hyperparams(data, out_dir)
    fixed_rows                   = check_fixed_params(data)
    optuna_rows                  = check_optuna_budget(data)
    latency_rows                 = check_latency(data)

    # persist
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "horizons_loaded": [hd.h for hd in data],
        "boundary_zone_frac": BOUNDARY_ZONE_FRAC,
        "boundary_hug_threshold": BOUNDARY_HUG_THRESHOLD,
        "fold_metrics_summary": metrics_summary,
        "hyperparam_stability": stability_rows,
        "boundary_check": boundary_rows,
        "fixed_param_sanity": fixed_rows,
        "optuna_budget": optuna_rows,
        "latency_summary": latency_rows,
    }
    (out_dir / "preanalysis_diagnostics.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    pl.DataFrame(metrics_summary).write_csv(out_dir / "fold_metrics_summary.csv")
    pl.DataFrame(stability_rows).write_csv(out_dir / "hyperparam_stability.csv")
    pl.DataFrame(boundary_rows).write_csv(out_dir / "boundary_check.csv")
    pl.DataFrame(latency_rows).write_csv(out_dir / "latency_summary.csv")

    print(f"\nall reports written to {out_dir}")


if __name__ == "__main__":
    main()
