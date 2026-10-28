"""Hyperparameter transfer across walk-forward folds."""
from __future__ import annotations

import argparse
import json
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from ml_train import (
    DayCache,
    build_outer_folds,
    load_split_days,
    train_fold,
)

DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
ALL_HORIZONS = [1, 5, 10, 30]
TARGET = "drift_mid"

# Map the inner_cv_h*.csv tuned_* columns onto LightGBM parameter names.
TUNED_COL_TO_PARAM = {
    "tuned_num_leaves": "num_leaves",
    "tuned_learning_rate": "learning_rate",
    "tuned_min_data_in_leaf": "min_data_in_leaf",
    "tuned_feature_fraction": "feature_fraction",
    "tuned_bagging_fraction": "bagging_fraction",
    "tuned_lambda_l2": "lambda_l2",
}
INT_PARAMS = {"num_leaves", "min_data_in_leaf"}

REPRO_TOL = 2e-3


def _model_dir(base: Path, h: int) -> Path:
    return base / "models" / f"lgbm_pre_analysis_{TARGET}_h{h}"


def load_fold_configs(base: Path, h: int) -> dict:
    """Assemble, per outer fold, the exact config + round count + baseline R2
    used by the production pre-analysis run, read straight from its artifacts."""
    md = _model_dir(base, h)
    run_config = json.loads((md / f"run_config_h{h}.json").read_text())
    base_params: dict = dict(run_config["lgb_params_base"])
    feature_cols: list[str] = list(run_config["feature_cols"])
    target_col: str = run_config["target_col"]
    tau: int = int(run_config["preanalysis_tau_sec"])

    inner = pl.read_csv(md / f"inner_cv_h{h}.csv")
    metrics = pl.read_csv(md / f"metrics_h{h}.csv")
    baseline_r2 = {int(r["fold"]): float(r["r2"]) for r in metrics.iter_rows(named=True)}
    baseline_rmse = {int(r["fold"]): float(r["rmse"]) for r in metrics.iter_rows(named=True)}
    baseline_dir = {int(r["fold"]): float(r["dir_acc"]) for r in metrics.iter_rows(named=True)}
    baseline_rounds = {
        int(r["fold"]): int(r["calibrated_rounds"]) for r in metrics.iter_rows(named=True)
    }

    configs: dict[int, dict] = {}
    rounds: dict[int, int] = {}
    for row in inner.iter_rows(named=True):
        f = int(row["fold"])
        params = dict(base_params)
        for col, pname in TUNED_COL_TO_PARAM.items():
            val = row[col]
            params[pname] = int(round(val)) if pname in INT_PARAMS else float(val)
        configs[f] = params
        rounds[f] = int(row["calibrated_rounds"])
        # inner_cv and metrics agree on calibrated_rounds; assert it to be safe.
        if f in baseline_rounds and baseline_rounds[f] != rounds[f]:
            print(f"    h={h}s fold {f}: inner_cv rounds {rounds[f]} != "
                  f"metrics rounds {baseline_rounds[f]} (using inner_cv)")

    return {
        "base_params": base_params,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "tau": tau,
        "configs": configs,
        "rounds": rounds,
        "baseline_r2": baseline_r2,
        "baseline_rmse": baseline_rmse,
        "baseline_dir": baseline_dir,
        "n_folds": len(configs),
    }


def _foreign_fold(j: int, n_folds: int, pairing: str, off: int) -> int:
    """Pick the donor fold for recipient fold j."""
    if pairing == "reverse":
        return n_folds - 1 - j
    if pairing == "backlag":
        return max(0, j - off)
    return (j + off) % n_folds


def run_horizon(
    base: Path,
    h: int,
    all_days: list[str],
    n_pre_analysis: int,
    offset: int | None,
    pairing: str,
    self_check_only: bool,
    csv_writer: csv.DictWriter,
    csv_handle,
) -> list[dict]:
    cfg = load_fold_configs(base, h)
    feature_cols = cfg["feature_cols"]
    target_col = cfg["target_col"]
    tau = cfg["tau"]
    configs = cfg["configs"]
    rounds = cfg["rounds"]
    baseline_r2 = cfg["baseline_r2"]
    baseline_rmse = cfg["baseline_rmse"]
    baseline_dir = cfg["baseline_dir"]
    n_folds = cfg["n_folds"]

    off = (n_folds // 2) if offset is None else offset
    folds = build_outer_folds(
        all_days, mode="pre_analysis", n_pre_analysis=n_pre_analysis, max_folds=None
    )
    if len(folds) != n_folds:
        print(f"    h={h}s: built {len(folds)} folds but artifacts have "
              f"{n_folds}; using the artifact fold ids that exist.")

    cache = DayCache(
        base, feature_cols, target_col, h, drop_stale=True, label_stem="labels",
    )

    if pairing == "reverse":
        pair_desc = f"reverse (j↔{n_folds-1}-j)"
    elif pairing == "backlag":
        pair_desc = f"backlag (donor=max(0,j-{off}), leak-free)"
    else:
        pair_desc = f"offset {off}"
    print(f"\n{'='*72}\nHORIZON h={h}s  (n_folds={n_folds}, pairing={pair_desc}, "
          f"tau={tau}s)\n{'='*72}")
    print(f"  {'fold':>4} {'src':>4} {'repro_d':>9} "
          f"{'r2_own':>8} {'r2_for':>8} {'d_r2':>9} "
          f"{'dir_own':>8} {'dir_for':>8} {'d_dir':>9} {'d_rmse':>9}")

    rows: list[dict] = []
    for fold in folds:
        j = int(fold["fold"])
        if j not in configs:
            continue
        c = _foreign_fold(j, n_folds, pairing, off)
        if c not in configs:
            c = j  # donor missing -> degrade to self-check (delta becomes NaN)

        own = train_fold(fold, cache, feature_cols, configs[j], rounds[j], tau)
        if own is None:
            continue
        r2_own = float(own["metrics"]["r2"])
        rmse_own = float(own["metrics"]["rmse"])
        dir_own = float(own["metrics"]["dir_acc"])
        repro_d = r2_own - baseline_r2.get(j, float("nan"))

        r2_foreign = rmse_foreign = dir_foreign = float("nan")
        d_r2 = d_rmse = d_dir = float("nan")
        if not self_check_only and c != j:
            foreign = train_fold(fold, cache, feature_cols, configs[c], rounds[c], tau)
            if foreign is not None:
                r2_foreign = float(foreign["metrics"]["r2"])
                rmse_foreign = float(foreign["metrics"]["rmse"])
                dir_foreign = float(foreign["metrics"]["dir_acc"])
                d_r2 = r2_own - r2_foreign        # >0 => own better
                d_dir = dir_own - dir_foreign      # >0 => own better
                d_rmse = rmse_foreign - rmse_own   # >0 => own better (lower rmse)

        flag = "  REPRO" if abs(repro_d) > REPRO_TOL else ""
        print(f"  {j:>4d} {c:>4d} {repro_d:>9.1e} "
              f"{r2_own:>8.5f} {r2_foreign:>8.5f} {d_r2:>9.5f} "
              f"{dir_own:>8.4f} {dir_foreign:>8.4f} {d_dir:>9.5f} {d_rmse:>9.4f}{flag}")

        row = {
            "horizon_s": h,
            "fold": j,
            "foreign_fold": c,
            "repro_delta": repro_d,
            "r2_own": r2_own,
            "r2_foreign": r2_foreign,
            "r2_baseline": baseline_r2.get(j, float("nan")),
            "d_r2_own_minus_foreign": d_r2,
            "dir_own": dir_own,
            "dir_foreign": dir_foreign,
            "dir_baseline": baseline_dir.get(j, float("nan")),
            "d_dir_own_minus_foreign": d_dir,
            "rmse_own": rmse_own,
            "rmse_foreign": rmse_foreign,
            "rmse_baseline": baseline_rmse.get(j, float("nan")),
            "d_rmse_foreign_minus_own": d_rmse,
            "rounds_own": rounds[j],
            "rounds_foreign": rounds.get(c, -1),
            "n_test": int(own["n_test"]),
        }
        rows.append(row)
        csv_writer.writerow(row)
        csv_handle.flush()

    repro_max = max((abs(r["repro_delta"]) for r in rows), default=float("nan"))
    print(f"\n  reproduction: max|r2_own - r2_baseline| = {repro_max:.2e} "
          f"({'OK' if repro_max <= REPRO_TOL else 'FAILED, deltas not trustworthy'})")

    specs = [
        ("R2",      "d_r2_own_minus_foreign",       "r2_baseline"),
        ("dir_acc", "d_dir_own_minus_foreign",      "dir_baseline"),
        ("RMSE",    "d_rmse_foreign_minus_own",     "rmse_baseline"),
    ]
    r2_is_h1 = True
    for name, dcol, bcol in specs:
        s = float(np.std([r[bcol] for r in rows])) if rows else float("nan")
        d = np.array([r[dcol] for r in rows if np.isfinite(r[dcol])], dtype=float)
        if not d.size:
            continue
        mean_d = float(np.mean(d))
        ratio = mean_d / s if s > 0 else float("nan")
        is_h1 = abs(mean_d) <= s and d.max() <= 2 * s
        print(f"  {name:<8} own-better delta: mean={mean_d:+.5f} median={np.median(d):+.5f} "
              f"max={d.max():+.5f} | s={s:.5f} mean/s={ratio:+.2f} "
              f"-> {'H1' if is_h1 else 'H2'}")
        if name == "R2":
            r2_is_h1 = is_h1
    verdict = ("H1 (interchangeable; weekly reuse safe)" if r2_is_h1
               else "H2 (per-fold tuning load-bearing; retune every fold)")
    print(f"  => verdict (R2, pre-registered): {verdict}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--horizons", type=int, nargs="+", default=ALL_HORIZONS,
                   choices=ALL_HORIZONS)
    p.add_argument("--offset", type=int, default=None,
                   help="foreign-config fold offset for --pairing offset (default n_folds//2)")
    p.add_argument("--pairing", choices=["offset", "reverse", "backlag"], default="offset",
                   help="offset: modular shift by --offset; reverse: mirror j<->n-1-j "
                        "(samples the zero-overlap extreme pairs); backlag: donor "
                        "max(0,j-offset), always strictly earlier => fully leak-free")
    p.add_argument("--self-check-only", action="store_true",
                   help="only run the own-config reproduction refits")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    base = Path(args.base)
    out_dir = (Path(args.out_dir) if args.out_dir
               else base / "validation" / "crossfold_stability")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_days, n_pre_analysis = load_split_days(base)

    print("ml_crossfold_stability.py")
    print(f"   base      : {base}")
    print(f"   horizons  : {sorted(args.horizons)}")
    print(f"   mode      : {'self-check only' if args.self_check_only else 'own vs foreign'}")
    print(f"   out_dir   : {out_dir}")

    csv_path = out_dir / "crossfold_stability.csv"
    fieldnames = ["horizon_s", "fold", "foreign_fold", "repro_delta",
                  "r2_own", "r2_foreign", "r2_baseline", "d_r2_own_minus_foreign",
                  "dir_own", "dir_foreign", "dir_baseline", "d_dir_own_minus_foreign",
                  "rmse_own", "rmse_foreign", "rmse_baseline", "d_rmse_foreign_minus_own",
                  "rounds_own", "rounds_foreign", "n_test"]
    csv_handle = csv_path.open("w", newline="")
    writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
    writer.writeheader()

    all_rows: list[dict] = []
    for h in sorted(args.horizons):
        all_rows.extend(
            run_horizon(base, h, all_days, n_pre_analysis, args.offset,
                        args.pairing, args.self_check_only, writer, csv_handle)
        )
    csv_handle.close()

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "self_check_only": args.self_check_only,
        "pairing": args.pairing,
        "offset": args.offset,
        "repro_tol": REPRO_TOL,
        "rows": all_rows,
    }
    (out_dir / "crossfold_stability.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwritten: {csv_path}")
    print(f"written: {out_dir / 'crossfold_stability.json'}")


if __name__ == "__main__":
    main()
