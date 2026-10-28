
"""Purge window from the autocorrelation of the microstructure features."""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
from statsmodels.tsa.stattools import acf

from pipeline_utils import load_labels_for_funding_day

warnings.filterwarnings("ignore")

BASE = Path("/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt")
SPLITS_PATH = BASE / "splits.json"
OUT_PATH = BASE / "validation" / "purge_window.json"


PURGE_FEATURES = [
    "ofi_l1", "ofi_l2_5", "ofi_l6_10", "ofi_l11_20",
    "imbalance_l1", "imbalance_l2_5", "imbalance_l6_10", "imbalance_l11_20",
    "trade_imbalance_30s", "signed_vol_30s", "micro_return_30s",
]

DEFAULT_MAX_LAG = 1800

DEFAULT_CONFIRM_LAGS = 5

# Standard normal 1.96 ≈ 2 for the Bartlett 95% white-noise CI band.
CI_Z = 2.0

TAU_ROUND_SEC = 60

MIN_FINITE_OBS = 1000  # below this, ACF estimates are too noisy to trust


def load_pre_analysis_days() -> list[str]:
    if not SPLITS_PATH.exists():
        raise SystemExit(
            f"splits.json not found at {SPLITS_PATH}. Run "
            f"`label_split.py --build-splits` first."
        )
    sp = json.loads(SPLITS_PATH.read_text())
    days = sp.get("splits", {}).get("pre_analysis")
    if not days:
        raise SystemExit(
            f"splits.json at {SPLITS_PATH} has no 'pre_analysis' key."
        )
    return list(days)


def _find_first_cross_below(
    acf_vals: np.ndarray, n_obs: int, *, confirm_lags: int
) -> int | None:
    """Smallest lag τ ≥ 1 such that |acf[k]| < CI for every k in
    [τ, τ + confirm_lags). Returns None if no such τ exists within
    `len(acf_vals) - confirm_lags`."""
    ci = CI_Z / math.sqrt(max(n_obs, 1))
    below = np.abs(acf_vals) < ci
    L = len(below)
    for tau in range(1, L - confirm_lags + 1):
        if below[tau : tau + confirm_lags].all():
            return tau
    return None


def _process_day(day: str, max_lag: int, confirm_lags: int, base: Path = BASE) -> dict:
    """Per-day worker: compute ACF and decay τ for every PURGE_FEATURES
    column. Designed to run in a ProcessPoolExecutor; raises only on
    catastrophic load failures (caller traps and records as 'error')."""
    raw = load_labels_for_funding_day(base, day)
    out: dict = {
        "funding_day": day,
        "n_rows": int(raw.height),
        "features": {},
    }
    if raw.is_empty():
        return out

    for feat in PURGE_FEATURES:
        if feat not in raw.columns:
            out["features"][feat] = {"reason": "missing_column"}
            continue
        s = raw[feat].cast(pl.Float64).to_numpy()
        finite_mask = np.isfinite(s)
        n_finite = int(finite_mask.sum())
        if n_finite < MIN_FINITE_OBS:
            out["features"][feat] = {
                "reason": "too_few_finite", "n_finite": n_finite,
            }
            continue
        try:
            vals = acf(s, nlags=max_lag, fft=True, missing="conservative")
        except Exception as exc:  # noqa: BLE001
            out["features"][feat] = {
                "reason": f"error: {type(exc).__name__}",
                "n_finite": n_finite,
            }
            continue
        tau = _find_first_cross_below(vals, n_finite, confirm_lags=confirm_lags)
        diag = {
            "acf_at_60s":  float(vals[60])  if len(vals) > 60  else None,
            "acf_at_300s": float(vals[300]) if len(vals) > 300 else None,
            "acf_at_max_lag": float(vals[-1]) if len(vals) else None,
        }
        out["features"][feat] = {
            "n_finite": n_finite,
            "ci_threshold": CI_Z / math.sqrt(n_finite),
            "tau_seconds": int(tau) if tau is not None else None,
            **diag,
            "_acf_array": [float(x) for x in vals],
        }
    return out


def _generate_acf_plots(
    records: list[dict],
    *,
    tau_purge_sec: int,
    per_feature_tau: dict,
    out_dir: Path,
    max_lag: int,
) -> int:
    """One PNG per PURGE_FEATURES feature: per-day ACF curves, the median across
    days, the white-noise CI band, the per-feature median-ACF tau and the global
    tau_purge. Returns the number of PNGs written. matplotlib is imported lazily
    so the worker processes never load it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    days_in_order = [rec.get("funding_day") for rec in records if rec]
    lags = np.arange(max_lag + 1)

    for feat in PURGE_FEATURES:
        # Collect every day's ACF array + per-day τ + per-day n_finite.
        per_day: list[tuple[str, np.ndarray, int | None, int]] = []
        for rec in records:
            info = (rec.get("features") or {}).get(feat) if rec else None
            if not info or "_acf_array" not in info:
                continue
            per_day.append((
                rec.get("funding_day"),
                np.asarray(info["_acf_array"], dtype=np.float64),
                info.get("tau_seconds"),
                int(info.get("n_finite", 0)),
            ))
        if not per_day:
            continue

        common_len = min(arr.size for _, arr, _, _ in per_day)
        mat = np.vstack([arr[:common_len] for _, arr, _, _ in per_day])
        lag_axis = lags[:common_len]

        median_n = int(np.median([n for _, _, _, n in per_day if n > 0]) or 1)
        ci_band = CI_Z / math.sqrt(median_n)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        # Per-day curves: thin grey, low alpha, no legend entry.
        for _, arr, _, _ in per_day:
            ax.plot(lag_axis, arr[:common_len], color="0.5", linewidth=0.5, alpha=0.25)
        # Median across days.
        median_curve = np.median(mat, axis=0)
        ax.plot(lag_axis, median_curve, color="C0", linewidth=2.0,
                label=f"median over {len(per_day)} days")
        ax.axhspan(-ci_band, ci_band, color="C1", alpha=0.15,
                   label=f"±2/√n_median  (n_median={median_n:,})")
        # Per-day τ markers along the x-axis (small dashes, diagnostic only).
        for _, _, tau, _ in per_day:
            if tau is not None:
                ax.axvline(tau, ymin=0.0, ymax=0.02, color="C2", alpha=0.4, linewidth=1.0)
        feat_info = per_feature_tau.get(feat) or {}
        feat_tau = feat_info.get("tau_seconds")
        if feat_tau is not None:
            ax.axvline(feat_tau, color="C1", linewidth=2.0, linestyle="-",
                       label=f"median-ACF τ_{{{feat}}} = {feat_tau}s")
        # Global τ_purge: thick red dashed vertical line.
        ax.axvline(tau_purge_sec, color="C3", linewidth=2.0, linestyle="--",
                   label=f"τ_purge = {tau_purge_sec}s")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_xlim(0, common_len - 1)
        y_lo = min(-0.05, float(mat.min()) * 1.1)
        y_hi = max(0.05, float(mat.max()) * 1.1)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("lag (seconds)")
        ax.set_ylabel("ACF")
        ax.set_title(f"ACF of {feat}  ({len(per_day)} pre-analysis days)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        png_path = out_dir / f"{feat}.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        n_written += 1
    return n_written


def _strip_acf_arrays(records: list[dict]) -> None:
    """Remove `_acf_array` from every per-feature dict in-place so the
    on-disk JSON stays small (~30 days × 11 features × ~1800 floats would
    bloat it ~5 MB)."""
    for rec in records:
        if not rec:
            continue
        for info in (rec.get("features") or {}).values():
            if isinstance(info, dict):
                info.pop("_acf_array", None)


def _compute_median_acf_tau(
    records: list[dict],
    *,
    max_lag: int,
    confirm_lags: int,
) -> tuple[dict, int | None, list[tuple]]:
    """Primary aggregation: per-feature median ACF across days, then max τ across features."""
    per_feature: dict = {}
    feature_taus: list[int] = []
    skipped: list[tuple] = []

    for feat in PURGE_FEATURES:
        arrays: list[np.ndarray] = []
        n_finites: list[int] = []
        for rec in records:
            if not rec:
                continue
            info = (rec.get("features") or {}).get(feat)
            if not info or "_acf_array" not in info:
                if info and "reason" in info:
                    skipped.append((rec.get("funding_day"), feat, info["reason"]))
                continue
            arrays.append(np.asarray(info["_acf_array"], dtype=np.float64))
            n_finites.append(int(info.get("n_finite", 0)))

        if not arrays:
            per_feature[feat] = {"tau_seconds": None, "reason": "no_valid_days"}
            continue

        common_len = min(arr.size for arr in arrays)
        mat = np.vstack([arr[:common_len] for arr in arrays])
        median_acf = np.median(mat, axis=0)
        median_n = int(np.median([n for n in n_finites if n > 0]) or 1)

        tau = _find_first_cross_below(
            median_acf, median_n, confirm_lags=confirm_lags
        )

        per_feature[feat] = {
            "tau_seconds": int(tau) if tau is not None else None,
            "n_days_used": len(arrays),
            "n_median_finite": median_n,
            "ci_threshold": CI_Z / math.sqrt(median_n),
            "median_acf_at_60s":  float(median_acf[60])  if len(median_acf) > 60  else None,
            "median_acf_at_300s": float(median_acf[300]) if len(median_acf) > 300 else None,
            "median_acf_at_max_lag": float(median_acf[-1]) if len(median_acf) else None,
        }
        feature_taus.append(int(tau) if tau is not None else int(max_lag))

    tau_max = int(max(feature_taus)) if feature_taus else None
    return per_feature, tau_max, skipped


def _collect_per_day_max_diagnostic(records: list[dict], max_lag: int) -> np.ndarray:
    """Diagnostic only: per-day max τ across features. Used for the
    distribution summary in the JSON (median/p75/p95) so the appendix
    can compare the median-ACF result against the day-level worst case.
    NOT used to set tau_purge_sec."""
    taus: list[int] = []
    for rec in records:
        if not rec:
            continue
        day_taus: list[int] = []
        for feat, info in (rec.get("features") or {}).items():
            if not isinstance(info, dict):
                continue
            tau = info.get("tau_seconds")
            if tau is not None:
                day_taus.append(int(tau))
            elif "reason" not in info:
                day_taus.append(int(max_lag))
        if day_taus:
            taus.append(max(day_taus))
    return np.array(taus, dtype=np.int64)


def compute_tau_for_days(
    days: list[str],
    *,
    base: Path = BASE,
    max_lag: int = DEFAULT_MAX_LAG,
    confirm_lags: int = DEFAULT_CONFIRM_LAGS,
    n_workers: int = 4,
) -> int | None:
    """Compute the empirical purge window for an arbitrary list of funding days."""
    if not days:
        return None

    records: list[dict] = [{} for _ in days]
    idx = {d: i for i, d in enumerate(days)}

    if len(days) <= 4 or n_workers <= 1:
        # Sequential path: avoids ProcessPoolExecutor overhead for small batches.
        for d in days:
            try:
                rec = _process_day(d, max_lag, confirm_lags, base)
            except Exception: 
                rec = {"funding_day": d, "error": "processing failed"}
            records[idx[d]] = rec
    else:
        effective_workers = min(n_workers, len(days))
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            futures = {
                pool.submit(_process_day, d, max_lag, confirm_lags, base): d
                for d in days
            }
            for fut in as_completed(futures):
                d = futures[fut]
                try:
                    rec = fut.result()
                except Exception:  
                    rec = {"funding_day": d, "error": "processing failed"}
                records[idx[d]] = rec

    _, tau_max, _ = _compute_median_acf_tau(
        records, max_lag=max_lag, confirm_lags=confirm_lags
    )
    if tau_max is None:
        return None
    return int(math.ceil(tau_max / TAU_ROUND_SEC) * TAU_ROUND_SEC)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel worker processes (default 4)")
    ap.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG,
                    help=f"max ACF lag in seconds (default {DEFAULT_MAX_LAG})")
    ap.add_argument("--confirm-lags", type=int, default=DEFAULT_CONFIRM_LAGS,
                    help=f"required consecutive below-CI lags (default {DEFAULT_CONFIRM_LAGS})")
    ap.add_argument("--max-days", type=int, default=None,
                    help="debug shortcut: only first N pre-analysis days")
    args = ap.parse_args()

    days = load_pre_analysis_days()
    if args.max_days is not None:
        days = days[: args.max_days]
        print(f" --max-days: only running the first {len(days)} day(s)")

    print(
        f"=== Empirical purge window: ACF on {len(days)} pre-analysis days "
        f"× {len(PURGE_FEATURES)} features, max_lag={args.max_lag}s "
        f"with {args.workers} worker(s) ==="
    )

    t0 = time.perf_counter()
    records: list[dict] = [{} for _ in days]
    idx = {d: i for i, d in enumerate(days)}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_day, d, args.max_lag, args.confirm_lags): d
            for d in days
        }
        done = 0
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR {d}: {exc}", file=sys.stderr)
                rec = {"funding_day": d, "error": repr(exc)}
            records[idx[d]] = rec
            done += 1
            if done % 5 == 0 or done == len(days):
                print(f"  {done}/{len(days)} days done", flush=True)

    per_feature_tau, tau_max, skipped = _compute_median_acf_tau(
        records, max_lag=args.max_lag, confirm_lags=args.confirm_lags
    )
    if tau_max is None:
        raise SystemExit(
            "No τ computed, no feature produced a valid median ACF. "
            f"({len(skipped)} (day, feature) pairs skipped, see records for reasons.)"
        )

    tau_purge = int(math.ceil(tau_max / TAU_ROUND_SEC) * TAU_ROUND_SEC)

    # Identify which feature drove the result (the slowest one).
    slowest_feat = max(
        ((f, v.get("tau_seconds") if v.get("tau_seconds") is not None else args.max_lag)
         for f, v in per_feature_tau.items()),
        key=lambda x: x[1],
    )

    diag_taus = _collect_per_day_max_diagnostic(records, args.max_lag)
    diagnostic = {
        "method": "per_day_max_across_features",
        "note": "diagnostic only; tau_purge is set by median-ACF method above",
        "n_days": int(diag_taus.size),
        "median_sec": float(np.median(diag_taus)) if diag_taus.size else None,
        "p75_sec":    float(np.percentile(diag_taus, 75)) if diag_taus.size else None,
        "p90_sec":    float(np.percentile(diag_taus, 90)) if diag_taus.size else None,
        "p95_sec":    float(np.percentile(diag_taus, 95)) if diag_taus.size else None,
        "max_sec":    float(diag_taus.max()) if diag_taus.size else None,
    }

    summary = {
        "n_days": len(days),
        "n_features": len(PURGE_FEATURES),
        "features": list(PURGE_FEATURES),
        "tau_purge_sec":  tau_purge,
        "tau_aggregation_method": "median_acf_max_across_features",
        "tau_driver_feature": slowest_feat[0],
        "tau_driver_value_sec": int(slowest_feat[1]),
        "per_feature_tau": per_feature_tau,
        "diagnostic_per_day_max_distribution": diagnostic,
        "n_skipped": len(skipped),
        "ci_z": CI_Z,
        "max_lag_sec": args.max_lag,
        "confirm_lags_sec": args.confirm_lags,
        "rounding_step_sec": TAU_ROUND_SEC,
        "min_finite_obs": MIN_FINITE_OBS,
        "computed_at_utc": int(time.time()),
        "per_day": records,
    }

    plots_dir = OUT_PATH.parent / "purge_window_plots"
    n_plots = _generate_acf_plots(
        records,
        tau_purge_sec=tau_purge,
        per_feature_tau=per_feature_tau,
        out_dir=plots_dir,
        max_lag=args.max_lag,
    )
    summary["n_plots"] = n_plots
    summary["plots_dir"] = str(plots_dir)

    _strip_acf_arrays(records)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2, default=str))

    dt = time.perf_counter() - t0
    print(f"\nWrote {OUT_PATH} in {dt:.1f}s")
    print(f"Wrote {n_plots} ACF plot(s) to {plots_dir}/")

    print(f"\nPer-feature τ (median ACF across {len(days)} days):")
    for feat in PURGE_FEATURES:
        info = per_feature_tau.get(feat, {})
        tau_str = (f"{info['tau_seconds']:>5}s"
                   if info.get("tau_seconds") is not None
                   else f">{args.max_lag}s (capped)")
        marker = "  <- driver" if feat == slowest_feat[0] else ""
        print(f"  {feat:<22} τ = {tau_str}{marker}")

    print(f"\nDiagnostic per-day-max distribution (not used for tau_purge):")
    print(f"  median = {diagnostic['median_sec']:6.0f}s")
    print(f"  p75    = {diagnostic['p75_sec']:6.0f}s")
    print(f"  p90    = {diagnostic['p90_sec']:6.0f}s")
    print(f"  p95    = {diagnostic['p95_sec']:6.0f}s")
    print(f"  max    = {diagnostic['max_sec']:6.0f}s")

    print(
        f"\n  tau_purge_sec = {tau_purge}s  "
        f"(driven by {slowest_feat[0]} = {slowest_feat[1]}s, "
        f"rounded up to nearest {TAU_ROUND_SEC}s)"
    )
    if skipped:
        print(f"\n  {len(skipped)} (day, feature) pair(s) skipped; "
              f"first few: {skipped[:5]}")
    print(
        f"\nWrote {OUT_PATH}, ml_train.py and ml_baseline_ridge.py read "
        f"tau_purge_sec={tau_purge} automatically via _load_preanalysis_tau()."
    )


if __name__ == "__main__":
    main()
