#!/usr/bin/env python3
"""Statistical battery on the 30 pre-analysis funding days."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

from pipeline_utils import load_labels_for_funding_day
from statistical_tests import adf_test, arch_test, kpss_test
from ml_train import FEATURE_COLS

warnings.filterwarnings("ignore")

BASE = Path("/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt")
SPLITS_PATH = BASE / "splits.json"
OUT_DIR = BASE / "validation"

HORIZONS = (1, 5, 10, 30)
LABEL_METRICS = ("drift_micro", "return_micro", "drift_mid", "return_mid")

ARCH_FEATURES: tuple[str, ...] = (
    # returns (the canonical case)
    "return_1s", "return_5s", "return_30s",
    "micro_return_1s", "micro_return_5s", "micro_return_30s",
    # realised vol (vol-of-vol semantics)
    "realized_vol_60s",
    # signed flow + trade imbalance
    "signed_vol_1s", "signed_vol_5s", "signed_vol_30s",
    "trade_imbalance_1s", "trade_imbalance_5s", "trade_imbalance_30s",
    # OFI (multi-level Cont-Stoikov)
    "ofi_l1", "ofi_l2_5", "ofi_l6_10", "ofi_l11_20",
    # VPIN at the three scales
    "vpin_micro", "vpin_base", "vpin_macro",
)

MIN_OBS = 1000

VAR_EPS = 1e-18


def load_pre_analysis_days() -> list[str]:
    """The 30 pre-analysis days from splits.json. Hard error if missing,
    every model-design decision depends on this exact set."""
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


def _label_validity_mask(df: pl.DataFrame, horizon: int) -> pl.Series:
    """label_valid_<horizon> if present, else a True-everywhere fallback."""
    col = f"label_valid_{horizon}"
    if col in df.columns:
        return df[col].fill_null(False)
    return pl.Series(np.ones(df.height, dtype=bool))


def _series_numpy(df: pl.DataFrame, col: str, *, mask: pl.Series | None = None) -> np.ndarray:
    """Extract a finite-valued numpy view of `col`, optionally masked.

    polars->numpy via to_numpy() with zero_copy_only=False; then we drop
    NaN/inf so the statsmodels routines never receive non-finite input.
    A constant-series check is left to the caller.
    """
    if col not in df.columns:
        return np.empty(0, dtype=np.float64)
    s = df[col]
    if mask is not None:
        s = s.filter(mask)
    arr = s.to_numpy()
    if arr.dtype.kind != "f":
        arr = arr.astype(np.float64, copy=False)
    arr = arr[np.isfinite(arr)]
    return arr


def _adf_one(arr: np.ndarray) -> dict:
    """Run ADF on `arr` and return a JSON-safe dict, or a skip-reason dict."""
    if arr.size < MIN_OBS:
        return {"reason": "too_few_obs", "n": int(arr.size)}
    if float(np.var(arr)) < VAR_EPS:
        return {"reason": "constant", "n": int(arr.size)}
    try:
        r = adf_test(arr)
    except Exception as exc:  # noqa: BLE001
        return {"reason": f"error: {type(exc).__name__}", "n": int(arr.size)}
    return {
        "stat": round(r.statistic, 6),
        "p": float(r.p_value),
        "lags": r.lags,
        "n": r.n_obs,
        "stationary": r.stationary,
    }


def _kpss_one(arr: np.ndarray) -> dict:
    if arr.size < MIN_OBS:
        return {"reason": "too_few_obs", "n": int(arr.size)}
    if float(np.var(arr)) < VAR_EPS:
        return {"reason": "constant", "n": int(arr.size)}
    try:
        r = kpss_test(arr, regression="c")
    except Exception as exc:  # noqa: BLE001
        return {"reason": f"error: {type(exc).__name__}", "n": int(arr.size)}
    return {
        "stat": round(r.statistic, 6),
        "p": float(r.p_value),
        "lags": r.lags,
        "n": r.n_obs,
        "stationary": r.stationary,
        "p_capped": r.p_value_capped,
    }


def _arch_one(arr: np.ndarray, lags: int = 10) -> dict:
    if arr.size < MIN_OBS:
        return {"reason": "too_few_obs", "n": int(arr.size)}
    if float(np.var(arr)) < VAR_EPS:
        return {"reason": "constant", "n": int(arr.size)}
    try:
        r = arch_test(arr, lags=lags)
    except Exception as exc:  # noqa: BLE001
        return {"reason": f"error: {type(exc).__name__}", "n": int(arr.size)}
    return {
        "lm_stat": round(r.statistic, 6),
        "lm_p": float(r.p_value),
        "f_stat": round(r.f_statistic, 6),
        "f_p": float(r.f_p_value),
        "heteroskedastic": r.heteroskedastic,
        "n": int(arr.size),
    }


def _process_adf_day(day: str) -> dict:
    """ADF on every feature (one mask = 'row is valid') and every label
    column (one mask per horizon = label_valid_<h>). Per-feature mask
    differs from per-label because a feature is meaningful on every row
    where the feature itself is finite, even if the forward label window
    overlaps a pause."""
    raw = load_labels_for_funding_day(BASE, day)
    out: dict = {"funding_day": day, "battery": "adf", "n_rows": int(raw.height)}
    results: dict = {}
    if raw.is_empty():
        out["results"] = results
        return out

    feat_mask = raw["valid"].fill_null(False) if "valid" in raw.columns else None
    for f in FEATURE_COLS:
        results[f] = _adf_one(_series_numpy(raw, f, mask=feat_mask))

    # Labels: per-horizon validity mask.
    for h in HORIZONS:
        lv = _label_validity_mask(raw, h)
        for m in LABEL_METRICS:
            col = f"{m}_{h}"
            results[col] = _adf_one(_series_numpy(raw, col, mask=lv))

    out["results"] = results
    return out


def _process_kpss_day(day: str) -> dict:
    raw = load_labels_for_funding_day(BASE, day)
    out: dict = {"funding_day": day, "battery": "kpss", "n_rows": int(raw.height)}
    results: dict = {}
    if raw.is_empty():
        out["results"] = results
        return out

    feat_mask = raw["valid"].fill_null(False) if "valid" in raw.columns else None
    for f in FEATURE_COLS:
        results[f] = _kpss_one(_series_numpy(raw, f, mask=feat_mask))

    for h in HORIZONS:
        lv = _label_validity_mask(raw, h)
        for m in LABEL_METRICS:
            col = f"{m}_{h}"
            results[col] = _kpss_one(_series_numpy(raw, col, mask=lv))

    out["results"] = results
    return out


def _process_arch_day(day: str) -> dict:
    raw = load_labels_for_funding_day(BASE, day)
    out: dict = {"funding_day": day, "battery": "arch", "n_rows": int(raw.height)}
    results: dict = {}
    if raw.is_empty():
        out["results"] = results
        return out

    feat_mask = raw["valid"].fill_null(False) if "valid" in raw.columns else None
    for f in ARCH_FEATURES:
        results[f] = _arch_one(_series_numpy(raw, f, mask=feat_mask))

    for h in HORIZONS:
        lv = _label_validity_mask(raw, h)
        for m in LABEL_METRICS:
            col = f"{m}_{h}"
            results[col] = _arch_one(_series_numpy(raw, col, mask=lv))

    out["results"] = results
    return out


def _run_battery(
    name: str,
    days: list[str],
    worker_fn,
    out_path: Path,
    *,
    n_workers: int,
) -> None:
    """Distribute the per-day worker across `n_workers` processes, write
    results to `out_path` in funding-day order, print a one-line summary."""
    print(
        f"\n=== {name} battery on {len(days)} pre-analysis days "
        f"with {n_workers} worker(s) ===",
        flush=True,
    )
    t0 = time.perf_counter()
    records: list[dict] = [{} for _ in days]
    day_index = {d: i for i, d in enumerate(days)}

    if n_workers <= 1:
        for i, d in enumerate(days):
            try:
                rec = worker_fn(d)
            except Exception as exc:  # noqa: BLE001
                rec = {"funding_day": d, "battery": name.lower(), "error": repr(exc)}
            records[day_index[d]] = rec
            if (i + 1) % 5 == 0 or i + 1 == len(days):
                print(f"  {i + 1}/{len(days)} days done", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(worker_fn, d): d for d in days}
            done = 0
            for fut in as_completed(futures):
                d = futures[fut]
                try:
                    rec = fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"  ERROR {d}: {exc}", file=sys.stderr, flush=True)
                    rec = {"funding_day": d, "battery": name.lower(), "error": repr(exc)}
                records[day_index[d]] = rec
                done += 1
                if done % 5 == 0 or done == len(days):
                    print(f"  {done}/{len(days)} days done", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, default=str))
    dt = time.perf_counter() - t0
    print(f"  wrote {out_path} in {dt:.1f}s")

    _print_battery_summary(name.lower(), records)


def _print_battery_summary(battery: str, records: list[dict]) -> None:
    """Console one-pager: per-series fraction of days rejecting H0.

    For ADF: % of days where the series is flagged stationary (p<0.05).
    For KPSS: % of days where the series is flagged stationary (p>=0.05).
    For ARCH: % of days flagged heteroskedastic (p<0.05).
    """
    if not records:
        return
    flag_key = {"adf": "stationary", "kpss": "stationary", "arch": "heteroskedastic"}[battery]
    counts: dict[str, list[bool]] = {}
    for rec in records:
        for series, res in (rec.get("results") or {}).items():
            if isinstance(res, dict) and flag_key in res:
                counts.setdefault(series, []).append(bool(res[flag_key]))
    if not counts:
        print("  (no successful tests to summarise)")
        return
    label = {"adf": "stationary", "kpss": "stationary", "arch": "ARCH"}[battery]
    print(f"  per-series {label} fraction (#days reject H0 / #days tested):")
    width = max(len(s) for s in counts)
    for series in sorted(counts):
        flags = counts[series]
        n = len(flags)
        k = sum(flags)
        print(f"    {series:<{width}}  {k:>2}/{n}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adf", action="store_true", help="run the ADF battery")
    ap.add_argument("--kpss", action="store_true", help="run the KPSS battery")
    ap.add_argument("--arch", action="store_true", help="run the ARCH battery")
    ap.add_argument("--all", action="store_true", help="shortcut for --adf --kpss --arch")
    ap.add_argument("--workers", type=int, default=4, help="parallel worker processes (default 4)")
    ap.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="debug shortcut: only run the first N pre-analysis days",
    )
    args = ap.parse_args()

    selected = {
        "adf": args.adf or args.all,
        "kpss": args.kpss or args.all,
        "arch": args.arch or args.all,
    }
    if not any(selected.values()):
        ap.error("specify at least one of --adf / --kpss / --arch / --all")

    days = load_pre_analysis_days()
    if args.max_days is not None:
        days = days[: args.max_days]
        print(f" --max-days: only running the first {len(days)} day(s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if selected["adf"]:
        _run_battery("ADF", days, _process_adf_day, OUT_DIR / "pre_analysis_adf.json", n_workers=args.workers)
    if selected["kpss"]:
        _run_battery("KPSS", days, _process_kpss_day, OUT_DIR / "pre_analysis_kpss.json", n_workers=args.workers)
    if selected["arch"]:
        _run_battery("ARCH", days, _process_arch_day, OUT_DIR / "pre_analysis_arch.json", n_workers=args.workers)


if __name__ == "__main__":
    main()
