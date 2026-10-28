import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

"""Run the ADF, ARCH and VIF tests across all funding days."""
import json
import sys
import warnings
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from pipeline_utils import load_labels_for_funding_day, select_training_rows
from statistical_tests import vif_check, adf_test, arch_test
from ml_train import FEATURE_COLS
import numpy as np
warnings.filterwarnings("ignore")

BASE = Path("/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt")
HORIZONS = [1, 5, 10, 30]
METRICS = ["drift_micro", "return_micro", "drift_mid", "return_mid"]
SUMMARY_PATH = BASE / "validation" / "summary.json"
OUT_PATH = BASE / "validation" / "statistical_tests.json"
VIF_DAY = "2026-02-15"
MAX_WORKERS = 4


def _process_day(day: str) -> dict:
    """Load one funding day and run ADF + ARCH for all metrics at every horizon."""
    base = Path("/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt")
    raw = load_labels_for_funding_day(base, day)

    record: dict = {"funding_day": day}
    null_keys = ("adf_stat", "adf_p", "stationary", "arch_stat", "arch_p", "clustering")

    for h in [1, 5, 10, 30]:
        # select_training_rows is the same for all metrics at a given horizon
        df = select_training_rows(raw, h) if not raw.is_empty() else None

        for metric in ["drift_micro", "return_micro", "drift_mid", "return_mid"]:
            col = f"{metric}_{h}"
            prefix = f"_{metric}_{h}s"

            if df is None or col not in df.columns:
                for k in null_keys:
                    record[f"{k}{prefix}"] = None
                continue

            series = df[col].drop_nulls().to_numpy()

            if len(series) < 100:
                for k in null_keys:
                    record[f"{k}{prefix}"] = None
                continue

            adf = adf_test(series)
            arch = arch_test(series)

            record[f"adf_stat{prefix}"] = round(adf.statistic, 4)
            record[f"adf_p{prefix}"] = float(adf.p_value)
            record[f"stationary{prefix}"] = bool(adf.stationary)
            record[f"arch_stat{prefix}"] = round(arch.statistic, 4)
            record[f"arch_p{prefix}"] = float(arch.p_value)
            record[f"clustering{prefix}"] = bool(arch.heteroskedastic)

    return record


def _compute_vif(day: str) -> tuple[float, int, "pd.Series"]:
    """Compute VIF on FEATURE_COLS for one representative day."""

    warnings.filterwarnings("ignore")

    base = Path("/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt")
    raw = load_labels_for_funding_day(base, day)
    df = select_training_rows(raw, 10)

    book_cols = [c for c in FEATURE_COLS if c in df.columns]
    vif = vif_check(df.select(book_cols).to_pandas())

    finite = vif[np.isfinite(vif)]
    vif_max = float(finite.max()) if not finite.empty else float("inf")
    n_high = int((vif > 10).sum())
    return vif_max, n_high, vif


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-vif",
        action="store_true",
        help="only compute VIF, print the per-feature table and exit",
    )
    parser.add_argument(
        "--vif-day",
        default=VIF_DAY,
        help=f"Representative day for VIF (default: {VIF_DAY}).",
    )
    args = parser.parse_args()

    if args.only_vif:
        print(f"Computing VIF on {args.vif_day}...")
        vif_max, vif_n_high, vif_series = _compute_vif(args.vif_day)
        print(f"\n  max VIF (finite): {vif_max:.1f}   features > 10: {vif_n_high}\n")
        print(f"  {'feature':<40}  VIF")
        print(f"  {'-'*40}  -------")
        for feat, val in vif_series.sort_values(ascending=False).items():
            marker = " !" if val > 10 else ""
            print(f"  {feat:<40}  {val:>7.1f}{marker}")
        return

    days = [d["funding_day"] for d in json.loads(SUMMARY_PATH.read_text())]
    total = len(days)
    print(f"Running ADF + ARCH across {total} funding days at horizons {HORIZONS}s "
          f"using {MAX_WORKERS} workers...")

    records: list[dict] = [{}] * total
    day_index = {d: i for i, d in enumerate(days)}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_process_day, d): d for d in days}
        done = 0
        for fut in as_completed(futures):
            day = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                print(f"  ERROR {day}: {exc}", file=sys.stderr)
                rec = {"funding_day": day}
            records[day_index[day]] = rec
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  {done}/{total} days done", flush=True)

    print(f"\nComputing VIF on {args.vif_day}...")
    vif_max, vif_n_high, _ = _compute_vif(args.vif_day)
    print(f"  VIF max (finite features): {vif_max:.1f},  features > 10: {vif_n_high}")

    for rec in records:
        rec["vif_max"] = vif_max
        rec["vif_n_high"] = vif_n_high

    OUT_PATH.write_text(json.dumps(records, indent=2))
    print(f"\nWrote {len(records)} records to {OUT_PATH}")

    # Quick pass/fail summary
    for metric in METRICS:
        print(f"\n  {metric}:")
        for h in HORIZONS:
            n_stat = sum(1 for r in records if r.get(f"stationary_{metric}_{h}s") is True)
            n_arch = sum(1 for r in records if r.get(f"clustering_{metric}_{h}s") is True)
            print(f"    {h:2d}s: stationary {n_stat}/{total},  ARCH clustering {n_arch}/{total}")


if __name__ == "__main__":
    main()
