"""Calibrate the VPIN bucket volume."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from feature_engineering import VPIN_NUM_BUCKETS, VPIN_SCALE_MULTIPLIERS

DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
DATASET_START = "2026-01-19"
FOLD_TRAIN_DAYS = 30          # first walk-forward fold's training window
TARGET_WINDOW_MIN = 37.5      # target wall-clock span of the base VPIN window
                              # (midpoint of the 30-45 min design range)


def _hourly_volumes(base: Path, start_date: str, n_days: int) -> pl.DataFrame:
    """Total trade Quantity per (date, hour) over the calibration window.

    Missing hourly files are skipped; an hour with zero recorded volume is
    likewise dropped so it cannot drag the median toward 0.
    """
    rows: list[dict] = []
    d0 = datetime.strptime(start_date, "%Y-%m-%d")
    for day in range(n_days):
        d = (d0 + timedelta(days=day)).strftime("%Y-%m-%d")
        for h in range(24):
            path = base / d / f"trades_{h:02d}h.parquet"
            if not path.exists():
                continue
            vol = (
                pl.scan_parquet(path)
                .select(pl.col("Quantity").cast(pl.Float64).sum())
                .collect()
                .item()
            )
            if vol is not None and vol > 0.0:
                rows.append({"date": d, "hour": h, "volume": float(vol)})
    return pl.DataFrame(rows)


def calibrate(base: Path, start_date: str, n_days: int) -> dict:
    """Compute V_base and the derived per-scale bucket volumes."""
    hv = _hourly_volumes(base, start_date, n_days)
    if hv.is_empty():
        raise SystemExit(
            f"no trade files found under {base} for "
            f"{start_date} + {n_days}d, check --base / --start-date"
        )

    median_h = float(hv["volume"].median())
    mean_h = float(hv["volume"].mean())
    v_base = median_h * TARGET_WINDOW_MIN / (VPIN_NUM_BUCKETS * 60.0)

    return {
        "dataset_start": start_date,
        "calibration_window_days": n_days,
        "hours_observed": hv.height,
        "hours_expected": n_days * 24,
        "median_hourly_volume_btc": median_h,
        "mean_hourly_volume_btc": mean_h,
        "num_buckets": VPIN_NUM_BUCKETS,
        "target_window_min": TARGET_WINDOW_MIN,
        "vpin_base_volume": v_base,
        "scale_bucket_volumes": {
            s: v_base * m for s, m in VPIN_SCALE_MULTIPLIERS.items()
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Calibrate VPIN_BASE_VOLUME on the first 30-day fold."
    )
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument(
        "--start-date",
        default=DATASET_START,
        help="first day of the dataset / first train fold (YYYY-MM-DD)",
    )
    p.add_argument("--train-days", type=int, default=FOLD_TRAIN_DAYS)
    args = p.parse_args()

    rep = calibrate(Path(args.base), args.start_date, args.train_days)

    print("\nVPIN base-volume calibration")
    print(
        f"  window           : {rep['dataset_start']} + "
        f"{rep['calibration_window_days']}d  "
        f"({rep['hours_observed']}/{rep['hours_expected']} hours present)"
    )
    print(
        f"  hourly volume    : median {rep['median_hourly_volume_btc']:.1f} "
        f"BTC/h  (mean {rep['mean_hourly_volume_btc']:.1f})"
    )
    print(
        f"  V_base           : {rep['vpin_base_volume']:.4f} BTC  "
        f"({rep['num_buckets']} buckets ~ "
        f"{rep['target_window_min']:.0f} min at typical volume)"
    )
    for s, v in rep["scale_bucket_volumes"].items():
        win = rep["target_window_min"] * VPIN_SCALE_MULTIPLIERS[s]
        print(
            f"  {s:<6} bucket   : {v:.4f} BTC  "
            f"(~{win:.0f} min typical window)"
        )

    out = Path(__file__).with_name("vpin_calibration.json")
    out.write_text(json.dumps(rep, indent=2))
    print(f"\n  written          : {out}")
    print(
        "  set feature_engineering.VPIN_BASE_VOLUME to "
        f"{rep['vpin_base_volume']:.4f}, or pass "
        f"--vpin-base-volume {rep['vpin_base_volume']:.4f}\n"
    )


if __name__ == "__main__":
    main()
