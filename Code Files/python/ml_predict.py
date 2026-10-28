"""Stage 07: convert the walk-forward predictions into the alpha_ML stream."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from pipeline_utils import load_labels_for_funding_day

DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
LABEL_HORIZONS_S = [1, 5, 10, 30]


def _predictions_path(
    models_dir: Path, target: str, horizon: int, model: str = "lgbm"
) -> Path:
    return (models_dir / f"{model}_production_{target}_h{horizon}"
            / f"oos_predictions_h{horizon}.parquet")


def _anchor_col(target: str) -> str:
    return "mid_price" if target.endswith("_mid") else "micro_price"


def _load_anchor_price(base: Path, day: str, target: str) -> pl.DataFrame:
    """Return (ts_ms, <anchor col>) for `day`; empty frame if day missing."""
    col = _anchor_col(target)
    raw = load_labels_for_funding_day(base, day)
    if raw.is_empty():
        return pl.DataFrame(schema={"ts_ms": pl.Int64, col: pl.Float64})
    if col not in raw.columns:
        raise SystemExit(
            f"{col} column absent from labelled parquet for {day}. "
            "Re-run Stage 03/04 feature engineering."
        )
    return raw.select(["ts_ms", col])


def convert_day(
    day_preds: pl.DataFrame,
    prices: pl.DataFrame,
    target: str,
    horizon: int,
) -> pl.DataFrame:
    """Join the anchor price and apply the unit conversion for one day."""
    col = _anchor_col(target)
    joined = day_preds.join(prices, on="ts_ms", how="left")

    n_null = joined[col].null_count()
    if n_null > 0:
        print(
            f"       {n_null}/{joined.height} rows have no matching "
            f"{col}, dropping"
        )
        joined = joined.drop_nulls(col)

    if joined.is_empty():
        return pl.DataFrame(schema={
            "ts_ms": pl.Int64, "fold": pl.Int32, "test_day": pl.String,
            "y_pred_raw": pl.Float64, "anchor_price": pl.Float64,
            "alpha_ml_usdt_s": pl.Float64,
        })

    if target.startswith("return_"):
        alpha = (pl.col(col) * pl.col("y_pred") / horizon).alias("alpha_ml_usdt_s")
    else:
        # drift_*: y_pred is already (target_price - anchor_price) / h, USDT/s.
        alpha = pl.col("y_pred").alias("alpha_ml_usdt_s")

    return (
        joined
        .with_columns(alpha)
        .select(["ts_ms", "fold", "test_day", "y_pred", col, "alpha_ml_usdt_s"])
        .rename({"y_pred": "y_pred_raw", col: "anchor_price"})
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--horizon", type=int, default=10, choices=LABEL_HORIZONS_S,
                   help="label horizon in seconds (must match the ml_train.py run)")
    p.add_argument("--target", default="drift_mid",
                   choices=["return_micro", "return_mid",
                            "drift_micro", "drift_mid"],
                   help="target family used in ml_train.py "
                        "(production default: drift_mid)")
    p.add_argument("--model", default="lgbm", choices=["lgbm", "ar"],
                   help="which ml_train.py --model run to convert. 'ar' is "
                        "the price-only AR(1) benchmark of strategy 3; the "
                        "conversion, the column names and the row grid are "
                        "identical, only the source directory and the output "
                        "file name differ (alpha_ar_h<h>.parquet), so the "
                        "replay engine loads either stream with one loader.")
    p.add_argument("--models-dir", default=None,
                   help="root of ml_train.py model output "
                        "(default <base>/models)")
    p.add_argument("--out-dir", default=None,
                   help="output directory (default <models-dir>/lgbm_production_<target>_h<h>/)")
    args = p.parse_args()

    base = Path(args.base)
    models_dir = Path(args.models_dir) if args.models_dir else base / "models"

    preds_path = _predictions_path(
        models_dir, args.target, args.horizon, model=args.model
    )
    if not preds_path.exists():
        raise SystemExit(
            f"prediction parquet not found: {preds_path}\n"
            f"Run: python ml_train.py --mode production --model {args.model} "
            f"--target {args.target} --horizon {args.horizon}"
        )
    # File name only; the schema is identical for both models.
    alpha_stem = "alpha_ml" if args.model == "lgbm" else "alpha_ar"

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else preds_path.parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _label = "α_ML" if args.model == "lgbm" else "α_AR"
    print(f"{_label} offline converter | target={args.target} h={args.horizon}s")
    print(f"   predictions: {preds_path}")
    print(f"   out:         {out_dir}")

    preds = pl.read_parquet(preds_path)
    test_days = preds["test_day"].unique(maintain_order=False).sort().to_list()
    print(f"   {preds.height} prediction rows across {len(test_days)} test days")

    result_frames: list[pl.DataFrame] = []
    n_dropped_total = 0

    for day in test_days:
        day_preds = preds.filter(pl.col("test_day") == day)
        prices = _load_anchor_price(base, day, args.target)

        if prices.is_empty():
            print(f"    {day}: no labelled data, skipping {day_preds.height} rows")
            n_dropped_total += day_preds.height
            continue

        converted = convert_day(day_preds, prices, args.target, args.horizon)
        n_dropped_total += day_preds.height - converted.height

        if not converted.is_empty():
            result_frames.append(converted)

    if not result_frames:
        raise SystemExit("no predictions could be converted, check data availability")

    out = pl.concat(result_frames, how="vertical").sort("ts_ms")

    alpha = out["alpha_ml_usdt_s"]
    print(
        f"\n   converted {out.height} rows "
        f"({n_dropped_total} dropped due to missing anchor price)"
    )
    print(
        f"   {_label} stats:  mean={alpha.mean():.4e}  std={alpha.std():.4e}"
        f"  p1={alpha.quantile(0.01):.4e}  p99={alpha.quantile(0.99):.4e}"
    )

    out_path = out_dir / f"{alpha_stem}_h{args.horizon}.parquet"
    out.write_parquet(out_path, compression="zstd")
    print(f"   -> {out_path}")

    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "polars_version": pl.__version__,
        "args": vars(args),
        "model": args.model,
        "source_parquet": str(preds_path),
        "target": args.target,
        "horizon_s": args.horizon,
        "conversion_formula": (
            "anchor_price * y_pred / h" if args.target.startswith("return_")
            else "y_pred (drift targets already in USDT/s)"
        ),
        "anchor_column": _anchor_col(args.target),
        "n_rows_in": preds.height,
        "n_rows_out": out.height,
        "n_rows_dropped": n_dropped_total,
        "test_days": test_days,
    }
    (out_dir / f"run_config_{alpha_stem}_h{args.horizon}.json").write_text(
        json.dumps(config, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
