"""Out-of-sample directional accuracy against signal strength."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

MODELS = Path("/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt/models")
CELLS = [("LightGBM", "lgbm_production_drift_mid_h{h}"),
         ("AR(1)", "ar_production_drift_mid_h{h}")]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Directional accuracy under both conventions, plus the decile profile."""
    called = y_pred != 0.0
    moved = y_true != 0.0
    hit = np.sign(y_pred) == np.sign(y_true)
    both = called & moved
    a = np.abs(y_pred)
    edges = np.quantile(a[called], np.arange(0.0, 1.01, 0.1))
    prof = []
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        sel = both & (a >= lo) & ((a <= hi) if i == 9 else (a < hi))
        prof.append(float(hit[sel].mean()) if sel.any() else float("nan"))
    return {
        "n": int(moved.sum()),
        "strict": float(hit[moved].mean()),
        "on_a_call": float(hit[both].mean()),
        "zero_pred": float((~called).mean()),
        "sd_pred": float(y_pred.std()),
        "sd_true": float(y_true.std()),
        "r2": 1.0 - float(((y_pred - y_true) ** 2).sum()
                          / ((y_true - y_true.mean()) ** 2).sum()),
        "deciles": prof,
    }


def per_day_top_decile(day: np.ndarray, y_true: np.ndarray,
                       y_pred: np.ndarray) -> np.ndarray:
    """Top-decile accuracy computed within each day, then returned per day.

    Pooling deciles across 64 days lets a few high-amplitude folds populate the
    top bucket, which would make the headline contrast a statement about fold
    heterogeneity rather than about signal strength. Cutting the decile inside
    each day removes that route.
    """
    called, moved = y_pred != 0.0, y_true != 0.0
    hit = np.sign(y_pred) == np.sign(y_true)
    out = []
    for d in np.unique(day):
        s = (day == d) & called
        if s.sum() < 10:
            continue
        thr = np.quantile(np.abs(y_pred[s]), 0.9)
        sel = s & moved & (np.abs(y_pred) >= thr)
        if sel.any():
            out.append(float(hit[sel].mean()))
    return np.array(out)


def report(models_dir: Path, horizon: int) -> None:
    for name, stem in CELLS:
        p = models_dir / stem.format(h=horizon) / f"oos_predictions_h{horizon}.parquet"
        d = pl.read_parquet(p).drop_nulls(["y_true", "y_pred"])
        y = d["y_true"].to_numpy()
        f = d["y_pred"].to_numpy()
        r = accuracy(y, f)
        top = per_day_top_decile(d["test_day"].to_numpy(), y, f)
        print(f"\n{name}  ({d.height:,} OOS rows, {d['test_day'].n_unique()} days)")
        print(f"  {p}")
        print(f"  directional accuracy  strict {r['strict']:.4f}   "
              f"on-a-call {r['on_a_call']:.4f}   "
              f"(zero forecasts {100 * r['zero_pred']:.2f}% of rows)")
        print(f"  pooled OOS R2 {r['r2']:+.5f}   "
              f"SD pred {r['sd_pred']:.4f} vs SD target {r['sd_true']:.4f}")
        print("  accuracy by decile of |signal|, weakest to strongest:")
        print("    " + "  ".join(f"{100 * v:.1f}" for v in r["deciles"]))
        print(f"  top decile cut within each day: mean {100 * top.mean():.2f}%  "
              f"median {100 * np.median(top):.2f}%  "
              f"min {100 * top.min():.1f}%  max {100 * top.max():.1f}%  "
              f"days above 50%: {int((top > 0.5).sum())}/{len(top)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", type=Path, default=MODELS)
    ap.add_argument("--horizon", type=int, default=10)
    args = ap.parse_args()
    report(args.models_dir, args.horizon)


if __name__ == "__main__":
    main()
