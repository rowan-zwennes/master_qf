"""Realised volatility estimates for the quoting engine."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from datetime import timedelta
from pathlib import Path

import numpy as np

try:
    import polars as pl
    _HAVE_PL = True
except Exception:  # pragma: no cover - env without polars
    _HAVE_PL = False

try:
    import pipeline_utils as pu
    _HAVE_PU = True
except Exception:  # pragma: no cover
    _HAVE_PU = False


DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
GRID_DT_S = 1.0          # features_*.parquet are on a 1 s grid
WINDOW_S_DEFAULT = 60
SPARSE_DT_S_DEFAULT = 5  # sparse-sampling spacing for the noise-robust variant
MIN_COVERAGE_DEFAULT = 0.5  # fraction of a window's increments that must be present
HALFLIFE_S_DEFAULT = 60  # EWMA half-life in seconds


@dataclass
class VolConfig:
    window_s: int = WINDOW_S_DEFAULT
    grid_dt_s: float = GRID_DT_S
    sparse_dt_s: int = SPARSE_DT_S_DEFAULT
    halflife_s: float = HALFLIFE_S_DEFAULT
    min_coverage: float = MIN_COVERAGE_DEFAULT
    price_col: str = "mid_price"


def _valid_increments(
    price: np.ndarray, ts_ms: np.ndarray, valid: np.ndarray, grid_dt_s: float
) -> np.ndarray:
    """Return DS_i = price[i]-price[i-1] for i>=1, set to NaN unless BOTH
    endpoints are valid AND the timestamps are exactly one grid step apart.
    DS has the same length as price; DS[0] is NaN by construction."""
    n = price.size
    ds = np.full(n, np.nan, dtype=float)
    if n < 2:
        return ds
    dprice = np.diff(price)
    dt_ms = np.diff(ts_ms.astype(np.int64))
    step_ok = dt_ms == int(round(grid_dt_s * 1000))
    ends_valid = valid[1:] & valid[:-1]
    finite = np.isfinite(price[1:]) & np.isfinite(price[:-1])
    good = step_ok & ends_valid & finite
    ds[1:] = np.where(good, dprice, np.nan)
    return ds


def _rolling_rv(
    ds: np.ndarray, window: int, dt_s: float, min_obs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trailing-window realised vol from increments.

    Returns (sigma, n_obs, raw_sigma) where raw_sigma is the per-step estimate
    BEFORE freezing (NaN where coverage < min_obs), sigma is forward-filled
    (frozen) across coverage gaps, and n_obs counts the present increments in
    each trailing window.
    """
    sq = ds * ds
    present = np.isfinite(ds).astype(float)
    sq_filled = np.where(np.isfinite(sq), sq, 0.0)

    csq = np.concatenate([[0.0], np.cumsum(sq_filled)])
    cpr = np.concatenate([[0.0], np.cumsum(present)])

    n = ds.size
    raw = np.full(n, np.nan, dtype=float)
    nobs = np.zeros(n, dtype=np.int64)
    for i in range(n):
        lo = max(0, i - window + 1)
        s = csq[i + 1] - csq[lo]
        c = cpr[i + 1] - cpr[lo]
        nobs[i] = int(c)
        if c >= min_obs:
            raw[i] = math.sqrt((s / c) / dt_s)

    # Freeze: carry the last finite estimate forward; never extrapolate.
    sigma = raw.copy()
    last = np.nan
    for i in range(n):
        if np.isfinite(sigma[i]):
            last = sigma[i]
        else:
            sigma[i] = last
    return sigma, nobs, raw


def _rolling_bipower(
    ds: np.ndarray, window: int, dt_s: float, min_obs: int
) -> np.ndarray:
    """Trailing-window bipower variation -> jump-robust sigma. Uses adjacent
    |DS_i|*|DS_{i-1}| products; a product is present only when both increments
    are present (no gap between them)."""
    a = np.abs(ds)
    prod = np.full(ds.size, np.nan)
    prod[1:] = a[1:] * a[:-1]
    present = np.isfinite(prod).astype(float)
    pf = np.where(np.isfinite(prod), prod, 0.0)
    cp = np.concatenate([[0.0], np.cumsum(pf)])
    cc = np.concatenate([[0.0], np.cumsum(present)])
    n = ds.size
    out = np.full(n, np.nan)
    raw = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        s = cp[i + 1] - cp[lo]
        c = cc[i + 1] - cc[lo]
        if c >= min_obs:
            raw[i] = math.sqrt((math.pi / 2.0) * (s / c) / dt_s)
    last = np.nan
    for i in range(n):
        if np.isfinite(raw[i]):
            last = raw[i]
        out[i] = last
    return out


def _rolling_sparse(
    price: np.ndarray, ts_ms: np.ndarray, valid: np.ndarray,
    window_s: int, grid_dt_s: float, sparse_dt_s: int, min_obs: int
) -> np.ndarray:
    """Microstructure-robust RV: build increments over `sparse_dt_s`-spaced
    samples (every k-th grid point), then a trailing window of width
    window_s / sparse_dt_s sparse increments. Returned on the FULL grid by
    forward-filling each sparse estimate until the next sparse point."""
    k = max(1, int(round(sparse_dt_s / grid_dt_s)))
    idx = np.arange(0, price.size, k)
    if idx.size < 3:
        return np.full(price.size, np.nan)
    sp = price[idx]
    sts = ts_ms[idx]
    sval = valid[idx]
    ds = _valid_increments(sp, sts, sval, grid_dt_s * k)
    win = max(2, int(round(window_s / sparse_dt_s)))
    sig_sparse, _, _ = _rolling_rv(ds, win, sparse_dt_s, max(2, int(min_obs / k)))
    # map back onto the full grid (step function between sparse points)
    out = np.full(price.size, np.nan)
    out[idx] = sig_sparse
    last = np.nan
    for i in range(price.size):
        if np.isfinite(out[i]):
            last = out[i]
        out[i] = last
    return out


def _rolling_bipower_sparse(
    price: np.ndarray, ts_ms: np.ndarray, valid: np.ndarray,
    window_s: int, grid_dt_s: float, sparse_dt_s: int, min_obs: int
) -> np.ndarray:
    """Jump-robust bipower variation on the sparse (`sparse_dt_s`) grid: the same sparse."""
    k = max(1, int(round(sparse_dt_s / grid_dt_s)))
    idx = np.arange(0, price.size, k)
    if idx.size < 3:
        return np.full(price.size, np.nan)
    sp = price[idx]
    sts = ts_ms[idx]
    sval = valid[idx]
    ds = _valid_increments(sp, sts, sval, grid_dt_s * k)
    win = max(2, int(round(window_s / sparse_dt_s)))
    bp_sparse = _rolling_bipower(ds, win, sparse_dt_s, max(2, int(min_obs / k)))
    out = np.full(price.size, np.nan)
    out[idx] = bp_sparse
    last = np.nan
    for i in range(price.size):
        if np.isfinite(out[i]):
            last = out[i]
        out[i] = last
    return out


def _ewma_rv(
    ds: np.ndarray, halflife_s: float, dt_s: float, min_obs: int
) -> np.ndarray:
    """Exponentially-weighted realised vol from increments."""
    lam = 0.5 ** (dt_s / halflife_s)
    n = ds.size
    out = np.full(n, np.nan)
    v = np.nan
    seen = 0
    for i in range(n):
        d = ds[i]
        if np.isfinite(d):
            x = (d * d) / dt_s
            v = x if not np.isfinite(v) else lam * v + (1.0 - lam) * x
            seen += 1
        if seen >= min_obs and np.isfinite(v):
            out[i] = math.sqrt(v)
    return out


ESTIMATOR_COLUMNS: dict[str, tuple[str, ...]] = {
    "sigma": ("sigma", "n_obs", "is_frozen"),
    "sigma_bipower": ("sigma_bipower",),
    "sigma_sparse": ("sigma_sparse",),
    "sigma_bipower_sparse": ("sigma_bipower_sparse",),
    "sigma_ewma": ("sigma_ewma",),
    "sigma_logret": ("sigma_logret",),
}
ALL_ESTIMATORS: tuple[str, ...] = tuple(ESTIMATOR_COLUMNS.keys())


def estimate_volatility(
    price: np.ndarray, ts_ms: np.ndarray, valid: np.ndarray, cfg: VolConfig,
    estimators: "set[str] | None" = None,
) -> dict[str, np.ndarray]:
    """Estimator bundle on aligned arrays (same length, sorted by ts).

    `estimators` selects which to compute (default: all of ALL_ESTIMATORS); the
    returned dict contains only the columns those estimators produce (see
    ESTIMATOR_COLUMNS). This lets a caller recompute, e.g., just `sigma_ewma`
    after a half-life change without paying for, or perturbing, the others."""
    want = set(ALL_ESTIMATORS) if estimators is None else set(estimators)
    unknown = want - set(ALL_ESTIMATORS)
    if unknown:
        raise ValueError(f"unknown estimator(s): {sorted(unknown)}; "
                         f"choose from {list(ALL_ESTIMATORS)}")

    price = np.asarray(price, dtype=float)
    ts_ms = np.asarray(ts_ms, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    window = int(round(cfg.window_s / cfg.grid_dt_s))
    min_obs = max(2, int(math.ceil(cfg.min_coverage * window)))

    ds = _valid_increments(price, ts_ms, valid, cfg.grid_dt_s)
    out: dict[str, np.ndarray] = {}

    if "sigma" in want:
        sigma, nobs, raw = _rolling_rv(ds, window, cfg.grid_dt_s, min_obs)
        out["sigma"] = sigma
        out["n_obs"] = nobs
        out["is_frozen"] = ~np.isfinite(raw)
    if "sigma_bipower" in want:
        out["sigma_bipower"] = _rolling_bipower(ds, window, cfg.grid_dt_s, min_obs)
    if "sigma_sparse" in want:
        out["sigma_sparse"] = _rolling_sparse(
            price, ts_ms, valid, cfg.window_s, cfg.grid_dt_s, cfg.sparse_dt_s, min_obs)
    if "sigma_bipower_sparse" in want:
        out["sigma_bipower_sparse"] = _rolling_bipower_sparse(
            price, ts_ms, valid, cfg.window_s, cfg.grid_dt_s, cfg.sparse_dt_s, min_obs)
    if "sigma_ewma" in want:
        out["sigma_ewma"] = _ewma_rv(ds, cfg.halflife_s, cfg.grid_dt_s, min_obs)
    if "sigma_logret" in want:
        # log-return cross-check (dimensionless basis); guard non-positive prices.
        with np.errstate(invalid="ignore", divide="ignore"):
            logp = np.where(price > 0, np.log(price), np.nan)
        dlog = _valid_increments(logp, ts_ms, valid, cfg.grid_dt_s)
        out["sigma_logret"], _, _ = _rolling_rv(dlog, window, cfg.grid_dt_s, min_obs)

    return out


def _load_features_hours(
    base: Path, paths: list[tuple[str, int]], price_col: str
) -> "pl.DataFrame | None":
    """Load the given (calendar_date, hour) features files, concatenated,
    sorted and de-duplicated on ts_ms. Missing files are skipped (a gap hour is
    absence of evidence, not zero data)."""
    frames = []
    for d, h in paths:
        p = base / d / f"features_{h:02d}h.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p).select(["ts_ms", "valid", price_col]))
    if not frames:
        return None
    return pl.concat(frames, how="vertical_relaxed").sort("ts_ms").unique("ts_ms", keep="first")


def _warmup_paths(fday: str, lead_seconds: int) -> list[tuple[str, int]]:
    """(calendar_date, hour) files covering ~lead_seconds immediately BEFORE the
    funding-day start (04:00 UTC of fday), so the first rolling window after
    04:00 is warm. The increment across 04:00 is a legitimate 1 s step, so sigma
    is continuous across the boundary, exactly as it is within a funding day."""
    n_hours = max(1, math.ceil(lead_seconds / 3600))
    start = pu.parse_date(fday).replace(hour=pu.FUNDING_DAY_START_HOUR)
    paths = []
    for i in range(n_hours, 0, -1):
        t = start - timedelta(hours=i)
        paths.append((t.strftime("%Y-%m-%d"), t.hour))
    return paths


def _merge_into_existing(
    new_cols: "pl.DataFrame", existing_path: Path, produced: list[str]
) -> "pl.DataFrame":
    """Update only the `produced` columns of an existing parquet, preserving the
    rest, aligned on ts_ms. `new_cols` carries ts_ms + the produced columns for
    this funding day's rows. Used so recomputing e.g. only sigma_ewma does not
    disturb (or require recomputing) the reviewed sigma/bipower/sparse columns."""
    old = pl.read_parquet(existing_path)
    keep = [c for c in old.columns if c == "ts_ms" or c not in produced]
    merged = old.select(keep).join(new_cols.select(["ts_ms", *produced]),
                                   on="ts_ms", how="left")
    return merged.sort("ts_ms")


def run_real(base: Path, fdays: list[str], out_dir: Path, cfg: VolConfig,
             estimators: "set[str] | None" = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    want = set(ALL_ESTIMATORS) if estimators is None else set(estimators)
    summary_path = out_dir / "volatility_summary.json"
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}
    summary = {"price_col": cfg.price_col, "window_s": cfg.window_s,
               "grid_dt_s": cfg.grid_dt_s, "halflife_s": cfg.halflife_s,
               "boundary": "funding_day_04utc",
               "funding_days": existing.get("funding_days", {})}
    for fday in fdays:
        out_path = out_dir / f"volatility_{fday}.parquet"
        compute = set(want)
        if compute != set(ALL_ESTIMATORS) and not out_path.exists():
            print(f"  {fday}: no existing parquet -> computing ALL estimators "
                  f"(requested subset {sorted(want)} ignored for a fresh file)")
            compute = set(ALL_ESTIMATORS)
        produced = [c for est in ALL_ESTIMATORS if est in compute
                    for c in ESTIMATOR_COLUMNS[est]]
        start_ms, end_ms = pu.funding_day_bounds(fday)
        # the funding day [04:00, +24h): hours 04..23 of fday + 00..03 of fday+1
        df_day = _load_features_hours(base, pu.funding_day_paths(base, fday), cfg.price_col)
        if df_day is None:
            summary["funding_days"][fday] = {"status": "no_features"}
            continue
        # warm-up tail carried across 04:00 so the first window of the day is full
        df_warm = _load_features_hours(
            base, _warmup_paths(fday, cfg.window_s * 2), cfg.price_col
        )
        if df_warm is not None:
            df = pl.concat([df_warm.tail(cfg.window_s * 2), df_day], how="vertical_relaxed")
        else:
            df = df_day
        df = df.sort("ts_ms").unique("ts_ms", keep="first")

        res = estimate_volatility(
            df[cfg.price_col].to_numpy(), df["ts_ms"].to_numpy(),
            df["valid"].to_numpy(), cfg, estimators=compute,
        )
        new_cols = pl.DataFrame({"ts_ms": df["ts_ms"],
                                 **{c: res[c] for c in produced}})
        # keep only this funding day's rows [04:00, next 04:00); drop the warm-up
        new_cols = new_cols.filter((pl.col("ts_ms") >= start_ms) & (pl.col("ts_ms") < end_ms))

        if compute == set(ALL_ESTIMATORS):
            out = new_cols  # full rewrite
        else:
            out = _merge_into_existing(new_cols, out_path, produced)
        out.write_parquet(out_path, compression="zstd")

        # Summary stats only for columns this run produced; preserve prior keys.
        day = dict(summary["funding_days"].get(fday, {}))
        day.update({"status": "ok", "rows": out.height})

        def _median_of(col: str) -> "float | None":
            if col not in out.columns:
                return None
            g = out.filter(pl.col(col).is_not_null())
            if "is_frozen" in out.columns:
                g = g.filter(~pl.col("is_frozen"))
            return float(g[col].median()) if g.height else None

        if "is_frozen" in out.columns:
            day["frozen_pct"] = round(100.0 * out["is_frozen"].mean(), 3) if out.height else None
        if "sigma" in produced:
            day["sigma_median"] = _median_of("sigma")
        if "sigma_bipower" in produced:
            day["sigma_bipower_median"] = _median_of("sigma_bipower")
        if "sigma_ewma" in produced:
            day["sigma_ewma_median"] = _median_of("sigma_ewma")
        summary["funding_days"][fday] = day
        print(f"  {fday}: rows={day['rows']:>6} "
              f"frozen={day.get('frozen_pct')}%  cols={sorted(produced)}")

    (out_dir / "volatility_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  written: {out_dir}/volatility_*.parquet (+ volatility_summary.json)")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Realised volatility sigma_hat(t) calibrator.")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--fdays", nargs="+", default=None,
                   help="explicit funding-day list YYYY-MM-DD (default: all from splits.json)")
    p.add_argument("--split", choices=["pre_analysis", "sim", "all"], default="all",
                   help="restrict to a split when --fdays not given (default: all)")
    p.add_argument("--out-dir", default=None, help="default: <base>/volatility")
    p.add_argument("--window-s", type=int, default=WINDOW_S_DEFAULT)
    p.add_argument("--sparse-dt-s", type=int, default=SPARSE_DT_S_DEFAULT)
    p.add_argument("--halflife-s", type=float, default=HALFLIFE_S_DEFAULT,
                   help="EWMA half-life for sigma_ewma (default: window-s)")
    p.add_argument("--min-coverage", type=float, default=MIN_COVERAGE_DEFAULT)
    p.add_argument("--price", choices=["mid", "micro"], default="mid",
                   help="mid_price (default, post-2026-05-30 anchor) or micro_price")
    p.add_argument("--estimators", nargs="+", default=None, choices=list(ALL_ESTIMATORS),
                   help="which estimator columns to (re)compute (default: "
                        "all); a subset merges into the existing parquet")
    args = p.parse_args()


    if not _HAVE_PL or not _HAVE_PU:
        raise SystemExit("polars/pipeline_utils unavailable")

    cfg = VolConfig(
        window_s=args.window_s, grid_dt_s=GRID_DT_S, sparse_dt_s=args.sparse_dt_s,
        halflife_s=args.halflife_s, min_coverage=args.min_coverage,
        price_col="micro_price" if args.price == "micro" else "mid_price",
    )
    base = Path(args.base)
    out_dir = Path(args.out_dir) if args.out_dir else base / "volatility"

    if args.fdays is not None:
        fdays = list(args.fdays)
    else:
        splits = pu.load_splits(base)["splits"]
        if args.split == "all":
            fdays = list(splits["pre_analysis"]) + list(splits["sim"])
        else:
            fdays = list(splits[args.split])

    estimators = set(args.estimators) if args.estimators else None
    print(f"sigma_hat calibration | price={cfg.price_col} window={cfg.window_s}s "
          f"| estimators={sorted(estimators) if estimators else 'all'} "
          f"| {len(fdays)} funding days [04:00 UTC] -> {out_dir}")
    run_real(base, fdays, out_dir, cfg, estimators=estimators)


if __name__ == "__main__":
    main()
