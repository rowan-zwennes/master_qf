"""Stage 05: checks on the labelled per-hour parquets."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"
LABEL_HORIZONS_S = [1, 5, 10, 30]
FUNDING_DAY_START_HOUR = 4

CHI2_95 = {
    1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
    6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307,
}

DEFAULT_FEATURE_COLS = [
    "spread", "spread_bps",
    "imbalance_l1", "imbalance_l2_5", "imbalance_l6_10", "imbalance_l11_20",
    "bid_depth_l1", "bid_depth_l2_5", "bid_depth_l6_10", "bid_depth_l11_20",
    "ask_depth_l1", "ask_depth_l2_5", "ask_depth_l6_10", "ask_depth_l11_20",
    "rel_bid_depth_l1", "rel_bid_depth_l2_5", "rel_bid_depth_l6_10", "rel_bid_depth_l11_20",
    "rel_ask_depth_l1", "rel_ask_depth_l2_5", "rel_ask_depth_l6_10", "rel_ask_depth_l11_20",
    "bid_slope", "ask_slope", "slope_imbalance",
    "ofi_l1", "ofi_l2_5", "ofi_l6_10", "ofi_l11_20",
    "vpin_micro", "vpin_window_micro_s",
    "vpin_base", "vpin_window_base_s",
    "vpin_macro", "vpin_window_macro_s",
    "micro_return_1s", "micro_return_5s", "micro_return_30s",
    "realized_vol_60s",
    "n_trades_1s", "buy_vol_1s", "sell_vol_1s", "signed_vol_1s",
    "trade_imbalance_1s", "largest_trade_1s", "return_1s",
    "n_trades_5s", "buy_vol_5s", "sell_vol_5s", "signed_vol_5s",
    "trade_imbalance_5s", "largest_trade_5s", "return_5s",
    "n_trades_30s", "buy_vol_30s", "sell_vol_30s", "signed_vol_30s",
    "trade_imbalance_30s", "largest_trade_30s", "return_30s",
    "funding_rate", "seconds_to_funding", "basis_bps",
    "book_age_ms", "mark_stale_age_ms",
]


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def funding_day_bounds(date_str: str) -> tuple[int, int]:
    d = parse_date(date_str).replace(hour=FUNDING_DAY_START_HOUR)
    end = d + timedelta(days=1)
    return int(d.timestamp() * 1000), int(end.timestamp() * 1000)


def _json_safe(obj):
    """Recursively replace non-finite floats (NaN, +/-inf) with None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def load_funding_day_labels(
    base: Path, fday: str, *, buffer_s: int = 0
) -> pl.DataFrame:
    """Concatenate the labelled per-hour files spanning the funding day."""
    start_ms, end_ms = funding_day_bounds(fday)
    load_end_ms = end_ms + buffer_s * 1000
    s_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    e_dt = datetime.fromtimestamp(load_end_ms / 1000, tz=timezone.utc)

    needed_dates = []
    cur = s_dt.date()
    while cur <= e_dt.date():
        needed_dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)

    frames = []
    for d in needed_dates:
        for h in range(24):
            p = base / d / f"labels_{h:02d}h.parquet"
            if p.exists():
                frames.append(pl.read_parquet(p))
    if not frames:
        return pl.DataFrame()
    full = pl.concat(frames, how="vertical_relaxed").sort("ts_ms")
    return full.filter(
        (pl.col("ts_ms") >= start_ms) & (pl.col("ts_ms") < load_end_ms)
    )


def check_funding_day_tag(df: pl.DataFrame, fday: str) -> dict:
    """The labeller stamps every row with funding_day and
    seconds_into_funding_day; verify they match this funding day. Catches a
    stale or misfiled labels_*.parquet being picked up by the loader."""
    res: dict = {"passed": True}
    if "funding_day" in df.columns:
        bad = int((df["funding_day"] != fday).sum() or 0)
        res["funding_day_mismatches"] = bad
        if bad:
            res["passed"] = False
    else:
        res["funding_day_present"] = False
    if "seconds_into_funding_day" in df.columns:
        start_ms, _ = funding_day_bounds(fday)
        expected = (df["ts_ms"] - start_ms) // 1000
        bad2 = int((df["seconds_into_funding_day"] != expected).sum() or 0)
        res["seconds_into_funding_day_mismatches"] = bad2
        if bad2:
            res["passed"] = False
    else:
        res["seconds_into_funding_day_present"] = False
    return res


def check_no_duplicates(df: pl.DataFrame) -> dict:
    n = df.height
    n_unique = df["ts_ms"].n_unique()
    return {
        "passed": n == n_unique,
        "n_rows": n,
        "n_unique_ts": n_unique,
        "n_duplicates": n - n_unique,
    }


def check_strict_1s_grid(df: pl.DataFrame, fday: str) -> dict:
    start_ms, end_ms = funding_day_bounds(fday)
    expected_rows = (end_ms - start_ms) // 1000
    diffs = df["ts_ms"].diff().drop_nulls()
    if diffs.is_empty():
        return {"passed": False, "reason": "empty"}
    bad_diffs = int((diffs != 1000).sum())
    return {
        "passed": bad_diffs == 0 and df.height == expected_rows,
        "n_rows": df.height,
        "expected_rows": expected_rows,
        "n_non_1s_gaps": bad_diffs,
        "min_diff_ms": int(diffs.min()),
        "max_diff_ms": int(diffs.max()),
    }


def check_label_causality(
    df: pl.DataFrame, horizons: list[int], end_ms: int
) -> dict:
    """For each h, verify target_mid_h[t] == mid_price[t+h] and
    target_micro_h[t] == micro_price[t+h] on funding-day rows where both sides
    are defined. `df` must be the extended frame (funding day plus a
    max(horizons)-second buffer); buffer rows act as shift targets only."""
    results: dict = {}
    in_day = df["ts_ms"] < end_ms
    for h in horizons:
        per_h: dict = {}
        for kind, price_col in (("mid", "mid_price"), ("micro", "micro_price")):
            col_target = f"target_{kind}_{h}"
            if col_target not in df.columns or price_col not in df.columns:
                per_h[kind] = {"passed": False, "reason": "column_missing"}
                continue
            computed = df[price_col].shift(-h)
            target = df[col_target]
            both = computed.is_not_null() & target.is_not_null() & in_day
            n_check = int(both.sum())
            if n_check == 0:
                per_h[kind] = {"passed": False, "reason": "no_defined_pairs"}
                continue
            diffs = (computed - target).abs().filter(both)
            max_diff = float(diffs.max() or 0.0)
            n_mismatch = int((diffs > 1e-9).fill_null(True).sum())
            per_h[kind] = {
                "passed": n_mismatch == 0,
                "n_pairs_checked": n_check,
                "n_mismatches": n_mismatch,
                "max_abs_diff": max_diff,
            }
        per_h["passed"] = all(
            v.get("passed", False)
            for v in per_h.values()
            if isinstance(v, dict)
        )
        results[h] = per_h
    return results


def check_return_drift_consistency(
    df: pl.DataFrame, horizons: list[int]
) -> dict:
    """return_*_h and drift_*_h must match their definitions exactly:

        return_mid_h == log(target_mid_h / mid_price)
        drift_mid_h  == (target_mid_h - mid_price) / h

    and likewise for micro. Recomputed from stored columns only, so no horizon
    buffer is needed."""
    results: dict = {}
    atol, rtol = 1e-9, 1e-9
    for h in horizons:
        per_h: dict = {}
        specs = [
            ("return_mid", "log", "mid_price", f"target_mid_{h}"),
            ("return_micro", "log", "micro_price", f"target_micro_{h}"),
            ("drift_mid", "drift", "mid_price", f"target_mid_{h}"),
            ("drift_micro", "drift", "micro_price", f"target_micro_{h}"),
        ]
        for name, kind, price_col, target_col in specs:
            stored_col = f"{name}_{h}"
            if (
                stored_col not in df.columns
                or price_col not in df.columns
                or target_col not in df.columns
            ):
                per_h[name] = {"passed": False, "reason": "column_missing"}
                continue
            p = df[price_col]
            t = df[target_col]
            stored = df[stored_col]
            if kind == "log":
                expected = (t / p).log()
            else:
                expected = (t - p) / h
            both = expected.is_not_null() & stored.is_not_null()
            n_check = int(both.sum())
            if n_check == 0:
                per_h[name] = {"passed": False, "reason": "no_defined_pairs"}
                continue
            exp_b = expected.filter(both)
            diff = (exp_b - stored.filter(both)).abs()
            tol = atol + rtol * exp_b.abs()
            n_mismatch = int((diff > tol).fill_null(True).sum())
            per_h[name] = {
                "passed": n_mismatch == 0,
                "n_checked": n_check,
                "n_mismatches": n_mismatch,
                "max_abs_diff": float(diff.max() or 0.0),
            }
        per_h["passed"] = all(
            v.get("passed", False)
            for v in per_h.values()
            if isinstance(v, dict)
        )
        results[h] = per_h
    return results


def check_label_valid_consistency(
    df: pl.DataFrame, horizons: list[int], end_ms: int
) -> dict:
    """label_valid_h must equal the labeller's definition exactly: valid over
    [t, t+h] and mid/micro price non-null at t and at t+h. Equality is checked
    in both directions. `df` is the extended frame; only rows with
    ts_ms < end_ms are reported on."""
    results: dict = {}
    valid = df["valid"]
    in_day = df["ts_ms"] < end_ms
    have_prices = "mid_price" in df.columns and "micro_price" in df.columns
    for h in horizons:
        col = f"label_valid_{h}"
        tmid = f"target_mid_{h}"
        tmicro = f"target_micro_{h}"
        if (
            col not in df.columns
            or not have_prices
            or tmid not in df.columns
            or tmicro not in df.columns
        ):
            results[h] = {"passed": False, "reason": "column_missing"}
            continue
        window = valid
        for k in range(1, h + 1):
            window = window & valid.shift(-k)
        expected = (
            window
            & df["mid_price"].is_not_null()
            & df[tmid].is_not_null()
            & df["micro_price"].is_not_null()
            & df[tmicro].is_not_null()
        )
        lv = df[col]
        mismatch = ((lv != expected) & in_day).fill_null(False)
        n_mismatch = int(mismatch.sum())
        results[h] = {
            "passed": n_mismatch == 0,
            "n_mismatches": n_mismatch,
            "n_label_valid_true": int((lv & in_day).fill_null(False).sum()),
        }
    return results


def check_no_nans_when_valid(df: pl.DataFrame, horizons: list[int]) -> dict:
    """When label_valid_h is True, every label series must be finite."""
    results = {}
    label_series = ["return_mid", "return_micro", "drift_mid", "drift_micro"]
    for h in horizons:
        lv = f"label_valid_{h}"
        cols = {name: f"{name}_{h}" for name in label_series}
        if lv not in df.columns or any(c not in df.columns for c in cols.values()):
            results[h] = {"passed": False, "reason": "column_missing"}
            continue
        sub = df.filter(pl.col(lv))
        if sub.is_empty():
            results[h] = {"passed": True, "n_checked": 0, "note": "no valid rows"}
            continue
        bad = {}
        for name, col in cols.items():
            bad[f"{name}_bad"] = int(
                (sub[col].is_null() | sub[col].is_nan() | sub[col].is_infinite()).sum()
            )
        results[h] = {
            "passed": all(v == 0 for v in bad.values()),
            "n_checked": sub.height,
            **bad,
        }
    return results


_DEPTH_BANDS = ("l1", "l2_5", "l6_10", "l11_20")
_BOOK_GATED_FEATURES = (
    ["mid_price", "micro_price", "spread", "spread_bps", "rel_spread",
     "weighted_mid_5", "weighted_mid_20", "book_age_ms",
     "bid_slope", "ask_slope", "slope_imbalance"]
    + [f"imbalance_{b}" for b in _DEPTH_BANDS]
    + [f"bid_depth_{b}" for b in _DEPTH_BANDS]
    + [f"ask_depth_{b}" for b in _DEPTH_BANDS]
    + [f"rel_bid_depth_{b}" for b in _DEPTH_BANDS]
    + [f"rel_ask_depth_{b}" for b in _DEPTH_BANDS]
)


def check_book_features_null_when_invalid(df: pl.DataFrame) -> dict:
    """Every book-derived feature must be NULL on rows where valid=False."""
    if "valid" not in df.columns:
        return {"passed": False, "reason": "no_valid_column"}
    invalid = df.filter(~pl.col("valid"))
    if invalid.is_empty():
        return {"passed": True, "n_invalid_rows": 0, "note": "no invalid rows"}
    present = [c for c in _BOOK_GATED_FEATURES if c in df.columns]
    if not present:
        return {"passed": False, "reason": "no_book_columns"}
    leaks = {}
    for c in present:
        n_non_null = int(invalid[c].is_not_null().sum())
        if n_non_null:
            leaks[c] = n_non_null
    return {
        "passed": len(leaks) == 0,
        "n_invalid_rows": invalid.height,
        "n_features_checked": len(present),
        "leaked_features": leaks,
    }


def _summarize_series(s: pl.Series) -> dict:
    s = s.drop_nulls().drop_nans()
    if s.is_empty():
        return {"n": 0}
    n = s.len()
    mean = float(s.mean())
    std = float(s.std() or 0.0)
    skew = float(s.skew() or 0.0)
    kurt = float(s.kurtosis() or 0.0)

    jb_stat = (n / 6.0) * (skew**2 + 0.25 * kurt**2) if n > 0 else 0.0
    jb_crit = CHI2_95[2]

    q = s.quantile
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurtosis": kurt,
        "jarque_bera": jb_stat,
        "jarque_bera_crit_5pct": jb_crit,
        "jarque_bera_rejects_normality": jb_stat > jb_crit,
        "q01": float(q(0.01)),
        "q05": float(q(0.05)),
        "q25": float(q(0.25)),
        "q50": float(q(0.5)),
        "q75": float(q(0.75)),
        "q95": float(q(0.95)),
        "q99": float(q(0.99)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def label_distributions(df: pl.DataFrame, horizons: list[int]) -> dict:
    out = {}
    for h in horizons:
        rm = f"return_mid_{h}"
        rmi = f"return_micro_{h}"
        dm = f"drift_mid_{h}"
        dmi = f"drift_micro_{h}"
        lv = f"label_valid_{h}"
        if rm not in df.columns or rmi not in df.columns or lv not in df.columns:
            continue
        valid_df = df.filter(pl.col(lv))
        entry = {
            "return_mid": _summarize_series(valid_df[rm]),
            "return_micro": _summarize_series(valid_df[rmi]),
        }
        # drift_* are the alternative HJB targets (price units / s)
        if dm in df.columns:
            entry["drift_mid"] = _summarize_series(valid_df[dm])
        if dmi in df.columns:
            entry["drift_micro"] = _summarize_series(valid_df[dmi])
        out[h] = entry
    return out


def label_autocorrelation(
    df: pl.DataFrame, horizons: list[int], lags: tuple[int, ...] = (1, 2, 3, 4, 5)
) -> dict:
    """Pearson autocorrelation of the return/drift series at each lag, plus the
    Ljung-Box Q statistic, annotated with its 5% chi-square critical value."""
    out = {}
    for h in horizons:
        lv = f"label_valid_{h}"
        if lv not in df.columns:
            continue
        valid_df = df.filter(pl.col(lv))

        horizon_res = {}
        for col_prefix in ["return_mid", "return_micro", "drift_mid", "drift_micro"]:
            col = f"{col_prefix}_{h}"
            if col not in df.columns:
                continue

            s = valid_df[col].drop_nulls().drop_nans()
            n = s.len()
            if n < max(lags) + 10:
                horizon_res[col_prefix] = {"n": n, "reason": "too_few_points"}
                continue

            def _lb_q(series: pl.Series, length: int) -> tuple[dict, float]:
                """Returns per-lag ACF dict and LB-Q scalar.

                Standard ACF estimator: single global mean, full-series variance
                in the denominator, overlapping n-lag pairs in the numerator.
                """
                acf: dict = {}
                lb = 0.0
                mu = float(series.mean())
                var = float(((series - mu) ** 2).sum())
                for lag in lags:
                    x = series.head(length - lag)
                    y = series.tail(length - lag)
                    num = float(((x - mu) * (y - mu)).sum())
                    rho = num / var if var > 0 else 0.0
                    acf[f"lag_{lag}"] = rho
                    lb += (rho**2) / (length - lag)
                return acf, length * (length + 2) * lb

            res, lb_q = _lb_q(s, n)
            res["ljung_box_q"] = lb_q
            lb_df = len(lags)
            lb_crit = CHI2_95.get(lb_df)
            res["ljung_box_df"] = lb_df
            res["ljung_box_crit_5pct"] = lb_crit
            res["ljung_box_rejects_white_noise"] = (
                lb_q > lb_crit if lb_crit is not None else None
            )
            horizon_res[col_prefix] = res
        out[h] = horizon_res
    return out


def stationarity_quartiles(df: pl.DataFrame, horizons: list[int]) -> dict:
    """Split funding-day into 4 quartiles by time, compute mean/std per quartile."""
    out = {}
    n = df.height
    if n < 16:
        return {"reason": "too_few_rows"}
    N_Q = 4
    q_size = n // N_Q
    quartile_slices = [df.slice(i * q_size, q_size) for i in range(N_Q)]

    series_names = ["return_mid", "return_micro", "drift_mid", "drift_micro"]

    for h in horizons:
        lv = f"label_valid_{h}"
        if lv not in df.columns:
            continue
        horizon_res: dict = {}
        for sname in series_names:
            col = f"{sname}_{h}"
            if col not in df.columns:
                continue
            per_q = []
            for qi, qd in enumerate(quartile_slices):
                valid = qd.filter(pl.col(lv))[col].drop_nulls().drop_nans()
                per_q.append(
                    {
                        "quartile": qi,
                        "n": int(valid.len()),
                        "mean": float(valid.mean()) if valid.len() else None,
                        "std": float(valid.std() or 0.0) if valid.len() else None,
                    }
                )
            means = [q["mean"] for q in per_q if q["mean"] is not None]
            stds = [q["std"] for q in per_q if q["std"] is not None]
            horizon_res[sname] = {
                "per_quartile": per_q,
                "var_of_quartile_means": float(pl.Series(means).var() or 0.0)
                if means
                else None,
                "var_of_quartile_stds": float(pl.Series(stds).var() or 0.0)
                if stds
                else None,
            }
        out[h] = horizon_res
    return out


def _pearson_corr(xs: pl.Series, ys: pl.Series) -> float:
    """Pearson correlation of two equal-length, finite, row-aligned series."""
    mx, my = float(xs.mean()), float(ys.mean())
    num = float(((xs - mx) * (ys - my)).sum())
    den = math.sqrt(
        float(((xs - mx) ** 2).sum()) * float(((ys - my) ** 2).sum())
    )
    return num / den if den > 0 else 0.0


def _top_k_by_abs(corrs: dict, k: int) -> dict:
    """The k entries of `corrs` with the largest absolute value."""
    ranked = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    return {f: r for f, r in ranked}


def feature_label_correlation(
    df: pl.DataFrame, horizons: list[int], feature_cols: list[str], top_k: int = 10
) -> dict:
    """Pearson AND Spearman correlation of each feature against each label series, restricted."""
    series_names = ["return_mid", "return_micro", "drift_mid", "drift_micro"]
    available_features = [c for c in feature_cols if c in df.columns]
    out = {}
    for h in horizons:
        lv = f"label_valid_{h}"
        if lv not in df.columns:
            continue
        valid_df = df.filter(pl.col(lv))
        horizon_res: dict = {}
        for sname in series_names:
            col = f"{sname}_{h}"
            if col not in df.columns:
                continue
            if valid_df.is_empty():
                horizon_res[sname] = {}
                continue
            y = valid_df[col]
            if y.drop_nulls().drop_nans().len() < 30:
                horizon_res[sname] = {"reason": "too_few_rows"}
                continue
            pearson: dict = {}
            spearman: dict = {}
            for f in available_features:
                x = valid_df[f]
                mask = x.is_not_null() & y.is_not_null() & x.is_finite() & y.is_finite()
                if int(mask.sum()) < 30:
                    continue
                xs = x.filter(mask)
                ys = y.filter(mask)
                pearson[f] = _pearson_corr(xs, ys)
                # Spearman = Pearson on average-ranked data (ties share a rank).
                spearman[f] = _pearson_corr(
                    xs.rank(method="average"), ys.rank(method="average")
                )
            horizon_res[sname] = {
                "pearson": _top_k_by_abs(pearson, top_k),
                "spearman": _top_k_by_abs(spearman, top_k),
            }
        out[h] = horizon_res
    return out


def book_pause_summary(df: pl.DataFrame) -> dict:
    """How much of the funding day is in a pause."""
    n = df.height
    if n == 0:
        return {"n": 0}
    n_invalid = int((~df["valid"]).sum())
    # Longest contiguous invalid run
    valid_int = df["valid"].cast(pl.Int8).to_list()
    max_run = 0
    cur = 0
    for v in valid_int:
        if v == 0:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return {
        "n_total_rows": n,
        "n_invalid_rows": n_invalid,
        "pct_invalid": round(n_invalid / n * 100, 4),
        "longest_pause_seconds": max_run,
    }


def mark_stale_summary(df: pl.DataFrame) -> dict:
    if "mark_is_stale" not in df.columns:
        return {"reason": "column_missing"}
    n = df.height
    if n == 0:
        return {"n": 0}
    stale = df["mark_is_stale"]
    n_defined = int(stale.is_not_null().sum())
    n_stale = int(stale.sum() or 0)
    return {
        "n_rows": n,
        "n_defined": n_defined,
        "n_stale": n_stale,
        "pct_stale": round(n_stale / n_defined * 100, 4) if n_defined else None,
    }


def feature_coverage(df: pl.DataFrame, feature_cols: list[str]) -> dict:
    """Per-feature null / non-finite coverage over the funding day."""
    n = df.height
    if n == 0:
        return {"n_rows": 0}
    per_feature: dict = {}
    all_null: list[str] = []
    missing: list[str] = []
    for c in feature_cols:
        if c not in df.columns:
            missing.append(c)
            continue
        s = df[c]
        n_null = int(s.is_null().sum())
        # is_nan / is_infinite are only defined for float dtypes.
        if s.dtype in (pl.Float32, pl.Float64):
            n_nonfinite = int(
                (s.is_nan() | s.is_infinite()).fill_null(False).sum()
            )
        else:
            n_nonfinite = 0
        per_feature[c] = {
            "frac_null": round(n_null / n, 6),
            "n_nonfinite": n_nonfinite,
        }
        if n_null == n:
            all_null.append(c)
    return {
        "n_rows": n,
        "n_features": len(feature_cols),
        "all_null": all_null,
        "missing": missing,
        "per_feature": per_feature,
    }


def validate_funding_day(
    base: Path, fday: str, *, horizons: list[int] = LABEL_HORIZONS_S
) -> dict:
    buffer_s = max(horizons)
    ext = load_funding_day_labels(base, fday, buffer_s=buffer_s)
    if ext.is_empty():
        return {"funding_day": fday, "status": "no_data"}
    _, end_ms = funding_day_bounds(fday)
    df = ext.filter(pl.col("ts_ms") < end_ms)
    if df.is_empty():
        return {"funding_day": fday, "status": "no_data"}

    report = {
        "funding_day": fday,
        "status": "ok",
        "n_rows": df.height,
        "correctness": {
            "funding_day_tag": check_funding_day_tag(df, fday),
            "no_duplicates": check_no_duplicates(df),
            "strict_1s_grid": check_strict_1s_grid(df, fday),
            "label_causality": check_label_causality(ext, horizons, end_ms),
            "return_drift_consistency": check_return_drift_consistency(df, horizons),
            "label_valid_consistency": check_label_valid_consistency(
                ext, horizons, end_ms
            ),
            "no_nans_when_valid": check_no_nans_when_valid(df, horizons),
            "book_features_null_when_invalid": check_book_features_null_when_invalid(df),
        },
        "stats": {
            "label_distributions": label_distributions(df, horizons),
            "label_autocorrelation": label_autocorrelation(df, horizons),
            "stationarity_quartiles": stationarity_quartiles(df, horizons),
            "feature_label_correlation": feature_label_correlation(
                df, horizons, DEFAULT_FEATURE_COLS
            ),
            "feature_coverage": feature_coverage(df, DEFAULT_FEATURE_COLS),
            "book_pause": book_pause_summary(df),
            "mark_stale": mark_stale_summary(df),
        },
    }

    # Roll up correctness pass/fail for fast scanning
    c = report["correctness"]
    all_pass = (
        c["funding_day_tag"].get("passed", False)
        and c["no_duplicates"].get("passed", False)
        and c["strict_1s_grid"].get("passed", False)
        and all(v.get("passed", False) for v in c["label_causality"].values())
        and all(
            v.get("passed", False)
            for v in c["return_drift_consistency"].values()
        )
        and all(
            v.get("passed", False)
            for v in c["label_valid_consistency"].values()
        )
        and all(v.get("passed", False) for v in c["no_nans_when_valid"].values())
        and c["book_features_null_when_invalid"].get("passed", False)
    )
    report["correctness_all_passed"] = all_pass
    return report


def print_report_summary(report: dict) -> None:
    fday = report["funding_day"]
    if report["status"] != "ok":
        extra = f": {report['error']}" if "error" in report else ""
        print(f"   {fday} {report['status']}{extra}")
        return
    c = report["correctness"]
    flag = "" if report["correctness_all_passed"] else ""
    pause = report["stats"]["book_pause"]
    stale = report["stats"]["mark_stale"]
    print(
        f"   {fday} {flag} rows={report['n_rows']} "
        f"pause={pause.get('pct_invalid', 0):.2f}% "
        f"stale={stale.get('pct_stale') or 0:.2f}% "
        f"longest_pause={pause.get('longest_pause_seconds', 0)}s"
    )
    if not report["correctness_all_passed"]:
        # Print which checks failed
        for name, res in c.items():
            if isinstance(res, dict):
                if "passed" in res and not res.get("passed"):
                    print(f"      {name}: {res}")
                else:
                    for h, r in res.items():
                        if isinstance(r, dict) and not r.get("passed", True):
                            print(f"      {name}[h={h}]: {r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--funding-day", help="single YYYY-MM-DD")
    p.add_argument("--from-day")
    p.add_argument("--to-day")
    p.add_argument(
        "--use-splits",
        action="store_true",
        help="Validate every funding day listed in splits.json",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Where to write JSON reports (default: <base>/validation/)",
    )
    args = p.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out_dir) if args.out_dir else (base / "validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.funding_day:
        days = [args.funding_day]
    elif args.use_splits:
        sp_path = base / "splits.json"
        if not sp_path.exists():
            raise SystemExit(
                f"--use-splits: {sp_path} not found "
                f"(run label_split.py --build-splits first)"
            )
        try:
            sp = json.loads(sp_path.read_text())
            days = sp["splits"]["pre_analysis"] + sp["splits"]["sim"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise SystemExit(f"--use-splits: {sp_path} is malformed ({e!r})")
    else:
        # Pick all available days based on label files
        cal = sorted(
            d.name
            for d in base.iterdir()
            if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"
        )
        days = []
        for d in cal:
            next_d = (parse_date(d) + timedelta(days=1)).strftime("%Y-%m-%d")
            ok = all(
                (base / d / f"labels_{h:02d}h.parquet").exists() for h in range(4, 24)
            ) and all(
                (base / next_d / f"labels_{h:02d}h.parquet").exists()
                for h in range(0, 4)
            )
            if ok:
                days.append(d)

    if args.from_day:
        days = [d for d in days if d >= args.from_day]
    if args.to_day:
        days = [d for d in days if d <= args.to_day]

    print(f"Validating {len(days)} funding days")

    summary = []
    for d in days:
        try:
            rep = validate_funding_day(base, d)
        except Exception as e:  
            rep = {"funding_day": d, "status": "error", "error": repr(e)}
        print_report_summary(rep)
        out_path = out_dir / f"funding_day_{d}.json"
        out_path.write_text(json.dumps(_json_safe(rep), indent=2, default=str))
        summary.append(
            {
                "funding_day": d,
                "status": rep.get("status"),
                "correctness_all_passed": rep.get("correctness_all_passed", False),
                "n_rows": rep.get("n_rows"),
                "pct_invalid": rep.get("stats", {})
                .get("book_pause", {})
                .get("pct_invalid"),
                "pct_stale": rep.get("stats", {})
                .get("mark_stale", {})
                .get("pct_stale"),
                "n_all_null_features": len(
                    rep.get("stats", {})
                    .get("feature_coverage", {})
                    .get("all_null", [])
                    or []
                ),
                # Representative econometric stats (10s horizon)
                "jarque_bera_10s": rep.get("stats", {})
                .get("label_distributions", {})
                .get(10, {})
                .get("return_micro", {})
                .get("jarque_bera"),
                "ljung_box_10s": rep.get("stats", {})
                .get("label_autocorrelation", {})
                .get(10, {})
                .get("return_micro", {})
                .get("ljung_box_q"),
            }
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, default=str))

    n_pass = sum(1 for s in summary if s.get("correctness_all_passed"))
    n_total = len(summary)
    n_error = sum(1 for s in summary if s.get("status") == "error")
    print(f"\n{n_pass}/{n_total} funding days passed all correctness checks.")
    if n_error:
        print(f"   {n_error} day(s) errored, see status='error' in the reports.")
    print(f"   reports written to: {out_dir}")


if __name__ == "__main__":
    main()
