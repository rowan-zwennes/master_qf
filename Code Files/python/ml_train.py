"""Stage 06: LightGBM walk-forward trainer for the drift signal."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

import lightgbm as lgb
import optuna

from pipeline_utils import (
    funding_day_bounds,
    load_labels_for_funding_day,
    load_splits,
    select_training_rows,
)
from compute_purge_window import compute_tau_for_days

DEFAULT_BASE = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt"

PRE_ANALYSIS_WINDOW = 14
PRODUCTION_WINDOW = 30


DEFAULT_RETUNE_EVERY_DAYS = 7

LABEL_HORIZONS_S = [1, 5, 10, 30]

TARGET_FAMILIES = ["return_micro", "return_mid", "drift_micro", "drift_mid"]

FEATURE_COLS: list[str] = [
    # --- static book state (OBI and shape) ---
    "spread", "spread_bps",
    "imbalance_l1", "imbalance_l2_5", "imbalance_l6_10", "imbalance_l11_20",
    "bid_slope", "ask_slope", "slope_imbalance",
    "rel_bid_depth_l1", "rel_bid_depth_l2_5", "rel_bid_depth_l6_10", "rel_bid_depth_l11_20",
    "rel_ask_depth_l1", "rel_ask_depth_l2_5", "rel_ask_depth_l6_10", "rel_ask_depth_l11_20",
    "book_age_ms",
    # --- order-flow imbalance (OFI) ---
    "ofi_l1", "ofi_l2_5", "ofi_l6_10", "ofi_l11_20",
    # --- VPIN (multiscale) ---
    "vpin_micro", "vpin_window_micro_s",
    "vpin_base", "vpin_window_base_s",
    "vpin_macro", "vpin_window_macro_s",
    # --- trade-window intensities (windows 1s / 5s / 30s) ---
    "n_trades_1s", "buy_vol_1s", "sell_vol_1s", "signed_vol_1s",
    "trade_imbalance_1s", "largest_trade_1s", "return_1s",
    "n_trades_5s", "buy_vol_5s", "sell_vol_5s", "signed_vol_5s",
    "trade_imbalance_5s", "largest_trade_5s", "return_5s",
    "n_trades_30s", "buy_vol_30s", "sell_vol_30s", "signed_vol_30s",
    "trade_imbalance_30s", "largest_trade_30s", "return_30s",
    # --- micro-price log-returns (drift signal alive in no-trade seconds) ---
    "micro_return_1s", "micro_return_5s", "micro_return_30s",
    # --- realised volatility (recent noise level) ---
    "realized_vol_60s",
    "seconds_to_funding", "basis_bps", "mark_stale_age_ms",
]


def ar_feature_name(horizon_s: int) -> str:
    return f"ar_ret_mid_{horizon_s}s"


def add_ar_feature(raw: pl.DataFrame, horizon_s: int) -> pl.DataFrame:
    """Append the lagged mid drift (mid(t) - mid(t-h)) / h, in USDT/s."""
    col = ar_feature_name(horizon_s)
    lag = int(horizon_s)
    win = lag + 1
    ok_grid = (
        (pl.col("ts_ms") - pl.col("ts_ms").shift(lag)) == lag * 1000
    ).fill_null(False)
    ok_valid = (
        pl.col("valid").fill_null(False).cast(pl.Int32)
        .rolling_sum(window_size=win) == win
    ).fill_null(False)
    return raw.sort("ts_ms").with_columns(
        pl.when(ok_grid & ok_valid)
        .then((pl.col("mid_price") - pl.col("mid_price").shift(lag)) / float(lag))
        .otherwise(None)
        .alias(col)
    )


def fit_ols_origin(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares through the origin. Returns the coefficient vector."""
    beta, *_ = np.linalg.lstsq(X.astype(np.float64), y.astype(np.float64), rcond=None)
    return beta

DIFF_FEATURES: list[str] = [
    "vpin_window_micro_s",
    "vpin_window_base_s",
    "vpin_window_macro_s",
    "vpin_macro",
]

DEFAULT_LGB_PARAMS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.02,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_data_in_leaf": 500,
    "lambda_l1": 0.0,
    "lambda_l2": 5.0,
    "max_depth": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
}

EARLY_STOPPING_ROUNDS = 100

MAX_BOOST_ROUNDS = 3000

MAX_BOOST_ROUNDS_PRE_ANALYSIS_SHORT = 500

OPTUNA_SEARCH_SPACE: dict[str, dict] = {
    "num_leaves":        {"type": "int",   "low": 15,    "high": 127,   "log": True},
    "learning_rate":     {"type": "float", "low": 0.01,  "high": 0.05,  "log": True},
    "min_data_in_leaf":  {"type": "int",   "low": 500,   "high": 10000, "log": True},
    "feature_fraction":  {"type": "float", "low": 0.5,   "high": 0.9,   "log": False},
    "bagging_fraction":  {"type": "float", "low": 0.5,   "high": 0.9,   "log": False},
    "lambda_l2":         {"type": "float", "low": 0.1,   "high": 100.0, "log": True},
}

DEFAULT_N_TRIALS = 40
DEFAULT_N_TRIALS_PRE_ANALYSIS_SHORT = 20

OPTUNA_N_STARTUP_TRIALS = 5


def sample_lgb_params(trial: optuna.trial.Trial, base_params: dict) -> dict:
    """Sample one point from `OPTUNA_SEARCH_SPACE` and overlay it on the
    fixed base LightGBM parameters (objective, deterministic flags, seed,
    num_threads, etc.). Returns a fresh dict; `base_params` is not mutated.
    """
    params = dict(base_params)
    for name, spec in OPTUNA_SEARCH_SPACE.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], log=spec["log"]
            )
        else:
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec["log"]
            )
    return params


def make_inner_folds(window_days: list[str]) -> list[dict]:
    n = len(window_days)
    if n == PRE_ANALYSIS_WINDOW:
        val_sizes = [2, 2, 2]
        initial_train = 8
    elif n == PRODUCTION_WINDOW:
        val_sizes = [3, 3, 4]
        initial_train = 20
    else:
        raise ValueError(
            f"unexpected inner-CV window size {n}; expected "
            f"{PRE_ANALYSIS_WINDOW} or {PRODUCTION_WINDOW}"
        )

    folds: list[dict] = []
    train_end = initial_train
    for k, vs in enumerate(val_sizes):
        folds.append({
            "inner_fold": k,
            "train_days": window_days[:train_end],
            "val_days": window_days[train_end:train_end + vs],
        })
        train_end += vs
    return folds


def build_outer_folds(
    all_days: list[str],
    *,
    mode: str,
    n_pre_analysis: int,
    max_folds: int | None = None,
) -> list[dict]:
    """Construct chronological outer walk-forward folds."""
    folds: list[dict] = []
    if mode == "pre_analysis":
        window = PRE_ANALYSIS_WINDOW
        pre = all_days[:n_pre_analysis]
        n_test = len(pre) - window
        for f in range(n_test):
            folds.append({
                "fold": f,
                "train_days": pre[f:f + window],
                "test_day": pre[f + window],
            })
    elif mode == "production":
        window = PRODUCTION_WINDOW
        n_test = len(all_days) - window
        for f in range(n_test):
            folds.append({
                "fold": f,
                "train_days": all_days[f:f + window],
                "test_day": all_days[f + window],
            })
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    if max_folds is not None:
        folds = folds[:max_folds]
    return folds


def build_xy(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    *,
    require_finite_x: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn a filtered labelled frame into (X, y, ts_ms) numpy arrays.
    """
    X = (
        df.select([pl.col(c).cast(pl.Float64) for c in feature_cols])
        .to_numpy()
        .astype(np.float32)
    )
    y = (
        df.select(pl.col(target_col).cast(pl.Float64))
        .to_numpy()
        .ravel()
        .astype(np.float64)
    )
    ts = df.select(pl.col("ts_ms")).to_numpy().ravel().astype(np.int64)
    finite = np.isfinite(y)
    if require_finite_x and X.size:
        finite &= np.isfinite(X).all(axis=1)
    if not finite.all():
        X, y, ts = X[finite], y[finite], ts[finite]
    return X, y, ts


class DayCache:
    """Loads, filters and converts each funding day exactly once. The same
    day recurs across many folds (as train, then test); caching the numpy
    arrays avoids re-reading and re-parsing the parquet on every fold."""

    def __init__(
        self,
        base: Path,
        feature_cols: list[str],
        target_col: str,
        horizon: int,
        *,
        drop_stale: bool,
        label_stem: str = "labels",
        ar_horizon: int | None = None,
        require_finite_x: bool = False,
    ) -> None:
        self.base = base
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.horizon = horizon
        self.drop_stale = drop_stale
        self.label_stem = label_stem
        self.ar_horizon = ar_horizon
        self.require_finite_x = require_finite_x
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get(self, fday: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hit = self._cache.get(fday)
        if hit is not None:
            return hit
        raw = load_labels_for_funding_day(self.base, fday, stem=self.label_stem)
        if raw.is_empty():
            empty = (
                np.empty((0, len(self.feature_cols)), dtype=np.float32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
            )
            self._cache[fday] = empty
            return empty
        if self.ar_horizon is not None:
            raw = add_ar_feature(raw, self.ar_horizon)
        diff_cols = [c for c in DIFF_FEATURES if c in raw.columns]
        if diff_cols:
            raw = raw.sort("ts_ms").with_columns(
                [pl.col(c).diff().alias(c) for c in diff_cols]
            )
        rows = select_training_rows(
            raw, self.horizon, drop_stale=self.drop_stale, drop_invalid_book=True
        )
        xy = build_xy(
            rows, self.feature_cols, self.target_col,
            require_finite_x=self.require_finite_x,
        )
        self._cache[fday] = xy
        return xy

    def stack(
        self,
        days: list[str],
        *,
        max_ts_ms: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Concatenate several funding days into one matrix."""
        xs, ys, tss = [], [], []
        for d in days:
            X, y, ts = self.get(d)
            if max_ts_ms is not None and ts.size:
                keep = ts < max_ts_ms
                if not keep.all():
                    X, y, ts = X[keep], y[keep], ts[keep]
            if X.shape[0]:
                xs.append(X)
                ys.append(y)
                tss.append(ts)
        if not xs:
            return (
                np.empty((0, len(self.feature_cols)), dtype=np.float32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
            )
        return np.vstack(xs), np.concatenate(ys), np.concatenate(tss)


def _effective_purge(raw_sec: int) -> int:
    """Effective purge window: the larger of the empirical tau and the longest
    label horizon. The label-horizon floor guarantees no training row's
    forward label window reaches into the held-out period."""
    return max(raw_sec, max(LABEL_HORIZONS_S))


def _load_preanalysis_tau(base: Path) -> int:
    """Load the pre-analysis purge window from validation/purge_window.json.

    Run `python compute_purge_window.py` to generate this file before
    launching ml_train.py or ml_baseline_ridge.py.
    """
    path = base / "validation" / "purge_window.json"
    if not path.exists():
        raise SystemExit(
            f"purge_window.json not found at {path}. "
            "Run `python compute_purge_window.py` first."
        )
    return int(json.loads(path.read_text())["tau_purge_sec"])


def train_max_ts_ms(seam_day: str, purge_sec: int) -> int:
    """Largest train `ts_ms` (exclusive) when the next day is `seam_day`."""
    seam_start_ms, _ = funding_day_bounds(seam_day)
    return seam_start_ms - _effective_purge(purge_sec) * 1000


def assert_purge_invariant(
    ttr: np.ndarray, seam_day: str, horizon_s: int, purge_sec: int
) -> None:
    """Raise if any retained training row's forward label window reaches into the
    seam day. Empty training arrays no-op."""
    if ttr.size == 0:
        return
    seam_start_ms, _ = funding_day_bounds(seam_day)
    last_label_ms = int(ttr.max()) + horizon_s * 1000
    if last_label_ms >= seam_start_ms:
        eff = _effective_purge(purge_sec)
        raise AssertionError(
            f"purge invariant violated for seam {seam_day}: last train "
            f"ts_ms={int(ttr.max())} + horizon={horizon_s}s reaches "
            f"{last_label_ms}, but seam starts at {seam_start_ms} "
            f"(purge_sec={purge_sec}, effective={eff})"
        )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, out-of-sample R2, and directional accuracy."""
    n = y_true.shape[0]
    if n == 0:
        return {"n": 0, "rmse": None, "r2": None, "dir_acc": None,
                "y_true_mean": None, "y_true_std": None,
                "y_pred_mean": None, "y_pred_std": None}
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    # Directional accuracy excludes exactly-flat truths (no move to call).
    nonzero = y_true != 0.0
    if int(nonzero.sum()) > 0:
        dir_acc = float(
            np.mean(np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero]))
        )
    else:
        dir_acc = float("nan")
    return {
        "n": int(n),
        "rmse": rmse,
        "r2": float(r2),
        "dir_acc": dir_acc,
        "y_true_mean": float(y_true.mean()),
        "y_true_std": float(y_true.std()),
        "y_pred_mean": float(y_pred.mean()),
        "y_pred_std": float(y_pred.std()),
    }


def measure_latency(model: lgb.Booster, X: np.ndarray) -> dict:
    """
    Inference-latency probe. `batch` is amortised throughput over a large
    block; `single_*` are p50/p99 of one-row predict calls. Both run through
    the Python binding and so over-state the true cost; the headline number
    is the C++ hot path.
    """
    if X.shape[0] == 0:
        return {"batch_ns_per_row": None, "single_p50_ns": None,
                "single_p99_ns": None}
    batch = X[: min(50_000, X.shape[0])]
    t0 = time.perf_counter_ns()
    model.predict(batch)
    t1 = time.perf_counter_ns()
    batch_ns = (t1 - t0) / batch.shape[0]

    n_single = min(2_000, X.shape[0])
    samples = np.empty(n_single, dtype=np.float64)
    for i in range(n_single):
        row = X[i : i + 1]
        s0 = time.perf_counter_ns()
        model.predict(row)
        samples[i] = time.perf_counter_ns() - s0
    return {
        "batch_ns_per_row": float(batch_ns),
        "single_p50_ns": float(np.percentile(samples, 50)),
        "single_p99_ns": float(np.percentile(samples, 99)),
    }


def tune_via_inner_cv(
    train_days: list[str],
    cache: DayCache,
    feature_cols: list[str],
    base_params: dict,
    purge_sec: int,
    n_trials: int,
    optuna_seed: int,
    max_boost_rounds: int,
) -> tuple[dict, int, list[int], dict]:
    """Run an Optuna study over `OPTUNA_SEARCH_SPACE` using the inner expanding 3-fold CV."""
    inner_folds = make_inner_folds(train_days)

    inner_data: list[dict] = []
    for ifold in inner_folds:
        seam_day = ifold["val_days"][0]
        seam_max_ts = train_max_ts_ms(seam_day, purge_sec)
        Xi_tr, yi_tr, ti_tr = cache.stack(ifold["train_days"], max_ts_ms=seam_max_ts)
        Xi_va, yi_va, _ = cache.stack(ifold["val_days"])
        assert_purge_invariant(ti_tr, seam_day, cache.horizon, purge_sec)
        if Xi_tr.shape[0] == 0 or Xi_va.shape[0] == 0:
            continue
        inner_data.append({
            "Xi_tr": Xi_tr, "yi_tr": yi_tr,
            "Xi_va": Xi_va, "yi_va": yi_va,
            "seam_day": seam_day,
        })

    empty_study_info = {
        "n_trials": 0, "n_complete": 0, "n_pruned": 0,
        "best_value": None, "best_params": {},
    }
    if not inner_data:
        return dict(base_params), max_boost_rounds, [], empty_study_info

    def objective(trial: optuna.trial.Trial) -> float:
        params = sample_lgb_params(trial, base_params)
        per_fold_best_iters: list[int] = []
        per_fold_rmse: list[float] = []
        for k, d in enumerate(inner_data):
            dtr = lgb.Dataset(
                d["Xi_tr"], label=d["yi_tr"], feature_name=feature_cols
            )
            dva = lgb.Dataset(
                d["Xi_va"], label=d["yi_va"],
                feature_name=feature_cols, reference=dtr,
            )
            booster = lgb.train(
                params, dtr,
                num_boost_round=max_boost_rounds,
                valid_sets=[dva], valid_names=["val"],
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            best_iter = max(1, int(booster.best_iteration or max_boost_rounds))
            val_rmse = float(booster.best_score["val"]["rmse"])
            per_fold_best_iters.append(best_iter)
            per_fold_rmse.append(val_rmse)
            trial.report(val_rmse, step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("best_iters", per_fold_best_iters)
        return float(np.mean(per_fold_rmse))

    sampler = optuna.samplers.TPESampler(seed=optuna_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=OPTUNA_N_STARTUP_TRIALS,
        n_warmup_steps=0,
    )
    study = optuna.create_study(
        direction="minimize", sampler=sampler, pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    n_complete = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    n_pruned = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    )
    base_study_info = {
        "n_trials": len(study.trials),
        "n_complete": n_complete,
        "n_pruned": n_pruned,
    }
    if n_complete == 0:
        return dict(base_params), max_boost_rounds, [], {
            **base_study_info, "best_value": None, "best_params": {},
        }

    best_trial = study.best_trial
    best_iters = list(best_trial.user_attrs.get("best_iters", []))
    if not best_iters:
        return dict(base_params), max_boost_rounds, [], {
            **base_study_info,
            "best_value": float(best_trial.value),
            "best_params": dict(best_trial.params),
        }
    calibrated_rounds = max(1, max(best_iters))
    best_params = dict(base_params)
    best_params.update(best_trial.params)
    study_info = {
        **base_study_info,
        "best_value": float(best_trial.value),
        "best_params": dict(best_trial.params),
    }
    return best_params, calibrated_rounds, best_iters, study_info


def train_fold(
    fold: dict,
    cache: DayCache,
    feature_cols: list[str],
    lgb_params: dict,
    calibrated_rounds: int,
    boundary_purge_sec: int,
) -> dict | None:
    """Refit on the entire training window for `calibrated_rounds` (no early stopping, no."""
    seam_day = fold["test_day"]
    seam_max_ts = train_max_ts_ms(seam_day, boundary_purge_sec)
    Xtr, ytr, ttr = cache.stack(fold["train_days"], max_ts_ms=seam_max_ts)
    Xte, yte, tste = cache.stack([seam_day])
    assert_purge_invariant(ttr, seam_day, cache.horizon, boundary_purge_sec)

    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        print(
            f"   fold {fold['fold']:02d}  skipped "
            f"(train={Xtr.shape[0]} test={Xte.shape[0]})"
        )
        return None

    t_start = time.perf_counter()
    dtrain = lgb.Dataset(Xtr, label=ytr, feature_name=feature_cols)
    booster = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=calibrated_rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    fit_seconds = time.perf_counter() - t_start

    y_pred = booster.predict(Xte)
    metrics = regression_metrics(yte, y_pred)
    latency = measure_latency(booster, Xte)

    return {
        "fold": fold["fold"],
        "booster": booster,
        "calibrated_rounds": calibrated_rounds,
        "fit_seconds": fit_seconds,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "test_day": fold["test_day"],
        "metrics": metrics,
        "latency": latency,
        "gain_importance": booster.feature_importance(importance_type="gain"),
        "split_importance": booster.feature_importance(importance_type="split"),
        "pred_ts_ms": tste,
        "y_true": yte,
        "y_pred": y_pred,
    }


def train_fold_ar(
    fold: dict,
    cache: DayCache,
    feature_cols: list[str],
    boundary_purge_sec: int,
) -> dict | None:
    """AR(1) analogue of `train_fold`: same folds, same purge, same seam
    handling, OLS through the origin instead of LightGBM. Returns the same
    result dict shape so the driver loop is shared, with `booster=None` and
    the fitted coefficients under `beta`."""
    seam_day = fold["test_day"]
    seam_max_ts = train_max_ts_ms(seam_day, boundary_purge_sec)
    Xtr, ytr, ttr = cache.stack(fold["train_days"], max_ts_ms=seam_max_ts)
    Xte, yte, tste = cache.stack([seam_day])
    assert_purge_invariant(ttr, seam_day, cache.horizon, boundary_purge_sec)

    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        print(
            f"   fold {fold['fold']:02d}  skipped "
            f"(train={Xtr.shape[0]} test={Xte.shape[0]})"
        )
        return None

    t_start = time.perf_counter()
    beta = fit_ols_origin(Xtr, ytr)
    fit_seconds = time.perf_counter() - t_start

    y_pred = Xte.astype(np.float64) @ beta
    metrics = regression_metrics(yte, y_pred)
    ss_tot = float(np.sum(ytr ** 2))
    r2_train = (
        1.0 - float(np.sum((ytr - Xtr.astype(np.float64) @ beta) ** 2)) / ss_tot
        if ss_tot > 0 else float("nan")
    )

    return {
        "fold": fold["fold"],
        "booster": None,
        "beta": beta,
        "r2_train": r2_train,
        "calibrated_rounds": 0,
        "fit_seconds": fit_seconds,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "test_day": fold["test_day"],
        "metrics": metrics,
        "latency": {"batch_ns_per_row": None,
                    "single_p50_ns": None,
                    "single_p99_ns": None},
        "gain_importance": np.abs(beta),
        "split_importance": np.zeros(len(feature_cols), dtype=np.float64),
        "pred_ts_ms": tste,
        "y_true": yte,
        "y_pred": y_pred,
    }


def resolve_feature_cols(
    base: Path,
    sample_day: str,
    feature_cols: list[str] | None = None,
    stem: str = "labels",
) -> list[str]:
    """Intersect the allowlist with the columns actually on disk; warn on any
    feature the Stage 03 output does not provide."""
    allowlist = FEATURE_COLS if feature_cols is None else feature_cols
    df = load_labels_for_funding_day(base, sample_day, stem=stem)
    if df.is_empty():
        raise SystemExit(
            f"cannot resolve features: no labelled data for {sample_day}"
        )
    available = set(df.columns)
    present = [c for c in allowlist if c in available]
    missing = [c for c in allowlist if c not in available]
    if missing:
        print(f" {len(missing)} allowlisted features absent from the "
              f"labelled parquet, dropping them: {missing}")
    if not present:
        raise SystemExit("no usable features, check Stage 03 output schema")
    return present


def load_split_days(base: Path) -> tuple[list[str], int]:
    splits = load_splits(base)["splits"]
    if "pre_analysis" not in splits or "sim" not in splits:
        raise SystemExit(
            "splits.json is in the legacy format (train/val/sim). "
            "Re-run `python label_split.py --build-splits` to regenerate it "
            "with the new pre_analysis / sim partition."
        )
    pre = list(splits["pre_analysis"])
    sim = list(splits["sim"])
    return pre + sim, len(pre)


def main() -> None:
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--mode", required=True, choices=["pre_analysis", "production"],
                   help="pre_analysis: 14-day rolling on first 30 days (16 folds, "
                        "Ridge-vs-LightGBM comparison). production: 30-day rolling "
                        "on the OOS sim days (64 folds, the alpha_ML production model).")
    p.add_argument("--horizon", type=int, default=10, choices=LABEL_HORIZONS_S,
                   help="label horizon in seconds (default 10)")
    p.add_argument("--target", default="drift_mid", choices=TARGET_FAMILIES,
                   help="target family; column read is <target>_<horizon>")
    p.add_argument("--model", default="lgbm", choices=["lgbm", "ar"],
                   help="lgbm: the alpha_ML production model. ar: the "
                        "price-only AR(1) benchmark of strategy 3, same "
                        "folds/purge/target, one lagged-drift feature, OLS "
                        "through the origin, no hyperparameter search")
    p.add_argument("--retune-every-days", type=int, default=DEFAULT_RETUNE_EVERY_DAYS,
                   help="production mode only: how often to re-run the inner CV "
                        "(1 = full inner CV every day, 7 = weekly; default 7). "
                        "Ignored in pre_analysis mode (always retunes per fold).")
    p.add_argument("--max-folds", type=int, default=None,
                   help="cap the number of outer folds (debugging)")
    p.add_argument("--out-dir", default=None,
                   help="output root (default <base>/models)")
    p.add_argument("--keep-stale", action="store_true",
                   help="keep rows where the mark price was stale "
                        "(default: drop them)")
    p.add_argument("--n-trials", type=int, default=None,
                   help="Optuna trials per inner-CV study. Default resolves "
                        f"to {DEFAULT_N_TRIALS} (long budget) or "
                        f"{DEFAULT_N_TRIALS_PRE_ANALYSIS_SHORT} (short "
                        "budget, pre-analysis only). MedianPruner kills "
                        f"underperformers after {OPTUNA_N_STARTUP_TRIALS} "
                        "startup trials. Explicit value overrides the "
                        "budget preset.")
    p.add_argument("--max-boost-rounds", type=int, default=None,
                   help="LightGBM round cap. Default resolves to "
                        f"{MAX_BOOST_ROUNDS} (long budget) or "
                        f"{MAX_BOOST_ROUNDS_PRE_ANALYSIS_SHORT} (short "
                        "budget, pre-analysis only). Explicit value "
                        "overrides the budget preset.")
    p.add_argument("--pre-analysis-budget", choices=["long", "short"],
                   default="long",
                   help="Compute budget preset for pre_analysis mode. "
                        f"'long' (default): {MAX_BOOST_ROUNDS}-round cap, "
                        f"{DEFAULT_N_TRIALS} Optuna trials, full tuning "
                        "parity with the Ridge baseline. 'short' is an "
                        "opt-in fast preset for iteration / sanity checks "
                        f"({MAX_BOOST_ROUNDS_PRE_ANALYSIS_SHORT}-round "
                        f"cap, {DEFAULT_N_TRIALS_PRE_ANALYSIS_SHORT} "
                        "Optuna trials, ~5-6x faster). Rejected in "
                        "production mode.")
    p.add_argument("--optuna-seed", type=int, default=None,
                   help="seed for the Optuna TPE sampler; default derives "
                        "from --seed + outer-fold index so each retune fold "
                        "explores from a different starting point")
    p.add_argument("--num-threads", type=int, default=0,
                   help="LightGBM threads (0 = all cores)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--objective", choices=["regression", "huber"],
                   default="regression",
                   help="LightGBM training objective; 'regression' (L2) is "
                        "the default/production loss, 'huber' is the opt-in "
                        "appendix robustness loss")
    p.add_argument("--huber-alpha", type=float, default=None,
                   help="Huber delta (residual scale); required when "
                        "--objective huber. Set to the median RMSE of the "
                        "L2 run at the same horizon")
    args = p.parse_args()
    if args.objective == "huber" and args.huber_alpha is None:
        raise SystemExit("--objective huber requires --huber-alpha "
                         "(set it to the L2-run median RMSE at this horizon)")
    if args.objective == "regression" and args.huber_alpha is not None:
        raise SystemExit("--huber-alpha is only valid with --objective huber")

    if args.retune_every_days < 1:
        raise SystemExit("--retune-every-days must be >= 1")
    if args.model == "ar":
        if args.objective != "regression":
            raise SystemExit("--model ar is least squares; --objective is "
                             "not applicable")
    if args.pre_analysis_budget == "short" and args.mode != "pre_analysis":
        raise SystemExit(
            "--pre-analysis-budget short is only valid with --mode pre_analysis"
        )
    use_short_budget = (
        args.mode == "pre_analysis" and args.pre_analysis_budget == "short"
    )
    if args.n_trials is None:
        args.n_trials = (
            DEFAULT_N_TRIALS_PRE_ANALYSIS_SHORT
            if use_short_budget
            else DEFAULT_N_TRIALS
        )
    if args.max_boost_rounds is None:
        args.max_boost_rounds = (
            MAX_BOOST_ROUNDS_PRE_ANALYSIS_SHORT
            if use_short_budget
            else MAX_BOOST_ROUNDS
        )
    if args.n_trials < 1:
        raise SystemExit("--n-trials must be >= 1")
    if args.max_boost_rounds < EARLY_STOPPING_ROUNDS:
        raise SystemExit(
            f"--max-boost-rounds ({args.max_boost_rounds}) must be >= "
            f"EARLY_STOPPING_ROUNDS ({EARLY_STOPPING_ROUNDS})"
        )

    base = Path(args.base)
    preanalysis_tau: int = _load_preanalysis_tau(base)
    out_root = Path(args.out_dir) if args.out_dir else (base / "models")
    target_col = f"{args.target}_{args.horizon}"
    label_stem = "labels"
    feature_allowlist = FEATURE_COLS
    obj_tag = "" if args.objective == "regression" else f"_{args.objective}"
    out_dir = out_root / f"{args.model}_{args.mode}_{args.target}_h{args.horizon}{obj_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- splits and chronological day list ---
    all_days, n_pre_analysis = load_split_days(base)

    lgb_params = dict(DEFAULT_LGB_PARAMS)
    lgb_params["seed"] = args.seed
    lgb_params["num_threads"] = args.num_threads
    if args.objective == "huber":
        lgb_params["objective"] = "huber"
        lgb_params["alpha"] = args.huber_alpha

    if args.model == "ar":
        feature_cols = [ar_feature_name(args.horizon)]
    else:
        feature_cols = resolve_feature_cols(
            base, all_days[0], feature_cols=feature_allowlist, stem=label_stem
        )
    folds = build_outer_folds(
        all_days,
        mode=args.mode,
        n_pre_analysis=n_pre_analysis,
        max_folds=args.max_folds,
    )

    if args.model == "ar":
        cadence_str = "no inner CV (OLS through the origin, no hyperparameters)"
    elif args.mode == "pre_analysis":
        cadence_str = "per-fold Optuna inner CV (pre-analysis always retunes every fold)"
    else:
        cadence_str = (
            "full Optuna inner CV every fold" if args.retune_every_days == 1
            else f"Optuna inner CV every {args.retune_every_days} folds, "
                 f"reuse hyperparameters + rounds in between"
        )
    _banner = ("LightGBM" if args.model == "lgbm"
               else "AR(1) price-only benchmark")
    print(f"{_banner} walk-forward | mode={args.mode} target={target_col}")
    print(f"   {len(all_days)} funding days (pre_analysis={n_pre_analysis}, "
          f"sim={len(all_days) - n_pre_analysis}), {len(folds)} outer folds")
    print(f"   {len(feature_cols)} features, {cadence_str}")
    print(f"   Optuna: TPE + MedianPruner, {args.n_trials} trials/study, "
          f"search space = {list(OPTUNA_SEARCH_SPACE.keys())}")
    if args.mode == "pre_analysis":
        _budget_tag = (
            "SHORT (opt-in, robustness panel, not headline)"
            if use_short_budget
            else "LONG (canonical thesis specification)"
        )
        print(f"   pre-analysis budget: {_budget_tag}")
    print(f"   LightGBM: max_boost_rounds={args.max_boost_rounds}, "
          f"early_stopping={EARLY_STOPPING_ROUNDS}")
    if args.mode == "pre_analysis":
        _purge_desc = (
            f"fixed {_effective_purge(preanalysis_tau)}s "
            f"(pre-analysis tau={preanalysis_tau}s from purge_window.json, "
            f"floor={max(LABEL_HORIZONS_S)}s)"
        )
    else:
        _purge_desc = (
            f"dynamic, tau recomputed per fold from the 30-day rolling "
            f"training window (median-ACF method, covers both inner-CV and "
            f"boundary seams); fallback={preanalysis_tau}s; "
            f"label-horizon floor={max(LABEL_HORIZONS_S)}s"
        )
    print(f"   purge: {_purge_desc}")
    print(f"   out -> {out_dir}")

    cache = DayCache(
        base, feature_cols, target_col, args.horizon,
        drop_stale=not args.keep_stale, label_stem=label_stem,
        ar_horizon=args.horizon if args.model == "ar" else None,
        require_finite_x=args.model == "ar",
    )

    folds_dir = out_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict] = []
    pred_frames: list[pl.DataFrame] = []
    inner_cv_rows: list[dict] = []
    gain_acc = np.zeros(len(feature_cols), dtype=np.float64)
    split_acc = np.zeros(len(feature_cols), dtype=np.float64)
    n_scored = 0

    current_rounds: int | None = None
    current_inner_iters: list[int] = []
    current_calibrated_at_fold: int | None = None
    current_lgb_params: dict | None = None

    current_purge_sec: int = preanalysis_tau

    wall0 = time.perf_counter()
    for fold in folds:
        f = fold["fold"]
        is_retune_fold = args.model == "lgbm" and (
            args.mode == "pre_analysis"
            or current_rounds is None
            or (f % args.retune_every_days == 0)
        )

        # --- Determine purge window for this fold ---
        if args.mode == "pre_analysis":
            # Pre-analysis: fixed tau from purge_window.json throughout.
            fold_purge_sec = preanalysis_tau
        else:
            tau_30d = compute_tau_for_days(
                fold["train_days"], base=base, n_workers=4,
            )
            if tau_30d is not None:
                current_purge_sec = tau_30d
            fold_purge_sec = current_purge_sec
        fold_inner_purge = fold_purge_sec
        fold_boundary_purge = fold_purge_sec

        if is_retune_fold:
            t_cv = time.perf_counter()
            study_seed = (
                args.optuna_seed if args.optuna_seed is not None
                else args.seed + f
            )
            (
                current_lgb_params,
                current_rounds,
                current_inner_iters,
                study_info,
            ) = tune_via_inner_cv(
                fold["train_days"], cache, feature_cols, lgb_params,
                purge_sec=fold_inner_purge,
                n_trials=args.n_trials,
                optuna_seed=study_seed,
                max_boost_rounds=args.max_boost_rounds,
            )
            current_calibrated_at_fold = f
            cv_seconds = time.perf_counter() - t_cv
            tuned = study_info.get("best_params", {})
            inner_cv_rows.append({
                "fold": f,
                "test_day": fold["test_day"],
                "inner_best_iters": ",".join(str(x) for x in current_inner_iters),
                "n_inner_folds": len(current_inner_iters),
                "calibrated_rounds": current_rounds,
                "calibration_rule": "max_inner_best_iter",
                "inner_purge_sec": fold_inner_purge,
                "cv_seconds": cv_seconds,
                "n_optuna_trials": study_info.get("n_trials", 0),
                "n_optuna_complete": study_info.get("n_complete", 0),
                "n_optuna_pruned": study_info.get("n_pruned", 0),
                "optuna_seed": study_seed,
                "best_val_rmse": study_info.get("best_value"),
                "tuned_num_leaves": tuned.get("num_leaves"),
                "tuned_learning_rate": tuned.get("learning_rate"),
                "tuned_min_data_in_leaf": tuned.get("min_data_in_leaf"),
                "tuned_feature_fraction": tuned.get("feature_fraction"),
                "tuned_bagging_fraction": tuned.get("bagging_fraction"),
                "tuned_lambda_l2": tuned.get("lambda_l2"),
            })

        if args.model == "ar":
            res = train_fold_ar(
                fold, cache, feature_cols,
                boundary_purge_sec=fold_boundary_purge,
            )
        else:
            res = train_fold(
                fold, cache, feature_cols,
                current_lgb_params or lgb_params,
                current_rounds or args.max_boost_rounds,
                boundary_purge_sec=fold_boundary_purge,
            )
        if res is None:
            continue
        n_scored += 1
        first_test = fold["test_day"]
        if args.model == "ar":
            model_file = f"fold_{f:02d}_{first_test}.json"
            (out_dir / model_file).write_text(json.dumps({
                "fold": f,
                "test_day": first_test,
                "feature_cols": feature_cols,
                "beta": [float(b) for b in res["beta"]],
                "intercept": 0.0,
                "r2_train": res["r2_train"],
                "n_train": res["n_train"],
            }, indent=2))
        else:
            model_file = f"fold_{f:02d}_{first_test}.txt"
            res["booster"].save_model(str(out_dir / model_file))

        m, lat = res["metrics"], res["latency"]
        metrics_rows.append({
            "fold": f,
            "test_day": first_test,
            "n_train": res["n_train"],
            "n_test": res["n_test"],
            "calibrated_rounds": res["calibrated_rounds"],
            "rounds_calibrated_at_fold": current_calibrated_at_fold,
            "is_retune_fold": bool(is_retune_fold),
            "boundary_purge_sec": fold_boundary_purge,
            "inner_purge_sec": fold_inner_purge,
            "rmse": m["rmse"],
            "r2": m["r2"],
            "dir_acc": m["dir_acc"],
            "y_true_mean": m["y_true_mean"],
            "y_true_std": m["y_true_std"],
            "y_pred_mean": m["y_pred_mean"],
            "y_pred_std": m["y_pred_std"],
            "predict_batch_ns_per_row": lat["batch_ns_per_row"],
            "predict_single_p50_ns": lat["single_p50_ns"],
            "predict_single_p99_ns": lat["single_p99_ns"],
            "fit_seconds": res["fit_seconds"],
            "model_file": model_file,
            **({"beta": float(res["beta"][0]), "r2_train": res["r2_train"]}
               if args.model == "ar" else {}),
        })

        pred_frames.append(pl.DataFrame({
            "ts_ms": res["pred_ts_ms"],
            "fold": np.full(res["n_test"], f, dtype=np.int32),
            "test_day": np.full(res["n_test"], first_test),
            "y_true": res["y_true"],
            "y_pred": res["y_pred"],
        }))
        _ckpt = folds_dir / f"preds_fold_{f:02d}_{first_test}.parquet"
        _tmp = _ckpt.with_suffix(".parquet.tmp")
        pred_frames[-1].write_parquet(_tmp, compression="zstd")
        _tmp.rename(_ckpt)
        (folds_dir / f"metrics_fold_{f:02d}.json").write_text(
            json.dumps(metrics_rows[-1], indent=2)
        )

        gain_acc += res["gain_importance"]
        split_acc += res["split_importance"]

        if is_retune_fold and inner_cv_rows:
            last = inner_cv_rows[-1]
            retune_tag = (
                f" [retune nl={last['tuned_num_leaves']} "
                f"lr={last['tuned_learning_rate']:.3g} "
                f"l2={last['tuned_lambda_l2']:.3g} "
                f"trials={last['n_optuna_complete']}/{last['n_optuna_trials']}]"
            )
        else:
            retune_tag = ""
        r2 = m["r2"]
        da = m["dir_acc"]
        _fit_tag = (f"beta={res['beta'][0]:+.5f}" if args.model == "ar"
                    else f"rounds={res['calibrated_rounds']}")
        print(
            f"   fold {f:02d} [{first_test}] R2={r2:+.5f} dir={da:.4f} "
            f"rmse={m['rmse']:.3e} {_fit_tag} "
            f"fit={res['fit_seconds']:.1f}s{retune_tag}"
        )

    if n_scored == 0:
        raise SystemExit("no fold produced a model, check data availability")

    # --- write metrics CSV ---
    metrics_df = pl.DataFrame(metrics_rows)
    metrics_df.write_csv(out_dir / f"metrics_h{args.horizon}.csv")

    # --- write concatenated out-of-sample predictions ---
    oos = pl.concat(pred_frames, how="vertical").sort("ts_ms")
    oos.write_parquet(
        out_dir / f"oos_predictions_h{args.horizon}.parquet", compression="zstd"
    )

    # --- write inner-CV diagnostics (LightGBM only; the AR fit has none) ---
    if inner_cv_rows:
        pl.DataFrame(inner_cv_rows).write_csv(
            out_dir / f"inner_cv_h{args.horizon}.csv"
        )

    # --- write fold-averaged feature importance ---
    imp_df = pl.DataFrame({
        "feature": feature_cols,
        "gain_mean": gain_acc / n_scored,
        "split_mean": split_acc / n_scored,
    }).sort("gain_mean", descending=True)
    imp_df.write_csv(out_dir / f"feature_importance_h{args.horizon}.csv")

    # --- write run config ---
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lightgbm_version": lgb.__version__,
        "optuna_version": optuna.__version__,
        "args": vars(args),
        "model": args.model,
        "target_col": target_col,
        "mode": args.mode,
        "pre_analysis_window": PRE_ANALYSIS_WINDOW,
        "production_window": PRODUCTION_WINDOW,
        "best_iter_rule": "max_inner_best_iter",
        "retune_every_days": args.retune_every_days,
        "n_folds": n_scored,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "max_boost_rounds": args.max_boost_rounds,
        "pre_analysis_budget": (
            args.pre_analysis_budget if args.mode == "pre_analysis" else None
        ),
        "lgb_params_base": lgb_params,
        "optuna": {
            "n_trials": args.n_trials,
            "n_startup_trials": OPTUNA_N_STARTUP_TRIALS,
            "sampler": "TPESampler",
            "pruner": "MedianPruner",
            "search_space": OPTUNA_SEARCH_SPACE,
        },
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "purge_strategy": (
            "fixed_preanalysis" if args.mode == "pre_analysis" else "dynamic"
        ),
        "preanalysis_tau_sec": preanalysis_tau,
        "label_horizon_floor_sec": max(LABEL_HORIZONS_S),
    }
    (out_dir / f"run_config_h{args.horizon}.json").write_text(
        json.dumps(config, indent=2, default=str)
    )

    # --- pooled out-of-sample summary ---
    pooled = regression_metrics(
        oos["y_true"].to_numpy(), oos["y_pred"].to_numpy()
    )
    r2_vals = [r["r2"] for r in metrics_rows if r["r2"] is not None]
    da_vals = [r["dir_acc"] for r in metrics_rows if r["dir_acc"] is not None]
    wall = time.perf_counter() - wall0

    print(f"\n{n_scored} folds trained in {wall:.0f}s")
    print(f"   pooled OOS: R2={pooled['r2']:+.5f} "
          f"dir_acc={pooled['dir_acc']:.4f} rmse={pooled['rmse']:.3e} "
          f"n={pooled['n']}")
    if r2_vals:
        print(f"   per-fold R2: mean={np.mean(r2_vals):+.5f} "
              f"median={np.median(r2_vals):+.5f} "
              f"min={np.min(r2_vals):+.5f} max={np.max(r2_vals):+.5f}")
    if da_vals:
        print(f"   per-fold dir_acc: mean={np.mean(da_vals):.4f}")
    if args.model == "ar":
        betas = np.array([r["beta"] for r in metrics_rows], dtype=np.float64)
        print(f"   beta: mean={betas.mean():+.5f} median={np.median(betas):+.5f} "
              f"min={betas.min():+.5f} max={betas.max():+.5f} "
              f"sign_pos={int((betas > 0).sum())}/{betas.size}")
    top5 = imp_df.head(5)["feature"].to_list()
    print(f"   top-5 features by gain: {top5}")
    print(f"   artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
