"""Stage 08: SHAP attributions for the LightGBM model."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

import lightgbm as lgb
import shap

from ml_train import (
    DEFAULT_BASE,
    DIFF_FEATURES,
    LABEL_HORIZONS_S,
    TARGET_FAMILIES,
    DayCache,
    load_split_days,
    resolve_feature_cols,
)
from pipeline_utils import funding_day_paths, funding_settlement_times


BOUNDARY_BUCKET_EDGES_S: tuple[int, ...] = (0, 60, 300, 900, 1800, 3600)
BOUNDARY_BUCKET_LABELS: tuple[str, ...] = (
    "0-1m", "1-5m", "5-15m", "15-30m", "30-60m", ">60m",
)

BOUNDARY_LAYER_CUTOFF_S = 3600

DEFAULT_BOUNDARY_SAMPLE_PER_DAY = 3000
DEFAULT_BULK_SAMPLE_PER_DAY = 2000

# Top-K for the interaction heatmap and Ridge-contrast table.
DEFAULT_TOP_K = 15

DEFAULT_INTERACTION_SAMPLE_ROWS = 5000

MIN_BUCKET_INTERACTION_ROWS = 200

BOUNDARY_FOCUS_FEATURES: tuple[str, ...] = (
    # funding-timing side (expected risers)
    "seconds_to_funding",
    "basis_bps",
    "mark_stale_age_ms",
    # microstructure side (expected flat)
    "ofi_l1", "ofi_l2_5", "ofi_l6_10", "ofi_l11_20",
    "imbalance_l1", "imbalance_l2_5",
    "vpin_micro", "vpin_base", "vpin_macro",
    "realized_vol_60s",
)

U_BIN_EDGES: tuple[int, ...] = (
    0, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400, 28800,
)

DEFAULT_INTERACTION_FOLD_STRIDE = 8

RANDOM_SEED = 1337


def production_dir(models_dir: Path, target: str, horizon: int) -> Path:
    return models_dir / f"lgbm_production_{target}_h{horizon}"


def ridge_dir(models_dir: Path, target: str, horizon: int) -> Path:
    return models_dir / f"ridge_preanalysis_{target}_h{horizon}"


def load_metrics(prod_dir: Path, horizon: int) -> pl.DataFrame:
    p = prod_dir / f"metrics_h{horizon}.csv"
    if not p.exists():
        raise SystemExit(
            f"production metrics CSV not found at {p}. Run "
            f"`python ml_train.py --mode production --target ... --horizon {horizon}` first."
        )
    return pl.read_csv(p)


def pick_representative_fold(metrics: pl.DataFrame) -> tuple[int, str]:
    """Median per-fold R^2 fold. Ties broken by lower fold index."""
    valid = metrics.filter(pl.col("r2").is_not_null()).sort(["r2", "fold"])
    if valid.is_empty():
        raise SystemExit("no fold has a valid R^2; cannot pick representative")
    median_r2 = float(valid["r2"].median())
    # closest fold to median R^2
    ranked = valid.with_columns(
        (pl.col("r2") - median_r2).abs().alias("r2_gap")
    ).sort(["r2_gap", "fold"])
    row = ranked.row(0, named=True)
    return int(row["fold"]), str(row["test_day"])


def load_fold_model(prod_dir: Path, fold: int, test_day: str) -> lgb.Booster:
    """Booster file is named fold_FF_<test_day>.txt by ml_train.train_fold."""
    p = prod_dir / f"fold_{fold:02d}_{test_day}.txt"
    if not p.exists():
        raise SystemExit(f"fold model not found: {p}")
    return lgb.Booster(model_file=str(p))


def seconds_to_next_settlement(
    base: Path, fday: str, ts_ms: np.ndarray,
) -> np.ndarray:
    """Backward time u = T_fund - t in seconds for each row timestamp, from the
    settlement CALENDAR (not the feature column). Same convention as
    run_simulation: the first settlement at/after each timestamp is used.
    Ported from the retired analysis_shap_boundary.py; selectable via
    --u-source calendar as a robustness cross-check on the seconds_to_funding
    feature column."""
    cands = sorted({s for d, _ in funding_day_paths(base, fday)
                    for s in funding_settlement_times(d)})
    if not cands:
        return np.full(ts_ms.shape, np.nan)
    settles = np.asarray(cands, dtype=np.int64)
    idx = np.searchsorted(settles, ts_ms, side="left")
    idx = np.clip(idx, 0, len(settles) - 1)
    u = (settles[idx] - ts_ms) / 1000.0
    return np.where(u >= 0, u, np.nan)


def assign_bucket(seconds_to_funding: np.ndarray) -> np.ndarray:
    """Map each row to an integer index into BOUNDARY_BUCKET_LABELS."""
    s = np.where(np.isfinite(seconds_to_funding), seconds_to_funding, np.inf)
    idx = np.searchsorted(np.asarray(BOUNDARY_BUCKET_EDGES_S), s, side="right")
    # idx 0 only if sec < 0 which Stage 04 cannot emit; clamp defensively.
    idx = np.clip(idx, 1, len(BOUNDARY_BUCKET_LABELS))
    return idx - 1  # 0-indexed into BOUNDARY_BUCKET_LABELS


def boundary_weighted_sample(
    rng: np.random.Generator,
    seconds_to_funding: np.ndarray,
    n_boundary: int,
    n_bulk: int,
) -> np.ndarray:
    """Return row indices: up to n_boundary rows with sec <= 3600 and up to
    n_bulk rows with sec > 3600 (or NaN), each drawn without replacement."""
    n = seconds_to_funding.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    boundary_mask = (seconds_to_funding <= BOUNDARY_LAYER_CUTOFF_S) & np.isfinite(
        seconds_to_funding
    )
    bulk_mask = ~boundary_mask  # NaN seconds_to_funding falls in bulk

    b_idx = np.flatnonzero(boundary_mask)
    u_idx = np.flatnonzero(bulk_mask)

    if b_idx.size > n_boundary:
        b_idx = rng.choice(b_idx, size=n_boundary, replace=False)
    if u_idx.size > n_bulk:
        u_idx = rng.choice(u_idx, size=n_bulk, replace=False)

    combined = np.concatenate([b_idx, u_idx])
    combined.sort()
    return combined


def compute_day_shap(
    booster: lgb.Booster,
    X: np.ndarray,
) -> np.ndarray:
    """
    TreeSHAP on a LightGBM Booster. Returns (n_rows, n_features).

    `feature_perturbation="tree_path_dependent"` is the default for trees;
    `model_output="raw"` returns SHAP values on the raw (pre-link) scale,
    which matches the booster's regression objective (no link transform).
    """
    explainer = shap.TreeExplainer(
        booster,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        # multi-output guard (regression should return a single ndarray)
        sv = sv[0]
    return np.asarray(sv, dtype=np.float64)


def compute_interactions(
    booster: lgb.Booster,
    X: np.ndarray,
) -> np.ndarray:
    """TreeSHAP interaction values; (n_rows, n_features, n_features),
    symmetric in the last two axes, diagonal = main effect."""
    explainer = shap.TreeExplainer(
        booster,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    iv = explainer.shap_interaction_values(X)
    if isinstance(iv, list):
        iv = iv[0]
    return np.asarray(iv, dtype=np.float64)


def nonadditivity_ratio(mean_abs_iv: np.ndarray) -> float:
    """Off-diagonal share of the mean(|interaction|) matrix: sum_{i!=j} |iv[i,j]|  / ."""
    total = float(np.abs(mean_abs_iv).sum())
    if total <= 0.0:
        return float("nan")
    offdiag = total - float(np.abs(np.diag(mean_abs_iv)).sum())
    return offdiag / total


def regime_interaction_matrices(
    booster: lgb.Booster,
    X: np.ndarray,
    u_s: np.ndarray,
    cap: int,
    rng: np.random.Generator,
) -> dict[str, tuple[np.ndarray, int]]:
    """Per-regime mean(|interaction|) (F,F) matrices for one fold's rows, split
    boundary (u <= cutoff) vs bulk on a BALANCED capped subsample so the two
    are comparable. Returns {regime: (mean_abs_iv, n_rows)} including a
    'pooled' (n-weighted) entry. Regimes with < MIN_BUCKET_INTERACTION_ROWS
    rows are omitted."""
    sel = (u_s <= BOUNDARY_LAYER_CUTOFF_S) & np.isfinite(u_s)
    out: dict[str, tuple[np.ndarray, int]] = {}
    for name, mask in (("boundary", sel), ("bulk", ~sel)):
        ridx = np.flatnonzero(mask)
        if ridx.size < MIN_BUCKET_INTERACTION_ROWS:
            continue
        Xb = X[ridx]
        if Xb.shape[0] > cap:
            Xb = Xb[rng.choice(Xb.shape[0], size=cap, replace=False)]
        out[name] = (np.abs(compute_interactions(booster, Xb)).mean(axis=0),
                     int(Xb.shape[0]))
    if out:
        ntot = sum(n for _, n in out.values())
        pooled = sum(n * m for m, n in out.values()) / ntot
        out["pooled"] = (pooled, ntot)
    return out


def matrix_summary(name: str, m_abs: np.ndarray, n_rows: int) -> dict:
    """Scalar non-additivity summary for one regime's interaction matrix."""
    total = float(m_abs.sum())
    offdiag = total - float(np.abs(np.diag(m_abs)).sum())
    return {
        "bucket": name,
        "n_rows": int(n_rows),
        "non_additivity_ratio": nonadditivity_ratio(m_abs),
        "total_abs_interaction": total,
        "offdiag_abs_interaction": offdiag,
    }


def aggregate_global(
    shap_values: np.ndarray,
    feature_cols: list[str],
    weights: np.ndarray | None = None,
) -> pl.DataFrame:
    abs_shap = np.abs(shap_values)
    if weights is None:
        mean_abs = abs_shap.mean(axis=0)
        std_abs = abs_shap.std(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        wsum = float(w.sum())
        if wsum <= 0.0:
            raise ValueError("aggregate_global: weights sum to zero")
        mean_abs = (w[:, None] * abs_shap).sum(axis=0) / wsum
        var = (w[:, None] * (abs_shap - mean_abs) ** 2).sum(axis=0) / wsum
        std_abs = np.sqrt(np.maximum(var, 0.0))
    return pl.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs,
        "std_abs_shap": std_abs,
        "n_rows": np.full(len(feature_cols), shap_values.shape[0], dtype=np.int64),
    }).sort("mean_abs_shap", descending=True)


def aggregate_boundary_layer(
    shap_values: np.ndarray,
    bucket_idx: np.ndarray,
    feature_cols: list[str],
) -> pl.DataFrame:
    """Per (bucket, feature): mean(|SHAP|) AND mean(SHAP) (signed)."""
    abs_shap = np.abs(shap_values)
    rows: list[dict] = []
    for b, label in enumerate(BOUNDARY_BUCKET_LABELS):
        mask = bucket_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        per_feat_abs = abs_shap[mask].mean(axis=0)
        per_feat_signed = shap_values[mask].mean(axis=0)
        for j, feat in enumerate(feature_cols):
            rows.append({
                "bucket": label,
                "bucket_order": b,
                "feature": feat,
                "mean_abs_shap": float(per_feat_abs[j]),
                "mean_signed_shap": float(per_feat_signed[j]),
                "n_rows": n_b,
            })
    return pl.DataFrame(rows)


def build_profile(
    shap_values: np.ndarray,
    u_s: np.ndarray,
    feature_cols: list[str],
) -> pl.DataFrame:
    """Mean(|SHAP|) and mean(SHAP) per (fine u-bin, feature), long format."""
    edges = np.asarray(U_BIN_EDGES, dtype=float)
    abs_shap = np.abs(shap_values)
    u = np.where(np.isfinite(u_s), u_s, np.inf)
    ub = np.digitize(u, edges) - 1  # bin index in [0, len(edges)-1]
    rows: list[dict] = []
    for k in range(len(edges) - 1):
        mask = ub == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        m_abs = abs_shap[mask].mean(axis=0)
        m_signed = shap_values[mask].mean(axis=0)
        for j, feat in enumerate(feature_cols):
            rows.append({
                "u_lo_s": float(edges[k]),
                "u_hi_s": float(edges[k + 1]),
                "n_rows": n_k,
                "feature": feat,
                "mean_abs_shap": float(m_abs[j]),
                "mean_signed_shap": float(m_signed[j]),
            })
    return pl.DataFrame(rows)


def load_ridge_coefficients(
    r_dir: Path, horizon: int,
) -> tuple[pl.DataFrame, list[str]]:
    """Returns (coef summary, list of features Ridge dropped vs LightGBM).

    The Ridge baseline drops `signed_vol_*` (see ml_baseline_ridge.py
    RIDGE_DROP_FEATURES) because they are exact linear combinations of
    `buy_vol_* - sell_vol_*`. Those columns appear in the LightGBM feature
    list but not in the Ridge coefficient table.
    """
    csv = r_dir / f"coefficient_summary_h{horizon}.csv"
    cfg = r_dir / f"run_config_h{horizon}.json"
    if not csv.exists():
        raise SystemExit(
            f"Ridge coefficient summary not found at {csv}. Run "
            f"`python ml_baseline_ridge.py --target ... --horizon {horizon}` first."
        )
    coefs = pl.read_csv(csv)
    dropped: list[str] = []
    if cfg.exists():
        cfg_obj = json.loads(cfg.read_text())
        dropped = list(cfg_obj.get("ridge_dropped_features", []))
    return coefs, dropped


def build_ridge_contrast(
    global_shap: pl.DataFrame,
    ridge_coefs: pl.DataFrame,
    ridge_dropped: list[str],
    top_k: int,
) -> pl.DataFrame:
    """Top-k features by LightGBM mean(|SHAP|), with the matching Ridge
    standardised coefficient (NULL if Ridge dropped the column)."""
    top = global_shap.head(top_k).with_columns(
        pl.col("mean_abs_shap").rank(method="ordinal", descending=True)
            .cast(pl.Int64).alias("lgbm_rank")
    )
    ridge_keyed = ridge_coefs.select([
        pl.col("feature"),
        pl.col("coef_mean").alias("ridge_coef_mean"),
        pl.col("abs_coef_mean").alias("ridge_abs_coef"),
    ]).with_columns(
        pl.col("ridge_abs_coef").rank(method="ordinal", descending=True)
            .cast(pl.Int64).alias("ridge_rank")
    )
    contrast = top.join(ridge_keyed, on="feature", how="left").with_columns(
        pl.col("feature").is_in(ridge_dropped).alias("ridge_dropped")
    )
    return contrast.select([
        "feature",
        "mean_abs_shap",
        "lgbm_rank",
        "ridge_coef_mean",
        "ridge_abs_coef",
        "ridge_rank",
        "ridge_dropped",
    ])


def write_latex_table(contrast: pl.DataFrame, out_path: Path, horizon: int) -> None:
    """Top-15 features, side-by-side LightGBM mean(|SHAP|) and Ridge
    standardised coefficient. NaN-safe rendering for Ridge-dropped rows."""
    rows = contrast.sort("mean_abs_shap", descending=True).to_dicts()
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Top-15 LightGBM features by mean$|\text{SHAP}|$ on the 64-day "
        r"OOS production walk-forward, with the matching Ridge standardised "
        r"coefficient from the pre-analysis baseline. The two columns are not "
        r"comparable in magnitude (different units); their rank correlation and "
        r"sign agreement are the comparison axes.}",
        rf"\label{{tab:shap_ridge_contrast_h{horizon}}}",
        r"\begin{tabular}{l r r r r}",
        r"\toprule",
        r"Feature & LGBM rank & mean$|\text{SHAP}|$ & Ridge rank & Ridge $\hat{\beta}$ \\",
        r"\midrule",
    ]
    for row in rows:
        feat = str(row["feature"]).replace("_", r"\_")
        lgb_r = row["lgbm_rank"]
        lgb_v = row["mean_abs_shap"]
        rdg_r = row["ridge_rank"]
        rdg_v = row["ridge_coef_mean"]
        rdg_dropped = bool(row["ridge_dropped"])
        if rdg_dropped or rdg_v is None:
            rdg_cell = r"\multicolumn{2}{c}{(Ridge: dropped)}"
            lines.append(rf"{feat} & {lgb_r} & {lgb_v:.3e} & {rdg_cell} \\")
        else:
            lines.append(
                rf"{feat} & {lgb_r} & {lgb_v:.3e} & {int(rdg_r)} & {rdg_v:+.3e} \\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines))


def shap_one_fold(
    fold_row: dict,
    prod_dir: Path,
    cache: DayCache,
    sec_to_fund_idx: int,
    boundary_sample: int,
    bulk_sample: int,
    rng: np.random.Generator,
    base: Path,
    u_source: str,
) -> dict | None:
    """Load the fold's booster, subsample the test day boundary-weighted,
    compute SHAP. Returns a dict with the per-row attribution shard, or None
    if the day has no usable rows. `u_source` selects the backward-time axis
    used for ALL bucketing / weighting / interaction-splitting: "feature" uses
    the seconds_to_funding feature column (what the model conditioned on, the
    default); "calendar" recomputes u from the settlement calendar."""
    fold = int(fold_row["fold"])
    test_day = str(fold_row["test_day"])
    X_full, _, ts_full = cache.get(test_day)
    if X_full.shape[0] == 0:
        return None
    sec_feat_full = X_full[:, sec_to_fund_idx].astype(np.float64)
    if u_source == "calendar":
        u_full = seconds_to_next_settlement(base, test_day, ts_full)
    else:
        u_full = sec_feat_full
    idx = boundary_weighted_sample(rng, u_full, boundary_sample, bulk_sample)
    if idx.size == 0:
        return None
    Xs = X_full[idx]
    ts = ts_full[idx]
    u = u_full[idx]
    sec_feat = sec_feat_full[idx]
    booster = load_fold_model(prod_dir, fold, test_day)
    shap_values = compute_day_shap(booster, Xs)
    bucket = assign_bucket(u)
    full_b = (u_full <= BOUNDARY_LAYER_CUTOFF_S) & np.isfinite(u_full)
    samp_b = (u <= BOUNDARY_LAYER_CUTOFF_S) & np.isfinite(u)
    n_b_full, n_u_full = int(full_b.sum()), int((~full_b).sum())
    n_b_samp, n_u_samp = int(samp_b.sum()), int((~samp_b).sum())
    w_b = (n_b_full / n_b_samp) if n_b_samp else 0.0
    w_u = (n_u_full / n_u_samp) if n_u_samp else 0.0
    weight = np.where(samp_b, w_b, w_u).astype(np.float64)
    return {
        "fold": fold,
        "test_day": test_day,
        "ts_ms": ts,
        "u": u,                          # chosen backward-time axis (seconds)
        "seconds_to_funding": sec_feat,  # raw feature value, for reference
        "bucket_idx": bucket,
        "weight": weight,
        "X": Xs,
        "shap_values": shap_values,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--horizon", type=int, default=10, choices=LABEL_HORIZONS_S,
                   help="label horizon in seconds (default 10)")
    p.add_argument("--target", default="drift_mid", choices=TARGET_FAMILIES,
                   help="target family; must match the ml_train.py production run "
                        "(default drift_mid, the project-standard target)")
    p.add_argument("--models-dir", default=None,
                   help="root containing lgbm_production_* and ridge_preanalysis_* "
                        "(default <base>/models)")
    p.add_argument("--out-dir", default=None,
                   help="output dir (default <models-dir>/shap_<target>_h<h>)")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                   help=f"top-k features for the interaction heatmap, beeswarm "
                        f"and Ridge contrast (default {DEFAULT_TOP_K})")
    p.add_argument("--boundary-sample", type=int,
                   default=DEFAULT_BOUNDARY_SAMPLE_PER_DAY,
                   help="rows/day subsampled inside the 60-min boundary layer "
                        f"(default {DEFAULT_BOUNDARY_SAMPLE_PER_DAY})")
    p.add_argument("--bulk-sample", type=int,
                   default=DEFAULT_BULK_SAMPLE_PER_DAY,
                   help="rows/day subsampled outside the boundary layer "
                        f"(default {DEFAULT_BULK_SAMPLE_PER_DAY})")
    p.add_argument("--interaction-rows", type=int,
                   default=DEFAULT_INTERACTION_SAMPLE_ROWS,
                   help="per-regime row cap for the representative-fold "
                        "interaction matrices (bulk and boundary each capped at "
                        f"this; default {DEFAULT_INTERACTION_SAMPLE_ROWS})")
    p.add_argument("--max-folds", type=int, default=None,
                   help="cap the number of OOS folds processed (debugging)")
    p.add_argument("--keep-stale", action="store_true",
                   help="keep rows where the mark price was stale "
                        "(default: drop them, matches ml_train.py default)")
    p.add_argument("--no-ridge", action="store_true",
                   help="skip the Ridge interpretability contrast (use when "
                        "ml_baseline_ridge.py has not been run yet)")
    p.add_argument("--u-source", choices=["feature", "calendar"], default="feature",
                   help="backward-time axis for bucketing/weighting/interaction "
                        "split: 'feature' = seconds_to_funding column (what the "
                        "model saw, default); 'calendar' = recomputed from the "
                        "settlement calendar (robustness cross-check)")
    p.add_argument("--interaction-fold-stride", type=int,
                   default=DEFAULT_INTERACTION_FOLD_STRIDE,
                   help="compute the non-additivity ratio on every k-th fold to "
                        "get a per-fold confidence interval (the representative "
                        "fold is always included); "
                        f"default {DEFAULT_INTERACTION_FOLD_STRIDE}")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = p.parse_args()

    base = Path(args.base)
    models_dir = Path(args.models_dir) if args.models_dir else (base / "models")
    prod_dir = production_dir(models_dir, args.target, args.horizon)
    if not prod_dir.exists():
        raise SystemExit(
            f"production model directory not found at {prod_dir}. Run "
            f"`python ml_train.py --mode production --target {args.target} "
            f"--horizon {args.horizon}` first."
        )

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else models_dir / f"shap_{args.target}_h{args.horizon}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    target_col = f"{args.target}_{args.horizon}"
    rng = np.random.default_rng(args.seed)

    # --- production run inventory ---
    metrics = load_metrics(prod_dir, args.horizon)
    rep_fold, rep_test_day = pick_representative_fold(metrics)

    # --- features and data loader ---
    all_days, _ = load_split_days(base)
    feature_cols = resolve_feature_cols(base, all_days[0])
    if "seconds_to_funding" not in feature_cols:
        raise SystemExit(
            "seconds_to_funding absent from the feature set; cannot bucket "
            "rows by minutes-to-settlement."
        )
    sec_to_fund_idx = feature_cols.index("seconds_to_funding")

    cache = DayCache(
        base, feature_cols, target_col, args.horizon,
        drop_stale=not args.keep_stale,
    )

    # --- fold list ---
    fold_rows = metrics.sort("fold").to_dicts()
    if args.max_folds is not None:
        fold_rows = fold_rows[: args.max_folds]

    print(f"SHAP -- target={target_col}, {len(fold_rows)} OOS folds")
    print(f"   models       <- {prod_dir}")
    print(f"   representative fold for interactions: "
          f"fold {rep_fold:02d} [{rep_test_day}]")
    print(f"   subsample per day: boundary={args.boundary_sample} "
          f"bulk={args.bulk_sample} (boundary cutoff "
          f"{BOUNDARY_LAYER_CUTOFF_S}s)")
    print(f"   features: {len(feature_cols)} (top-{args.top_k} for figures/tables)")
    print(f"   out -> {out_dir}")

    # --- per-fold SHAP ---
    shards: list[dict] = []
    t0 = time.perf_counter()
    for k, fold_row in enumerate(fold_rows):
        shard = shap_one_fold(
            fold_row, prod_dir, cache,
            sec_to_fund_idx,
            args.boundary_sample, args.bulk_sample, rng,
            base, args.u_source,
        )
        if shard is None:
            print(f"   fold {int(fold_row['fold']):02d} [{fold_row['test_day']}] "
                  f" skipped (no rows)")
            continue
        shards.append(shard)
        n = shard["shap_values"].shape[0]
        if (k + 1) % 8 == 0 or k == len(fold_rows) - 1:
            elapsed = time.perf_counter() - t0
            print(f"   fold {int(fold_row['fold']):02d} [{fold_row['test_day']}] "
                  f"shap n={n}  total_elapsed={elapsed:.0f}s")
    if not shards:
        raise SystemExit("no fold produced SHAP rows -- check data availability")

    # --- aggregate ---
    shap_all = np.vstack([s["shap_values"] for s in shards])
    ts_all = np.concatenate([s["ts_ms"] for s in shards])
    sec_all = np.concatenate([s["seconds_to_funding"] for s in shards])
    u_all = np.concatenate([s["u"] for s in shards])
    bucket_all = np.concatenate([s["bucket_idx"] for s in shards])
    weight_all = np.concatenate([s["weight"] for s in shards])
    fold_all = np.concatenate([
        np.full(s["shap_values"].shape[0], s["fold"], dtype=np.int32)
        for s in shards
    ])
    day_all = np.concatenate([
        np.full(s["shap_values"].shape[0], s["test_day"]) for s in shards
    ])

    global_shap = aggregate_global(shap_all, feature_cols, weights=weight_all)
    boundary = aggregate_boundary_layer(shap_all, bucket_all, feature_cols)

    global_shap.write_csv(out_dir / f"shap_global_h{args.horizon}.csv")
    boundary.write_csv(out_dir / f"shap_boundary_layer_h{args.horizon}.csv")

    shap_cols = {
        f"shap__{feat}": shap_all[:, j] for j, feat in enumerate(feature_cols)
    }
    pl.DataFrame({
        "ts_ms": ts_all,
        "fold": fold_all,
        "test_day": day_all,
        "u_s": u_all,                       # chosen backward-time axis (seconds)
        "seconds_to_funding": sec_all,      # raw feature value
        "bucket": np.asarray(BOUNDARY_BUCKET_LABELS)[bucket_all],
        "weight": weight_all,
        **shap_cols,
    }).write_parquet(
        out_dir / f"shap_values_h{args.horizon}.parquet", compression="zstd"
    )

    # --- attribution-vs-backward-time profile parquet (figure source) ---
    build_profile(shap_all, u_all, feature_cols).write_parquet(
        out_dir / f"shap_boundary_profile_h{args.horizon}.parquet"
    )

    # --- top-k for figures/tables ---
    top_features = global_shap.head(args.top_k)["feature"].to_list()
    top_idx = np.array([feature_cols.index(f) for f in top_features], dtype=np.int64)

    rep_shard = next((s for s in shards if s["fold"] == rep_fold), None)
    if rep_shard is not None:
        bee = {"bucket": np.asarray(BOUNDARY_BUCKET_LABELS)[rep_shard["bucket_idx"]]}
        for j, feat in zip(top_idx.tolist(), top_features):
            bee[f"val__{feat}"] = rep_shard["X"][:, j]
            bee[f"shap__{feat}"] = rep_shard["shap_values"][:, j]
        pl.DataFrame(bee).write_parquet(
            out_dir / f"shap_beeswarm_h{args.horizon}.parquet", compression="zstd"
        )

    stride = max(1, args.interaction_fold_stride)
    fold_ratio_rows: list[dict] = []
    rep_matrices: dict[str, tuple[np.ndarray, int]] | None = None
    t_iv0 = time.perf_counter()
    for k, s in enumerate(shards):
        if (k % stride != 0) and (s["fold"] != rep_fold):
            continue
        booster_f = load_fold_model(prod_dir, s["fold"], s["test_day"])
        mats = regime_interaction_matrices(
            booster_f, s["X"], s["u"], args.interaction_rows, rng,
        )
        for name, (m_abs, n) in mats.items():
            fold_ratio_rows.append({
                "fold": int(s["fold"]), "test_day": s["test_day"],
                **matrix_summary(name, m_abs, n),
            })
        if s["fold"] == rep_fold:
            rep_matrices = mats
    print(f"   per-fold interactions: {len(fold_ratio_rows)} (fold,regime) rows "
          f"in {time.perf_counter() - t_iv0:.0f}s")

    interaction_buckets = None
    if fold_ratio_rows:
        folds_df = pl.DataFrame(fold_ratio_rows)
        folds_df.write_csv(out_dir / f"shap_interaction_folds_h{args.horizon}.csv")
        # per-regime mean / std / sem of the ratio across folds (the CI source)
        interaction_buckets = folds_df.group_by("bucket").agg(
            pl.col("non_additivity_ratio").mean().alias("mean_non_additivity_ratio"),
            pl.col("non_additivity_ratio").std(ddof=1).alias("std_non_additivity_ratio"),
            pl.len().alias("n_folds"),
            pl.col("n_rows").sum().alias("n_rows_total"),
        ).with_columns(
            (pl.col("std_non_additivity_ratio")
             / pl.col("n_folds").cast(pl.Float64).sqrt())
            .alias("sem_non_additivity_ratio")
        ).sort("bucket")
        interaction_buckets.write_csv(
            out_dir / f"shap_interaction_buckets_h{args.horizon}.csv")
        d = {r["bucket"]: r for r in interaction_buckets.to_dicts()}
        if "bulk" in d and "boundary" in d:
            print(f"   non-additivity ratio (mean over {d['boundary']['n_folds']} "
                  f"folds): bulk={d['bulk']['mean_non_additivity_ratio']:.3f} "
                  f"-> boundary={d['boundary']['mean_non_additivity_ratio']:.3f} "
                  f"(Ridge = 0 by construction)")

    # top-k interaction matrices from the representative fold (heatmap + diff)
    if rep_matrices is not None:
        if "pooled" in rep_matrices:
            pooled_top = rep_matrices["pooled"][0][top_idx[:, None], top_idx[None, :]]
            pl.DataFrame([
                {"feat_i": fi, "feat_j": fj,
                 "mean_abs_interaction": float(pooled_top[i, j])}
                for i, fi in enumerate(top_features)
                for j, fj in enumerate(top_features)
                if i != j  # diagonal is the main effect, not an interaction
            ]).write_csv(out_dir / f"shap_interactions_h{args.horizon}.csv")
            pooled_full = rep_matrices["pooled"][0]
            F = pooled_full.shape[0]
            pairs = sorted(
                ((float(pooled_full[i, j]), i, j)
                 for i in range(F) for j in range(i + 1, F)),
                key=lambda t: t[0], reverse=True,
            )[: args.top_k]
            M_b_full = rep_matrices.get("boundary", (None,))[0]
            M_u_full = rep_matrices.get("bulk", (None,))[0]
            rows = []
            for r, (val, i, j) in enumerate(pairs):
                row = {"rank": r + 1,
                       "feat_i": feature_cols[i], "feat_j": feature_cols[j],
                       "mean_abs_interaction": val}
                if M_b_full is not None:
                    row["boundary"] = float(M_b_full[i, j])
                if M_u_full is not None:
                    row["bulk"] = float(M_u_full[i, j])
                rows.append(row)
            pl.DataFrame(rows).write_csv(
                out_dir / f"shap_top_interactions_h{args.horizon}.csv")
        if "boundary" in rep_matrices and "bulk" in rep_matrices:
            M_b = rep_matrices["boundary"][0][top_idx[:, None], top_idx[None, :]]
            M_u = rep_matrices["bulk"][0][top_idx[:, None], top_idx[None, :]]
            diff = M_b - M_u
            pl.DataFrame([
                {"feat_i": fi, "feat_j": fj,
                 "boundary": float(M_b[i, j]), "bulk": float(M_u[i, j]),
                 "diff_boundary_minus_bulk": float(diff[i, j])}
                for i, fi in enumerate(top_features)
                for j, fj in enumerate(top_features)
                if i != j  # diagonal is the main effect, not an interaction
            ]).write_csv(out_dir / f"shap_interactions_diff_h{args.horizon}.csv")
    else:
        print(f"    representative fold {rep_fold} not in processed shards; "
              f"interaction heatmap/diff skipped")

    # --- Ridge contrast ---
    ridge_contrast = None
    if not args.no_ridge:
        r_dir = ridge_dir(models_dir, args.target, args.horizon)
        if not (r_dir / f"coefficient_summary_h{args.horizon}.csv").exists():
            print(f"    Ridge coefficients not found under {r_dir}; "
                  f"skipping interpretability contrast (rerun with --no-ridge to "
                  f"silence)")
        else:
            ridge_coefs, ridge_dropped = load_ridge_coefficients(r_dir, args.horizon)
            ridge_contrast = build_ridge_contrast(
                global_shap, ridge_coefs, ridge_dropped, args.top_k,
            )
            ridge_contrast.write_csv(
                out_dir / f"shap_ridge_contrast_h{args.horizon}.csv"
            )
            write_latex_table(
                ridge_contrast,
                out_dir / f"shap_summary_h{args.horizon}.tex",
                args.horizon,
            )

    # --- run config ---
    import sklearn
    cfg = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lightgbm_version": lgb.__version__,
        "shap_version": shap.__version__,
        "sklearn_version": sklearn.__version__,
        "args": vars(args),
        "target_col": target_col,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "diff_features": [c for c in DIFF_FEATURES if c in feature_cols],
        "boundary_bucket_edges_s": list(BOUNDARY_BUCKET_EDGES_S),
        "boundary_bucket_labels": list(BOUNDARY_BUCKET_LABELS),
        "boundary_layer_cutoff_s": BOUNDARY_LAYER_CUTOFF_S,
        "boundary_focus_features": [
            f for f in BOUNDARY_FOCUS_FEATURES if f in feature_cols
        ],
        "n_folds_processed": len(shards),
        "n_shap_rows_total": int(shap_all.shape[0]),
        "representative_fold": rep_fold,
        "representative_test_day": rep_test_day,
        "top_features": top_features,
        "ridge_contrast_written": ridge_contrast is not None,
        "u_source": args.u_source,
        "interaction_fold_stride": stride,
        "global_aggregate_weighting": "horvitz_thompson_inverse_stratum_prob",
        "interaction_buckets": (
            interaction_buckets.to_dicts() if interaction_buckets is not None
            else []
        ),
        "shap_on_differenced_features": [
            c for c in DIFF_FEATURES if c in feature_cols
        ],
    }
    (out_dir / f"run_config_h{args.horizon}.json").write_text(
        json.dumps(cfg, indent=2, default=str)
    )

    # --- summary ---
    wall = time.perf_counter() - t0
    print(f"\nSHAP complete in {wall:.0f}s")
    print(f"   {shap_all.shape[0]} attributed rows over {len(shards)} folds")
    print(f"   top-{args.top_k} features by mean(|SHAP|): {top_features}")
    if ridge_contrast is not None:
        rc = ridge_contrast.filter(pl.col("ridge_rank").is_not_null())
        if rc.height >= 3:
            from scipy.stats import spearmanr
            spearman = float(spearmanr(
                rc["mean_abs_shap"].to_numpy(),
                rc["ridge_abs_coef"].to_numpy(),
            ).statistic)
            print(f"   top-{args.top_k} Ridge-vs-LGBM Spearman rank corr = "
                  f"{spearman:+.3f} ({rc.height} non-dropped features)")
    print(f"   artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
