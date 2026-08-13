"""SHAP figures, rendered from the artifacts written by ml_shap.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ml_shap import (
    BOUNDARY_BUCKET_LABELS,
    BOUNDARY_FOCUS_FEATURES,
    DIFF_FEATURES,
)

FIG_DPI = 150
TOP_BULK = 6          # profile panel (a): top features by bulk share
TOP_RISERS = 5        # profile panel (b): top features by boundary/bulk ratio
BULK_SHARE_FLOOR = 0.005   # panel (b): ignore features with negligible attribution
BOUNDARY_LABELS = tuple(b for b in BOUNDARY_BUCKET_LABELS if b != ">60m")


def _save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)
    print(f"   written {out_base.name}.pdf/.png")


def _pivot_matrix(df: pl.DataFrame, value_col: str) -> tuple[list[str], np.ndarray]:
    """Long (feat_i, feat_j, value) -> (ordered feature list, dense matrix).
    Feature order follows first appearance, which ml_shap.py writes in
    descending global-importance order."""
    feats = list(dict.fromkeys(df["feat_i"].to_list()))
    idx = {f: i for i, f in enumerate(feats)}
    M = np.full((len(feats), len(feats)), np.nan)
    for r in df.iter_rows(named=True):
        M[idx[r["feat_i"]], idx[r["feat_j"]]] = r[value_col]
    return feats, M


def fig_global_bar(global_csv: Path, top_k: int, out_base: Path) -> None:
    df = pl.read_csv(global_csv).head(top_k)
    feats = df["feature"].to_list()[::-1]
    vals = df["mean_abs_shap"].to_numpy()[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * top_k + 1))
    ax.barh(feats, vals, color="#1f77b4")
    ax.set_xlabel("mean(|SHAP|)  [target units, HT-reweighted]")
    ax.set_title(f"Top-{top_k} LightGBM features (pooled 64-fold OOS)")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    _save(fig, out_base)


def _boundary_panels(
    boundary_csv: Path, value_col: str, ylabel: str, title: str,
    out_base: Path, diverging: bool,
) -> None:
    df = pl.read_csv(boundary_csv)
    avail = set(df["feature"].unique().to_list())
    present = [f for f in BOUNDARY_FOCUS_FEATURES if f in avail]
    if not present:
        print(f"   skip {out_base.name}: no focus features present")
        return
    n = len(present)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.6 * nrows))
    axes = np.atleast_2d(axes)
    for k, feat in enumerate(present):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        sub = df.filter(pl.col("feature") == feat).sort("bucket_order")
        labels = sub["bucket"].to_list()
        vals = sub[value_col].to_numpy()
        if diverging:
            colors = ["#d62728" if v >= 0 else "#1f77b4" for v in vals]
            ax.axhline(0.0, color="black", lw=0.8)
        else:
            colors = "#d62728"
        ax.bar(labels, vals, color=colors)
        ax.set_title(feat)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        for lab in ax.get_xticklabels():
            lab.set_rotation(35)
            lab.set_horizontalalignment("right")
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")
    diffed = [f for f in present if f in DIFF_FEATURES]
    note = f"  ({', '.join(diffed)} first-differenced)" if diffed else ""
    fig.suptitle(title + note)
    fig.tight_layout()
    _save(fig, out_base)


def fig_profile(profile_parquet: Path, out_base: Path) -> None:
    df = pl.read_parquet(profile_parquet)
    df = df.with_columns(
        (pl.col("u_lo_s") + pl.col("u_hi_s")).alias("u_sum"),
        (pl.col("mean_abs_shap")
         / pl.col("mean_abs_shap").sum().over(["u_lo_s", "u_hi_s"]))
        .alias("share"),
    ).with_columns((pl.col("u_sum") / 2.0 / 60.0).alias("u_min"))
    bulk = (df.filter(pl.col("u_lo_s") >= 7200)
            .group_by("feature").agg(pl.col("share").mean().alias("bulk")))
    df = df.join(bulk, on="feature").with_columns(
        (pl.col("share") / pl.col("bulk")).alias("rel"))
    top_bulk = (bulk.sort("bulk", descending=True)
                .head(TOP_BULK)["feature"].to_list())

    bnd = (df.filter(pl.col("u_hi_s") <= 1800.0)
           .group_by("feature")
           .agg(((pl.col("share") * pl.col("n_rows")).sum()
                 / pl.col("n_rows").sum()).alias("bnd")))
    ranked = (bulk.join(bnd, on="feature")
              .filter(pl.col("bulk") >= BULK_SHARE_FLOOR)
              .with_columns((pl.col("bnd") / pl.col("bulk")).alias("rel_bnd"))
              .sort("rel_bnd", descending=True))
    risers = ranked.head(TOP_RISERS)["feature"].to_list()

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 7, "axes.titlesize": 9.5})
    fig, axes = plt.subplots(1, 2, figsize=(5.7, 2.7))
    for f in top_bulk:
        d = df.filter(pl.col("feature") == f).sort("u_min")
        axes[0].plot(d["u_min"], 100 * d["share"], marker="o", ms=2.5, label=f)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"backward time to settlement $u$ (min)")
    axes[0].set_ylabel("attribution share (%)")
    axes[0].set_title("(a) bulk-dominant features")
    axes[0].invert_xaxis()
    axes[0].legend(frameon=True, framealpha=1.0, edgecolor="0.85",
                   fancybox=False, loc="center", bbox_to_anchor=(0.5, 0.33),
                   ncol=2, handlelength=1.2, columnspacing=0.8)
    for f in risers:
        d = df.filter(pl.col("feature") == f).sort("u_min")
        lw = 2.2 if f == "seconds_to_funding" else 1.1
        axes[1].plot(d["u_min"], d["rel"], marker="o", ms=2.5, lw=lw, label=f)
    axes[1].axhline(1.0, color="grey", lw=0.8, ls="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"backward time to settlement $u$ (min)")
    axes[1].set_ylabel("share / bulk share")
    axes[1].set_title("(b) attribution relative to bulk")
    axes[1].invert_xaxis()
    axes[1].set_ylim(bottom=0.0)
    axes[1].legend(frameon=True, framealpha=1.0, edgecolor="0.85",
                   fancybox=False, loc="lower left", handlelength=1.2,
                   fontsize=7, labelspacing=0.3)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    for ax in axes:
        ax.xaxis.set_major_locator(FixedLocator([100, 10, 1]))
        ax.xaxis.set_major_formatter(FixedFormatter(["100", "10", "1"]))
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    _save(fig, out_base)


def fig_interaction_heatmap(inter_csv: Path, out_base: Path) -> None:
    feats, M = _pivot_matrix(pl.read_csv(inter_csv), "mean_abs_interaction")
    M = np.abs(M)
    np.fill_diagonal(M, np.nan)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(feats)))
    ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=75, ha="right")
    ax.set_yticklabels(feats)
    ax.set_title("LightGBM SHAP interactions (mean |interaction|, diagonal masked)")
    fig.colorbar(im, ax=ax, label="mean(|interaction|)")
    _save(fig, out_base)


def fig_interaction_diff(diff_csv: Path, out_base: Path) -> None:
    feats, M = _pivot_matrix(pl.read_csv(diff_csv), "diff_boundary_minus_bulk")
    np.fill_diagonal(M, np.nan)
    finite = M[np.isfinite(M)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    vmax = vmax or 1.0
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(feats)))
    ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=75, ha="right")
    ax.set_yticklabels(feats)
    ax.set_title("SHAP interaction change: boundary - bulk\n"
                 "(red = stronger near settlement, diagonal masked)")
    fig.colorbar(im, ax=ax, label=r"$\Delta$ mean(|interaction|)  (boundary - bulk)")
    _save(fig, out_base)


def fig_nonadditivity(buckets_csv: Path, out_base: Path) -> None:
    df = pl.read_csv(buckets_csv)
    order = [b for b in ("bulk", "boundary", "pooled")
             if b in df["bucket"].to_list()]
    if not order:
        print(f"   skip {out_base.name}: no regimes in buckets csv")
        return
    rows = {r["bucket"]: r for r in df.to_dicts()}
    means = [rows[b]["mean_non_additivity_ratio"] for b in order]
    sems = [rows[b].get("sem_non_additivity_ratio") or 0.0 for b in order]
    colors = {"bulk": "#1f77b4", "boundary": "#d62728", "pooled": "#7f7f7f"}
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.bar(order, means, yerr=sems, capsize=4,
           color=[colors[b] for b in order])
    ax.axhline(0.0, color="black", lw=1.0)
    ax.text(-0.45, 0.0, "Ridge = 0", va="bottom", ha="left",
            fontsize=8, color="grey")
    nf = rows[order[0]].get("n_folds")
    ax.set_ylabel("non-additive share of |SHAP|\n(off-diagonal / total)")
    ax.set_title(f"LightGBM non-additivity by regime"
                 + (f"  (mean +/- sem, {nf} folds)" if nf else ""))
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    _save(fig, out_base)


def _beeswarm(bee_parquet: Path, global_csv: Path, out_base: Path,
              boundary_only: bool) -> None:
    import shap  # imported lazily; only the beeswarms need it
    df = pl.read_parquet(bee_parquet)
    if boundary_only:
        df = df.filter(pl.col("bucket").is_in(list(BOUNDARY_LABELS)))
    if df.height == 0:
        print(f"   skip {out_base.name}: no rows after filter")
        return
    # order features by global importance, restricted to those persisted
    persisted = {c[len("shap__"):] for c in df.columns if c.startswith("shap__")}
    order = [f for f in pl.read_csv(global_csv)["feature"].to_list()
             if f in persisted]
    if not order:
        print(f"   skip {out_base.name}: no beeswarm features")
        return
    sv = np.column_stack([df[f"shap__{f}"].to_numpy() for f in order])
    xv = np.column_stack([df[f"val__{f}"].to_numpy() for f in order])
    fig = plt.figure(figsize=(8, 0.32 * len(order) + 1))
    shap.summary_plot(sv, features=xv, feature_names=order,
                      max_display=len(order), show=False, plot_size=None)
    suffix = " (boundary rows only)" if boundary_only else ""
    plt.title("SHAP beeswarm, representative fold" + suffix, fontsize=10)
    _save(fig, out_base)


def fig_ridge_contrast(contrast_csv: Path, out_base: Path) -> None:
    df = pl.read_csv(contrast_csv).with_columns([
        (pl.col("mean_abs_shap") / pl.col("mean_abs_shap").max()).alias("lgbm_norm"),
        (pl.col("ridge_abs_coef").fill_null(0.0)
         / pl.col("ridge_abs_coef").fill_null(0.0).max().clip(lower_bound=1e-12))
        .alias("ridge_norm"),
    ]).sort("mean_abs_shap", descending=True)
    feats = df["feature"].to_list()[::-1]
    lgb_vals = df["lgbm_norm"].to_numpy()[::-1]
    rdg_vals = df["ridge_norm"].to_numpy()[::-1]
    dropped = df["ridge_dropped"].to_numpy()[::-1]
    y = np.arange(len(feats))
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(feats) + 1))
    ax.barh(y - 0.2, lgb_vals, height=0.4, label="LightGBM mean(|SHAP|)",
            color="#1f77b4")
    ax.barh(y + 0.2, rdg_vals, height=0.4, label="Ridge |std. coef|",
            color="#ff7f0e")
    for i, drop in enumerate(dropped):
        if drop:
            ax.text(0.01, i + 0.2, "Ridge: dropped", va="center", ha="left",
                    fontsize=7, color="grey")
    ax.set_yticks(y)
    ax.set_yticklabels(feats)
    ax.set_xlabel("normalised to per-model max (= 1.0)")
    ax.set_title("LightGBM SHAP vs Ridge standardised coefficient")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    _save(fig, out_base)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shap-dir", type=Path, required=True,
                    help="ml_shap.py output dir (e.g. <base>/models/shap_drift_mid_h10)")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=15,
                    help="top-k features for the global bar")
    args = ap.parse_args()

    d = args.shap_dir
    h = args.horizon
    figs = d / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    print(f"figures -> {figs}")

    def have(name: str) -> Path | None:
        p = d / name
        if p.exists():
            return p
        print(f"   skip: missing {name}")
        return None

    global_csv = have(f"shap_global_h{h}.csv")
    if global_csv:
        fig_global_bar(global_csv, args.top_k, figs / "shap_global")

    if (bl := have(f"shap_boundary_layer_h{h}.csv")):
        _boundary_panels(
            bl, "mean_abs_shap", "mean(|SHAP|)",
            "Boundary-layer SHAP attribution by minutes-to-settlement",
            figs / "shap_boundary_layer", diverging=False)
        _boundary_panels(
            bl, "mean_signed_shap", "mean(SHAP)",
            "Directional (signed) boundary-layer SHAP by minutes-to-settlement",
            figs / "shap_boundary_layer_signed", diverging=True)

    if (prof := have(f"shap_boundary_profile_h{h}.parquet")):
        fig_profile(prof, figs / "shap_profile")

    if (inter := have(f"shap_interactions_h{h}.csv")):
        fig_interaction_heatmap(inter, figs / "shap_interactions")
    if (diff := have(f"shap_interactions_diff_h{h}.csv")):
        fig_interaction_diff(diff, figs / "shap_interactions_diff")
    if (bk := have(f"shap_interaction_buckets_h{h}.csv")):
        fig_nonadditivity(bk, figs / "shap_nonadditivity")

    if (bee := have(f"shap_beeswarm_h{h}.parquet")) and global_csv:
        _beeswarm(bee, global_csv, figs / "shap_beeswarm", boundary_only=False)
        _beeswarm(bee, global_csv, figs / "shap_beeswarm_boundary",
                  boundary_only=True)

    if (rc := have(f"shap_ridge_contrast_h{h}.csv")):
        fig_ridge_contrast(rc, figs / "shap_ridge_contrast")

    print("done.")


if __name__ == "__main__":
    main()
