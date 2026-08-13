"""Sensitivity of the full hybrid to gamma and k miscalibration."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from run_simulation import BASE_DEFAULT, INTENSITY_DIR_DEFAULT, SimConfig
from sensitivity_common import cell_summary, run_cell

GAMMA_SCALES = (0.5, 1.0, 1.5)
K_SCALES = (0.5, 1.0, 1.5)
A_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)
GAMMA_SCALES_WIDE = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
SID = 6


def _tag_gk(g: float, k: float) -> str:
    return f"gk_g{g:g}_k{k:g}"


def _tag_a(a: float) -> str:
    return f"A_{a:g}"


def sweep(cfg0: SimConfig, days_or_n, sids: tuple[int, ...], out_root: Path,
          *, gamma_scales=GAMMA_SCALES, k_scales=K_SCALES,
          a_scales=A_SCALES, real_kw: dict | None = None) -> pl.DataFrame:
    rows = []
    cells: list[tuple[str, SimConfig, dict]] = []
    for g in gamma_scales:
        for k in k_scales:
            cells.append((_tag_gk(g, k),
                          replace(cfg0, gamma_scale=g, k_scale=k),
                          {"sweep": "gk", "gamma_scale": g, "k_scale": k,
                           "A_scale": 1.0}))
    for a in a_scales:
        if a == 1.0:
            continue  # shares the gk centre cell
        cells.append((_tag_a(a), replace(cfg0, A_scale=a),
                      {"sweep": "A", "gamma_scale": 1.0, "k_scale": 1.0,
                       "A_scale": a}))
    for tag, cfg, meta in cells:
        cell = run_cell(tag, cfg, days_or_n, sids, out_root=out_root,
                        **real_kw)
        for sid in sids:
            rows.append({"tag": tag, "strategy": sid, **meta,
                         **cell_summary(cell, sid)})
    df = pl.DataFrame(rows)
    if not a_scales:
        return df
    centre = _tag_gk(1.0, 1.0)
    a_rows = (df.filter(pl.col("tag") == centre)
              .with_columns(pl.lit("A").alias("sweep"),
                            pl.lit(centre).alias("tag")))
    return pl.concat([df, a_rows], how="vertical_relaxed")


def _grid_matrix(df: pl.DataFrame, value: str, gamma_scales, k_scales,
                 sid: int = SID) -> np.ndarray:
    m = np.full((len(gamma_scales), len(k_scales)), np.nan)
    for i, g in enumerate(gamma_scales):
        for j, k in enumerate(k_scales):
            sub = df.filter((pl.col("tag") == _tag_gk(g, k))
                            & (pl.col("strategy") == sid))
            if sub.height:
                m[i, j] = float(sub[value][0])
    return m


def fig_heatmap(m: np.ndarray, gamma_scales, k_scales, path: Path,
                title: str, fmt: str = "{:.2f}") -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(m, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(k_scales)),
                  [f"{k:g}x" for k in k_scales])
    ax.set_yticks(np.arange(len(gamma_scales)),
                  [f"{g:g}x" for g in gamma_scales])
    ax.set_xlabel(r"$k$ scale")
    ax.set_ylabel(r"$\gamma$ scale")
    ax.set_title(title, fontsize=10)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center",
                        fontsize=8)
    ci, cj = list(gamma_scales).index(1.0), list(k_scales).index(1.0)
    ax.add_patch(plt.Rectangle((cj - 0.5, ci - 0.5), 1, 1, fill=False,
                               edgecolor="black", lw=2))
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_a_sweep(df: pl.DataFrame, a_scales, path: Path,
                sid: int = SID) -> None:
    xs, sh, fr = [], [], []
    for a in a_scales:
        tag = _tag_gk(1.0, 1.0) if a == 1.0 else _tag_a(a)
        sub = df.filter((pl.col("tag") == tag)
                        & (pl.col("strategy") == sid)).head(1)
        if sub.height:
            xs.append(a)
            sh.append(float(sub["sharpe"][0]))
            fr.append(float(sub["fills_per_day"][0]))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(xs, sh, "o-", color="tab:red", label="Sharpe")
    ax.set_xlabel(r"$A$ scale")
    ax.set_ylabel("annualised Sharpe", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(xs, fr, "s--", color="tab:blue", label="fills/day")
    ax2.set_ylabel("fills per day", color="tab:blue")
    ax.axvline(1.0, color="k", lw=0.6, ls=":")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_tex(df: pl.DataFrame, gamma_scales, k_scales, path: Path,
              sid: int = SID) -> None:
    sh = _grid_matrix(df, "sharpe", gamma_scales, k_scales, sid)
    # summaries store mean_abs_inv in BTC; print LOTS (0.005 BTC each)
    q = _grid_matrix(df, "mean_abs_inv", gamma_scales, k_scales, sid) / 0.005
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Sensitivity of the full hybrid strategy to joint "
        r"$\pm 50\%$ miscalibration of $\gamma$ and $k$: annualised Sharpe "
        r"(mean $|q|$ in lots in parentheses). The centre cell is the "
        r"calibrated configuration.}",
        r"\label{tab:sens_gk}",
        r"\begin{tabular}{l" + "r" * len(k_scales) + "}", r"\toprule",
        " & " + " & ".join(rf"$k \times {k:g}$" for k in k_scales) + r" \\",
        r"\midrule",
    ]
    for i, g in enumerate(gamma_scales):
        cells = " & ".join(
            f"{sh[i, j]:.2f} ({q[i, j]:.1f})" if np.isfinite(sh[i, j])
            else "n/a" for j in range(len(k_scales)))
        lines.append(rf"$\gamma \times {g:g}$ & " + cells + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def write_tex_1d(df: pl.DataFrame, gamma_scales, path: Path,
                 gamma0: float, sid: int = SID) -> None:
    """1-D gamma table (the reported form). Columns are chosen to show the
    CHANNEL gamma acts through: it is an inventory dial, so mean |q| and the
    time spent at the cap matter more than the Sharpe cell it lands in."""
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Sensitivity of the full hybrid strategy to "
        r"miscalibration of the risk-aversion hyperparameter $\gamma$ over a "
        r"factor of sixteen, on the sixteen-day sweep subset. "
        r"$\gamma \times 1$ is the calibrated configuration. Mean $|q|$ is in "
        r"lots of $0.005$ BTC.}",
        r"\label{tab:sens_gamma}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"& $\gamma$ & P\&L/day & daily s.d. & mean $|q|$ & fills/day \\",
        r"& & (USDT) & (USDT) & (lots) & \\",
        r"\midrule",
    ]
    for g in gamma_scales:
        sub = df.filter((pl.col("tag") == _tag_gk(g, 1.0))
                        & (pl.col("strategy") == sid)).head(1)
        if not sub.height:
            continue
        r = sub.row(0, named=True)
        lines.append(
            rf"$\gamma \times {g:g}$ & {gamma0 * g:.2e} & "
            rf"{r['mean_daily_pnl']:.0f} & {r['sd_daily_pnl']:.0f} & "
            rf"{r['mean_abs_inv'] / 0.005:.2f} & "
            rf"{r['fills_per_day']:.0f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def fig_gamma_1d(df: pl.DataFrame, gamma_scales, path: Path,
                 sid: int = SID) -> None:
    xs, sh, q = [], [], []
    for g in gamma_scales:
        sub = df.filter((pl.col("tag") == _tag_gk(g, 1.0))
                        & (pl.col("strategy") == sid)).head(1)
        if sub.height:
            xs.append(g)
            sh.append(float(sub["sharpe"][0]))
            q.append(float(sub["mean_abs_inv"][0]) / 0.005)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(xs, sh, "o-", color="tab:red", label="Sharpe")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$ scale")
    ax.set_ylabel("annualised Sharpe", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(xs, q, "s--", color="tab:blue", label="mean |q| (lots)")
    ax2.set_ylabel("mean |q| (lots)", color="tab:blue")
    ax.axvline(1.0, color="k", lw=0.6, ls=":")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def verdict(df: pl.DataFrame, gamma_scales, k_scales,
            sid: int = SID) -> dict:
    sh = _grid_matrix(df, "sharpe", gamma_scales, k_scales, sid)
    centre = sh[list(gamma_scales).index(1.0), list(k_scales).index(1.0)]
    cliff = []
    if np.isfinite(centre) and centre > 0:
        for i, g in enumerate(gamma_scales):
            for j, k in enumerate(k_scales):
                if np.isfinite(sh[i, j]) and sh[i, j] < 0.25 * centre:
                    cliff.append(_tag_gk(g, k))
    return {"centre_sharpe": float(centre), "min_sharpe": float(np.nanmin(sh)),
            "max_sharpe": float(np.nanmax(sh)),
            "graceful": not cliff, "cliff_cells": cliff}


def run(out_root: Path, report_dir: Path, days_or_n, sids=(SID,), *,
        cfg0: SimConfig | None = None,
        gamma_scales=GAMMA_SCALES, k_scales=K_SCALES, a_scales=A_SCALES,
        real_kw: dict | None = None) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    cfg0 = cfg0 or SimConfig()
    df = sweep(cfg0, days_or_n, sids, out_root,
               gamma_scales=gamma_scales, k_scales=k_scales,
               a_scales=a_scales, real_kw=real_kw)
    df.filter(pl.col("sweep") == "gk").write_parquet(
        report_dir / "sens_gk_grid.parquet")
    if a_scales:
        df.filter(pl.col("sweep") == "A").write_parquet(
            report_dir / "sens_A_sweep.parquet")
        fig_a_sweep(df, a_scales, report_dir / "fig_sens_A.pdf")
    if len(k_scales) > 1:
        fig_heatmap(_grid_matrix(df, "sharpe", gamma_scales, k_scales),
                    gamma_scales, k_scales,
                    report_dir / "fig_sens_gk_sharpe.pdf",
                    "s6 Sharpe under gamma x k miscalibration")
        fig_heatmap(_grid_matrix(df, "mean_abs_inv", gamma_scales, k_scales),
                    gamma_scales, k_scales, report_dir / "fig_sens_gk_inv.pdf",
                    "s6 mean |q|", fmt="{:.1f}")
        write_tex(df, gamma_scales, k_scales, report_dir / "tab_sens_gk.tex")
    else:
        # reported form: 1-D gamma sweep at the calibrated k
        fig_gamma_1d(df, gamma_scales, report_dir / "fig_sens_gamma.pdf")
        write_tex_1d(df, gamma_scales, report_dir / "tab_sens_gamma.tex",
                     gamma0=cfg0.gamma)
    v = verdict(df, gamma_scales, k_scales)
    (report_dir / "sens_gk_verdict.json").write_text(json.dumps(v, indent=2))
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description="gamma x k (+A) sensitivity.")
    ap.add_argument("--base", type=Path, default=BASE_DEFAULT)
    ap.add_argument("--split", default="sim")
    ap.add_argument("--days", type=str)
    ap.add_argument("--alpha-parquet", type=Path, required=False)
    ap.add_argument("--intensity-dir", type=Path,
                    default=INTENSITY_DIR_DEFAULT)
    ap.add_argument("--out-root", type=Path, default=Path("runs/sensitivity"))
    ap.add_argument("--report-dir", type=Path,
                    default=Path("reports/sensitivity"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--gamma-scales", type=str, default=None,
                    help="comma list, e.g. 0.25,0.5,0.75,1,1.5,2,4")
    ap.add_argument("--k-scales", type=str, default=None,
                    help="comma list; a single value gives the 1-D gamma table")
    ap.add_argument("--a-scales", type=str, default=None,
                    help="comma list; empty string disables the A sweep")
    args = ap.parse_args()

    def _scales(s: str | None, default):
        if s is None:
            return default
        return tuple(float(x) for x in s.split(",") if x.strip())

    from pipeline_utils import load_splits
    days = (args.days.split(",") if args.days
            else load_splits(args.base)["splits"][args.split])
    real_kw = dict(base=args.base, alpha_parquet=args.alpha_parquet,
                   intensity_dir=args.intensity_dir, workers=args.workers)
    v = run(args.out_root, args.report_dir, days,
            gamma_scales=_scales(args.gamma_scales, GAMMA_SCALES),
            k_scales=_scales(args.k_scales, K_SCALES),
            a_scales=_scales(args.a_scales, A_SCALES),
            real_kw=real_kw)
    print(json.dumps(v, indent=2))


if __name__ == "__main__":
    main()
