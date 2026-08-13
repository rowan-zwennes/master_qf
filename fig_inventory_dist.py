"""Inventory distribution figure."""
from __future__ import annotations
import glob
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt/runs/rerun_20260807_ar"
QS, Q = 0.005, 10
SERIES = [
    ("Naive (matched spread)", f"{R}/frontier/naive_hs8p6", 1, "tab:gray"),
    ("GLT", f"{R}/ablation_h10_qa", 2, "tab:blue"),
    ("GLT+ML", f"{R}/ablation_h10_qa", 4, "tab:red"),
]


def q_hist(run: str, s: int) -> tuple[np.ndarray, float, float]:
    """Density over integer q in [-Q, Q], plus (E[q^2], frac time at |q|=Q)."""
    edges = np.arange(-Q - 0.5, Q + 1.5, 1.0)
    counts = np.zeros(len(edges) - 1)
    eq2n = eq2d = capn = tot = 0.0
    for f in sorted(glob.glob(f"{run}/s{s}/pnl_*.parquet")):
        d = pl.read_parquet(f, columns=["inv", "paused"]).filter(~pl.col("paused"))
        if d.height == 0:
            continue
        q = np.round(d["inv"].to_numpy() / QS)          # lots (integer)
        counts += np.histogram(q, bins=edges)[0]
        eq2n += float(np.sum(q ** 2)); eq2d += q.size
        capn += float(np.sum(np.abs(q) >= Q)); tot += q.size
    dens = counts / counts.sum()
    return dens, eq2n / eq2d, 100.0 * capn / tot


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 9})
    centers = np.arange(-Q, Q + 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for lab, run, s, col in SERIES:
        dens, eq2, cap = q_hist(run, s)
        print(f"  {lab}: E[q^2]={eq2:.1f}, {cap:.2f}% at cap")
        ax.step(centers, dens, where="mid", color=col, lw=1.7, label=lab)
        if s == 1:                                       # fill the star only
            ax.fill_between(centers, dens, step="mid", color=col, alpha=0.15)
    for x in (-Q, Q):
        ax.axvline(x, color="k", lw=0.6, ls=":")
    ytop = ax.get_ylim()[1]
    ax.text(-Q + 0.25, ytop * 0.97, "$q=-Q$", fontsize=8,
            va="top", ha="left", color="0.35")
    ax.text(Q - 0.25, ytop * 0.97, "$q=Q$", fontsize=8,
            va="top", ha="right", color="0.35")
    ax.set_xlabel("held inventory $q_t$ (lots)")
    ax.set_ylabel("fraction of live OOS time")
    ax.set_xticks(centers[::2])
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0, 1.0, 0.08),
              frameon=False, handlelength=1.6, columnspacing=1.2)
    ax.margins(x=0.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"reports/rerun_20260807_ar/fig_inventory_dist.{ext}"
        fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote reports/rerun_20260807_ar/fig_inventory_dist.{pdf,png}")


if __name__ == "__main__":
    main()
