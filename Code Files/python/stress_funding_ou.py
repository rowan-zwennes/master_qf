"""Ornstein-Uhlenbeck funding-rate stress scenarios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from run_simulation import SimConfig
from stress_engine import (
    StressSpec,
    run_scenarios,
    scenario_table,
    write_scenario_tex,
)

SIDS = (1, 2, 4, 5, 6)
CAP = 7.5e-3


def build_scenarios(epoch_s: int = 3_600, n_epochs: int = 2) -> list[StressSpec]:
    common = dict(epoch_s=epoch_s, n_epochs=n_epochs)
    return [
        StressSpec(name="baseline_const", funding_const=1.0e-4, **common),
        StressSpec(name="persistent_pos",
                   funding_ou=(+0.6 * CAP, 14_400.0, 0.1 * CAP, CAP,
                               +0.6 * CAP), **common),
        StressSpec(name="persistent_neg",
                   funding_ou=(-0.6 * CAP, 14_400.0, 0.1 * CAP, CAP,
                               -0.6 * CAP), **common),
        StressSpec(name="cap_pinned",
                   funding_ou=(+CAP, 14_400.0, 0.05 * CAP, CAP, +CAP),
                   **common),
        StressSpec(name="documented_carry",
                   funding_ou=(+5.5e-4, 14_400.0, 5.5e-5, CAP, +5.5e-4),
                   **common),
        StressSpec(name="sign_flip",
                   funding_ou=(0.0, 600.0, 0.8 * CAP, CAP, 0.0), **common),
    ]


def funding_gap_table(df: pl.DataFrame) -> pl.DataFrame:
    """Per scenario: paired P&L gaps plus carry diagnostics."""
    contrasts = ((5, 2), (6, 2), (4, 2), (6, 5), (6, 4))
    recs = []
    for scen in sorted(df["scenario"].unique().to_list()):
        sub = df.filter(pl.col("scenario") == scen)
        wide = sub.pivot(on="strategy", index="seed",
                         values="terminal_pnl").sort("seed")
        row = {"scenario": scen, "n_paths": wide.height}
        for hi, lo in contrasts:
            if str(hi) in wide.columns and str(lo) in wide.columns:
                gap = (wide[str(hi)] - wide[str(lo)]).to_numpy()
                row[f"gap_s{hi}_s{lo}_mean"] = float(gap.mean())
                row[f"gap_s{hi}_s{lo}_se"] = (float(gap.std(ddof=1)
                                                    / np.sqrt(gap.size))
                                              if gap.size > 1 else float("nan"))
        for sid in (2, 4, 5, 6):
            s = sub.filter(pl.col("strategy") == sid)
            if s.height:
                row[f"funding_s{sid}"] = float(s["funding"].mean())
                row[f"inv_s{sid}"] = float(s["mean_signed_inv"].mean())
        recs.append(row)
    return pl.DataFrame(recs)


def fig_gap_vs_carry(df: pl.DataFrame, gap_tab: pl.DataFrame,
                     path: Path) -> None:
    """s6-s2 and s5-s2 mean paired gap per scenario vs mean settlement |f|."""
    design_absf = {"baseline_const": 1.0e-4,
                   "documented_carry": 5.5e-4,
                   "persistent_pos": 0.6 * CAP, "persistent_neg": 0.6 * CAP,
                   "cap_pinned": CAP, "sign_flip": 0.8 * CAP * 0.798}
    fig, ax = plt.subplots(figsize=(7, 4))
    for sid, col in ((5, "tab:orange"), (6, "tab:red"), (4, "tab:green")):
        xs, ys, es, labs = [], [], [], []
        for r in gap_tab.to_dicts():
            if (f"gap_s{sid}_s2_mean" not in r
                    or r["scenario"] not in design_absf):
                continue
            xs.append(design_absf[r["scenario"]] * 1e4)
            ys.append(r[f"gap_s{sid}_s2_mean"])
            es.append(1.96 * (r.get(f"gap_s{sid}_s2_se") or 0.0))
            labs.append(r["scenario"])
        ax.errorbar(xs, ys, yerr=es, fmt="o", color=col,
                    label=f"s{sid} - s2 paired gap")
        for x, y, lab in zip(xs, ys, labs):
            ax.annotate(lab.replace("_", " "), (x, y), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"scenario design mean $|f_t|$ (bp per epoch)")
    ax.set_ylabel("paired P&L gap vs s2 (USDT)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(n_paths: int, processes: int, out_dir: Path,
        epoch_s: int = 3_600, n_epochs: int = 2,
        sids: tuple[int, ...] = SIDS,
        ml_shift_mode: str = "horizon",
        fixed_half_spread: float = SimConfig.fixed_half_spread,
        scenarios: str | None = None) -> pl.DataFrame:
    cfg = SimConfig(lut_min_rebuild_s=60.0, ml_shift_mode=ml_shift_mode,
                    fixed_half_spread=fixed_half_spread)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = build_scenarios(epoch_s, n_epochs)
    if scenarios is not None:
        want = {s.strip() for s in scenarios.split(",")}
        unknown = want - {sp.name for sp in specs}
        if unknown:
            raise SystemExit(f"unknown scenario(s): {sorted(unknown)}")
        specs = [sp for sp in specs if sp.name in want]
    df = run_scenarios(specs, n_paths, sids,
                       cfg=cfg, processes=processes)
    df.write_parquet(out_dir / "stress_funding_paths.parquet")
    tab = scenario_table(df)
    tab.write_parquet(out_dir / "stress_funding_table.parquet")
    gap = funding_gap_table(df)
    gap.write_parquet(out_dir / "stress_funding_gaps.parquet")
    write_scenario_tex(
        tab, out_dir / "tab_stress_funding.tex",
        caption=(rf"Ornstein-Uhlenbeck funding-rate stress: {n_paths} paired synthetic paths "
                 rf"per scenario, funding cap $\pm${CAP * 1e4:.0f} bp."
                 + (r" The signal-carrying strategies (4, 6) run the symmetric "
                    r"horizon coupling." if {4, 6} & set(sids) else "")
                 + (rf" The naive arm (strategy 1) quotes at a width-matched "
                    rf"fixed half-spread of {fixed_half_spread:g} USDT."
                    if 1 in sids else "")
                 + (r" Every entry is a mean over the "
                    rf"{n_paths} paths of that scenario: Mean P\&L, "
                    r"ES$_{5\%}$ (the mean of the worst $5\%$ of paths) and "
                    r"Funding collected are USDT per path, Cap (\%) is the "
                    r"share of path time spent at the inventory limit "
                    r"$|q| = Q$, Fills is the path fill count, and P\&L/fill "
                    r"and Markout are per fill, in USDT and in basis points "
                    r"of the fill price respectively.")),
        label="tab:stress_funding")
    fig_gap_vs_carry(df, gap, out_dir / "fig_stress_funding_gap.pdf")
    return gap


def main() -> None:
    ap = argparse.ArgumentParser(description="OU funding-rate stress test.")
    ap.add_argument("--n-paths", type=int, default=200)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--epoch-s", type=int, default=3_600)
    ap.add_argument("--n-epochs", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/stress"))
    ap.add_argument("--sids", type=str, default=None,
                    help="comma list of strategy ids (default: all); seeds "
                         "depend on the path index only")
    ap.add_argument("--fixed-half-spread", type=float,
                    default=SimConfig.fixed_half_spread,
                    help="naive (strategy 1) half-spread in USDT")
    ap.add_argument("--scenarios", type=str, default=None,
                    help="comma list of scenario names (default: all)")
    args = ap.parse_args()
    sids = (tuple(int(s) for s in args.sids.split(",")) if args.sids
            else SIDS)
    gap = run(args.n_paths, args.processes, args.out_dir,
              args.epoch_s, args.n_epochs, sids=sids,
              fixed_half_spread=args.fixed_half_spread,
              scenarios=args.scenarios)
    print(gap)
    print(json.dumps({"out": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
