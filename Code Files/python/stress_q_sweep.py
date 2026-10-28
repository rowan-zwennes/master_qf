"""Inventory-capacity sweep under the high-carry scenarios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from run_simulation import SimConfig
from stress_engine import run_scenarios, scenario_table
from stress_funding_ou import build_scenarios

SIDS = (2, 5, 6)
SCENARIOS = ("cap_pinned", "documented_carry")
Q_NEW = (20, 40)
Q_BASE = 10
HEADLINE_PATHS = Path("reports/stress_20260710/funding_ou/"
                      "stress_funding_paths.parquet")


def run_q(q: int, n_paths: int, processes: int,
          epoch_s: int = 3_600, n_epochs: int = 2,
          sids: tuple[int, ...] = SIDS,
          scenarios: tuple[str, ...] = SCENARIOS) -> pl.DataFrame:
    cfg = SimConfig(lut_min_rebuild_s=60.0, ml_shift_mode="horizon", Q=q)
    specs = [s for s in build_scenarios(epoch_s, n_epochs)
             if s.name in scenarios]
    df = run_scenarios(specs, n_paths, sids, cfg=cfg, processes=processes)
    return df.with_columns(pl.lit(q).alias("Q"))


def q_summary(paths: pl.DataFrame) -> pl.DataFrame:
    """Per (Q, scenario, strategy): the lever diagnostics. Sharpe is the
    per-path mean/sd (NOT annualised: synthetic days have no calendar),
    consistent with tail_sharpe's convention in stress_engine."""
    recs = []
    for (q, scen, sid), grp in sorted(
            paths.group_by("Q", "scenario", "strategy"),
            key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        x = grp["terminal_pnl"].to_numpy()
        sd = float(x.std(ddof=1)) if x.size > 1 else float("nan")
        recs.append({
            "Q": q, "scenario": scen, "strategy": sid, "n_paths": x.size,
            "mean_pnl": float(x.mean()), "sd_pnl": sd,
            "sharpe_path": float(x.mean() / sd) if sd > 0 else float("nan"),
            "mean_funding": float(grp["funding"].mean()),
            "mean_signed_inv": float(grp["mean_signed_inv"].mean()),
            "mean_abs_inv": float(grp["mean_abs_inv"].mean()),
            "mean_cap_time": float(grp["frac_time_at_cap"].mean()),
            "cvar5": float(np.sort(x)[:max(int(np.ceil(0.05 * x.size)), 1)]
                           .mean()),
            "worst": float(x.min()),
        })
    return pl.DataFrame(recs)


def load_q10_leg(headline: Path, scenarios: tuple[str, ...] = SCENARIOS,
                 sids: tuple[int, ...] = SIDS) -> pl.DataFrame:
    df = pl.read_parquet(headline)
    return (df.filter(pl.col("scenario").is_in(list(scenarios))
                      & pl.col("strategy").is_in(list(sids)))
            .with_columns(pl.lit(Q_BASE).alias("Q")))


def write_q_sweep_tex(paths: pl.DataFrame, path: Path) -> None:
    """Compact appendix table: per (scenario, Q), s2 vs s5 economics and the
    paired s5-s2 gap (seeds shared across Q legs and strategies). s6 is
    omitted as a column: it tracks s5 within ~2 USD everywhere (caption)."""
    scen_label = {"cap_pinned": "Cap-pinned", "documented_carry": "Doc.\\ carry"}
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{GLT (s2)} & \multicolumn{3}{c}{Funding (s5)}"
        r" & \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
        r"Scenario & $Q$ & P\&L & ES$_{5\%}$ & P\&L & Funding & ES$_{5\%}$ &"
        r" $\Delta$(s5$-$s2) \\",
        r"\midrule",
    ]
    for scen in ("cap_pinned", "documented_carry"):
        for q in sorted(paths.filter(pl.col("scenario") == scen)["Q"]
                        .unique().to_list()):
            sub = paths.filter((pl.col("scenario") == scen)
                               & (pl.col("Q") == q))
            wide = sub.pivot(on="strategy", index="seed",
                             values="terminal_pnl").sort("seed")
            gap = (wide["5"] - wide["2"]).to_numpy()
            se = gap.std(ddof=1) / np.sqrt(gap.size)
            row = {}
            for sid in (2, 5):
                g = sub.filter(pl.col("strategy") == sid)
                x = g["terminal_pnl"].to_numpy()
                k5 = max(int(np.ceil(0.05 * x.size)), 1)
                row[sid] = (x.mean(), np.sort(x)[:k5].mean(),
                            g["funding"].mean())
            lines.append(
                f"{scen_label[scen]} & {q} & {row[2][0]:.0f} & {row[2][1]:.0f}"
                f" & {row[5][0]:.0f} & {row[5][2]:.1f} & {row[5][1]:.0f}"
                f" & {gap.mean():+.1f} $\\pm$ {se:.1f} \\\\")
        lines.append(r"\midrule" if scen == "cap_pinned" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def run(n_paths: int, processes: int, out_dir: Path,
        qs: tuple[int, ...] = Q_NEW,
        headline: Path = HEADLINE_PATHS,
        sids: tuple[int, ...] = SIDS) -> pl.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    legs = [load_q10_leg(headline, sids=sids)]
    for q in qs:
        leg = run_q(q, n_paths, processes, sids=sids)
        leg.write_parquet(out_dir / f"q{q}_paths.parquet")  # checkpoint
        legs.append(leg)
    paths = pl.concat(legs, how="vertical_relaxed")
    paths.write_parquet(out_dir / "q_sweep_paths.parquet")
    summ = q_summary(paths)
    summ.write_parquet(out_dir / "q_sweep_summary.parquet")
    write_q_sweep_tex(paths, out_dir / "tab_q_sweep.tex")
    with open(out_dir / "q_sweep_summary.json", "w") as fh:
        json.dump({"n_paths": n_paths, "qs": [Q_BASE, *qs],
                   "sids": list(sids), "scenarios": list(SCENARIOS),
                   "q10_source": str(headline),
                   "rows": summ.to_dicts()}, fh, indent=1)
    return summ


def main() -> None:
    ap = argparse.ArgumentParser(description="Q-capacity sweep (item 13).")
    ap.add_argument("--n-paths", type=int, default=200)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("reports/stress_20260710/q_sweep"))
    ap.add_argument("--headline", type=Path, default=HEADLINE_PATHS,
                    help="funding-suite paths parquet supplying the Q=10 leg; "
                         "must come from the same run as the headline table")
    ap.add_argument("--sids", type=str, default=None,
                    help="comma list of strategy ids (default: 2,5,6)")
    args = ap.parse_args()
    sids = (tuple(int(s) for s in args.sids.split(",")) if args.sids
            else SIDS)
    summ = run(args.n_paths, args.processes, args.out_dir,
               headline=args.headline, sids=sids)
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(summ)


if __name__ == "__main__":
    main()
