"""Rebuild the stress scenario table from an existing paths parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from stress_engine import scenario_table, write_scenario_tex
from stress_funding_ou import CAP

def _footer(df: pl.DataFrame) -> list[str]:
    gaps = []
    for sc in sorted(df["scenario"].unique().to_list()):
        w = (df.filter(pl.col("scenario") == sc)
             .pivot(on="strategy", index="seed", values="terminal_pnl")
             .sort("seed"))
        if not {"2", "5"} <= set(w.columns):
            continue
        a, b = w["2"].to_numpy(), w["5"].to_numpy()
        d = b - a
        se = d.std(ddof=1) / np.sqrt(d.size)
        name = sc.replace("_", " ")
        gaps.append(rf"{name} ${d.mean():+.1f} \pm {se:.2f}$")
    return [
        r"\textit{Paired mean P\&L gap, s5 $-$ s2, USDT/path $\pm$ s.e.:} "
        + "; ".join(gaps) + ".",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="stress root containing funding_ou/")
    ap.add_argument("--half-spread", type=float, default=8.6)
    args = ap.parse_args()

    src = args.root / "funding_ou" / "stress_funding_paths.parquet"
    df = pl.read_parquet(src)
    n_paths = (df.filter((pl.col("scenario") == df["scenario"][0])
                         & (pl.col("strategy") == df["strategy"][0])).height)
    out = args.root / "funding_ou" / "tab_stress_funding.tex"
    write_scenario_tex(
        scenario_table(df), out,
        caption=(rf"Ornstein-Uhlenbeck funding-rate stress: {n_paths} paired "
                 rf"synthetic paths per scenario, funding cap "
                 rf"$\pm${CAP * 1e4:.0f} bp. The naive arm (strategy 1) quotes "
                 rf"at a width-matched fixed half-spread of "
                 rf"{args.half_spread:g} USDT. Scenario definitions are in "
                 rf"Appendix~\ref{{app:stress_spec}}."),
        label="tab:stress_funding",
        footer_lines=_footer(df))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
