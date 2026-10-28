"""Appendix table for the hyperparameter-transfer experiment."""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

BASE = Path("/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt/validation")
OUT = Path(__file__).resolve().parent / "reports" / "tab_crossfold_stability.tex"
HORIZON_S = 10

PAIRINGS = [
    ("crossfold_stability", "Offset $+8$ (time-disjoint)"),
    ("crossfold_stability_off2", "Offset $+2$ (adjacent)"),
    ("crossfold_stability_reverse", "Reversed order"),
]


def pairing_stats(rows: list[dict]) -> dict:
    rows = [r for r in rows if r["horizon_s"] == HORIZON_S]
    sd = st.stdev([r["r2_own"] for r in rows])
    lf = [r for r in rows if r["foreign_fold"] < r["fold"]]
    return {
        "n": len(rows),
        "r2_own": st.mean(r["r2_own"] for r in rows),
        "r2_foreign": st.mean(r["r2_foreign"] for r in rows),
        "ratio": st.mean(r["d_r2_own_minus_foreign"] for r in rows) / sd,
        "n_lf": len(lf),
        "ratio_lf": (st.mean(r["d_r2_own_minus_foreign"] for r in lf) / sd
                     if lf else float("nan")),
    }


def render(stats: list[tuple[str, dict]]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Cross-fold hyperparameter transfer at the operational "
        r"horizon ($h = 10$\,s), 16 pre-analysis folds. Each fold is refit "
        r"under the hyperparameter vector tuned on a different (donor) fold; "
        r"$\Delta R^2/\mathrm{sd}$ is the mean own-minus-foreign $R^2$ "
        r"difference as a fraction of the fold-to-fold standard deviation of "
        r"own $R^2$. The leak-free column is the same ratio restricted to "
        r"donor folds strictly earlier than the test fold, over the same "
        r"fold-to-fold standard deviation.}",
        r"\label{tab:crossfold_stability}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"& & & \multicolumn{2}{c}{$\Delta R^2/\mathrm{sd}$} \\",
        r"\cmidrule(lr){4-5}",
        r"Pairing & $\bar{R}^2$ own & $\bar{R}^2$ foreign & "
        r"all donors & leak-free \\",
        r"\midrule",
    ]
    for label, s in stats:
        lines.append(
            f"{label} & {s['r2_own']:.4f} & {s['r2_foreign']:.4f} & "
            f"${s['ratio']:+.3f}$ & ${s['ratio_lf']:+.3f}$ "
            f"($n{{=}}{s['n_lf']}$) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    stats = []
    for name, label in PAIRINGS:
        rows = json.loads(
            (BASE / name / "crossfold_stability.json").read_text())["rows"]
        stats.append((label, pairing_stats(rows)))
    OUT.write_text(render(stats))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
