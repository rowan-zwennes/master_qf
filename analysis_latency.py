"""Latency benchmark table from the C++ engine dumps."""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAGIC = b"HFTB"
PCTS = (50.0, 90.0, 99.0, 99.9)


def read_hftb(path: Path) -> dict[str, np.ndarray]:
    """{stage name: ns/op samples (float)}."""
    buf = path.read_bytes()
    if buf[:4] != MAGIC:
        raise ValueError(f"bad HFTB magic in {path}")
    version, n_stages, _pad = struct.unpack_from("<III", buf, 4)
    if version != 1:
        raise ValueError(f"unsupported HFTB version {version}")
    off = 16
    out: dict[str, np.ndarray] = {}
    for _ in range(n_stages):
        name = buf[off:off + 24].split(b"\x00")[0].decode()
        off += 24
        (n,) = struct.unpack_from("<Q", buf, off)
        off += 8
        ps = np.frombuffer(buf, dtype="<u8", count=n, offset=off)
        off += 8 * n
        out[name] = ps.astype(np.float64) / 1000.0   # ps -> ns
    return out


def stage_stats(ns: np.ndarray) -> dict:
    return {"n": int(ns.size),
            **{f"p{p:g}": float(np.percentile(ns, p)) for p in PCTS},
            "max": float(ns.max()) if ns.size else float("nan")}


def write_tex(stats: dict[str, dict], path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{C++ quoting hot-path latency (nanoseconds). Micro-bench "
        r"rows measure the arithmetic via batched timing; the replay row "
        r"timestamps each individual quote decision during a full-day "
        r"replay and includes the timer overhead (conservative).}",
        r"\label{tab:latency}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Stage & $n$ & p50 & p90 & p99 & p99.9 & max \\",
        r"\midrule",
    ]
    for name, s in stats.items():
        lines.append(
            f"{name.replace('_', ' ')} & {s['n']:,} & {s['p50']:,.0f} & "
            f"{s['p90']:,.0f} & {s['p99']:,.0f} & {s['p99.9']:,.0f} & "
            f"{s['max']:,.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def fig_hist(stages: dict[str, np.ndarray], path: Path) -> None:
    n_st = max(len(stages), 1)
    fig, axes = plt.subplots(1, n_st, figsize=(4.0 * n_st, 3.2))
    axes = np.atleast_1d(axes)
    for ax, (name, ns) in zip(axes, stages.items()):
        clip = np.percentile(ns, 99.5)
        ax.hist(np.clip(ns, None, clip), bins=80, color="tab:blue")
        ax.axvline(np.percentile(ns, 99), color="tab:red", lw=1.0,
                   label="p99")
        ax.set_title(name.replace("_", " "), fontsize=9)
        ax.set_xlabel("ns per op")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(dumps: list[Path], report_dir: Path) -> dict[str, dict]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stages: dict[str, np.ndarray] = {}
    for p in dumps:
        for name, ns in read_hftb(p).items():
            stages[name] = (np.concatenate([stages[name], ns])
                            if name in stages else ns)
    stats = {name: stage_stats(ns) for name, ns in stages.items()}
    write_tex(stats, report_dir / "tab_latency.tex")
    fig_hist(stages, report_dir / "fig_latency_hist.pdf")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Latency table.")
    ap.add_argument("dumps", nargs="*", type=Path)
    ap.add_argument("--report-dir", type=Path, default=Path("reports/latency"))
    args = ap.parse_args()
    if not args.dumps:
        raise SystemExit("provide one or more .hftb dumps")
    stats = run(args.dumps, args.report_dir)
    for name, s in stats.items():
        print(f"{name:24s} n={s['n']:>9,}  p50={s['p50']:>8,.1f}  "
              f"p99={s['p99']:>8,.1f}  p99.9={s['p99.9']:>9,.1f} ns")


if __name__ == "__main__":
    main()
