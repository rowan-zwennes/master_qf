"""Shared runner for the sensitivity sweeps."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_simulation import SimConfig, run_one_day
from statistical_tests import sharpe_ratio


def cell_done(cell_dir: Path, days: list[str], sids: tuple[int, ...]) -> bool:
    return all((cell_dir / f"s{sid}" / f"summary_{d}.json").exists()
               for d in days for sid in sids)


def run_cell(tag: str, cfg: SimConfig, days: list[str],
             sids: tuple[int, ...], base: Path,
             alpha_parquet: Path | None, intensity_dir: Path,
             out_root: Path, workers: int = 2,
             alpha_ar_parquet: Path | None = None) -> Path:
    """One sweep cell over real data; resumable per day."""
    cell = out_root / tag
    cell.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days
            if not cell_done(cell, [d], sids)]
    (cell / "cell_config.json").write_text(json.dumps(
        {"tag": tag, "days": days, "sids": list(sids),
         "config": {k: v for k, v in cfg.__dict__.items()}}, indent=2,
        default=str))
    if not todo:
        return cell
    jobs = [(base, d, cfg, list(sids), alpha_parquet, intensity_dir, cell,
             alpha_ar_parquet) for d in todo]
    if workers <= 1:
        for jb in jobs:
            fday, info = run_one_day(jb)
            print(f"  [{tag}] {fday}: {info['elapsed_s']:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_one_day, jb): jb[1] for jb in jobs}
            for fut in as_completed(futs):
                fday, info = fut.result()
                print(f"  [{tag}] {fday}: {info['elapsed_s']:.0f}s",
                      flush=True)
    return cell


def cell_daily_pnl(cell: Path, sid: int) -> tuple[list[str], np.ndarray]:
    """Per-day terminal P&L of one strategy in one cell, day-sorted."""
    rows = []
    for fp in sorted((cell / f"s{sid}").glob("summary_*.json")):
        d = json.loads(fp.read_text())
        rows.append((d["funding_day"], float(d["terminal_pnl"])))
    rows.sort()
    return [r[0] for r in rows], np.asarray([r[1] for r in rows])


def cell_summary(cell: Path, sid: int) -> dict:
    """Headline metrics of one (cell, strategy): Sharpe, P&L, fills, |q|, fees, funding."""
    days, pnl = cell_daily_pnl(cell, sid)
    n_fills, mean_q, frac_cap, fees, funding = [], [], [], [], []
    for fp in sorted((cell / f"s{sid}").glob("summary_*.json")):
        d = json.loads(fp.read_text())
        n_fills.append(d["n_fills"])
        mean_q.append(d["mean_abs_inv"])
        frac_cap.append(d["frac_time_at_cap"])
        fees.append(d.get("fees", float("nan")))
        funding.append(d.get("funding", float("nan")))
    return {
        "n_days": len(days),
        "total_pnl": float(pnl.sum()) if pnl.size else float("nan"),
        "sharpe": sharpe_ratio(pnl) if pnl.size >= 3 else float("nan"),
        "mean_daily_pnl": float(pnl.mean()) if pnl.size else float("nan"),
        "sd_daily_pnl": float(pnl.std(ddof=1)) if pnl.size >= 2
        else float("nan"),
        "fills_per_day": float(np.mean(n_fills)) if n_fills else 0.0,
        "mean_abs_inv": float(np.mean(mean_q)) if mean_q else float("nan"),
        "frac_time_at_cap": float(np.mean(frac_cap)) if frac_cap
        else float("nan"),
        "fees_total": float(np.nansum(fees)) if fees else float("nan"),
        "funding_total": float(np.nansum(funding)) if funding
        else float("nan"),
    }
