"""Shared machinery for the Monte Carlo stress suite."""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from run_simulation import (
    BOOK_LEVELS,
    DayArrays,
    SimConfig,
    simulate_day,
)

TICK = 0.1
HALF = 0.3  # synthetic touch half-spread, on the tick grid


@dataclass(frozen=True)
class StressSpec:
    """Everything that defines one stress scenario (path-level knobs)."""

    name: str
    epoch_s: int = 3_600              # compressed funding epoch
    n_epochs: int = 2
    s0: float = 100_000.0
    sigma0: float = 2.5               # USDT / sqrt(s), baseline true vol
    # drift signal
    alpha_sd: float = 0.05            # stationary sd of alpha_true (USDT/s)
    alpha_hl_s: float = 20.0          # OU half-life of alpha_true
    alpha_noise_sd: float = 0.05      # observation noise on alpha_ML
    # trades / liquidity
    trade_rate: float = 6.0           # taker arrivals per second
    k_true: float = 0.0963            # true depth-decay of aggressor sweeps
    # jumps (cascading liquidations)
    jump_times_s: tuple[float, ...] = ()
    jump_sigmas: tuple[float, ...] = ()   # sizes in units of sigma0*sqrt(1s)
    burst_trades: int = 25            # one-sided sweep trades per jump
    # multiplicative windows (start_s, end_s, mult)
    sigma_windows: tuple[tuple[float, float, float], ...] = ()
    liq_windows: tuple[tuple[float, float, float], ...] = ()
    # funding
    funding_const: float = 1.0e-4
    funding_path: tuple[float, ...] | None = None   # per-second, overrides
    funding_ou: tuple[float, float, float, float, float] | None = None

    @property
    def horizon_s(self) -> int:
        return self.epoch_s * self.n_epochs


def _window_mult(sec: np.ndarray,
                 windows: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    m = np.ones(sec.size)
    for lo, hi, mult in windows:
        m[(sec >= lo) & (sec < hi)] *= mult
    return m


def make_stress_day(spec: StressSpec, seed: int) -> tuple[DayArrays, dict]:
    """One synthetic funding day under `spec`. Returns (DayArrays, truth)
    where `truth` carries the generator-level series the strategies cannot
    see (true sigma path etc.) for the approximation-error metrics."""
    rng = np.random.default_rng(seed)
    t0 = 1_780_000_000_000
    n = spec.horizon_s * 10                       # 100 ms grid
    dt = 0.1
    tick_ts = t0 + 100 * np.arange(n, dtype=np.int64)
    sec_grid = np.arange(n) * dt

    # true sigma on the tick grid (stress windows applied multiplicatively)
    sigma_true = spec.sigma0 * _window_mult(sec_grid, spec.sigma_windows)

    # alpha_true: exact-discretisation OU on the tick grid
    phi = 0.5 ** (dt / spec.alpha_hl_s)
    innov_sd = spec.alpha_sd * math.sqrt(1.0 - phi * phi)
    alpha_true = np.empty(n)
    alpha_true[0] = rng.normal(0, spec.alpha_sd)
    eps = rng.normal(0, innov_sd, n)
    for i in range(1, n):
        alpha_true[i] = phi * alpha_true[i - 1] + eps[i]

    # mid path: drift + diffusion + jumps
    incr = alpha_true * dt + sigma_true * math.sqrt(dt) * rng.normal(0, 1, n)
    jump_at = np.zeros(n)
    for jt, js in zip(spec.jump_times_s, spec.jump_sigmas):
        i = min(int(jt / dt), n - 1)
        jump_at[i] += js * spec.sigma0           # sigma0 * sqrt(1 s) units
    mid = spec.s0 + np.cumsum(incr + jump_at)
    mid = np.round(mid / TICK) * TICK
    mid = np.maximum(mid, 100.0 * TICK)

    liq_mult = _window_mult(sec_grid, spec.liq_windows)
    L = BOOK_LEVELS
    bid_p = np.empty((n, L), dtype=np.float64)
    ask_p = np.empty((n, L), dtype=np.float64)
    for lv in range(L):
        bid_p[:, lv] = np.round((mid - HALF - TICK * lv) / TICK) * TICK
        ask_p[:, lv] = np.round((mid + HALF + TICK * lv) / TICK) * TICK
    bid_q = (rng.exponential(2.0, (n, L))
             * liq_mult[:, None] + 0.05).astype(np.float32)
    ask_q = (rng.exponential(2.0, (n, L))
             * liq_mult[:, None] + 0.05).astype(np.float32)

    lam = spec.trade_rate * dt * liq_mult
    counts = rng.poisson(np.minimum(lam, 20.0))
    idx = np.repeat(np.arange(n), counts)
    m = idx.size
    off_ms = rng.integers(0, 100, m)
    trade_ts = tick_ts[idx] + off_ms
    side = np.where(rng.random(m) < 0.5, 1, -1).astype(np.int8)
    depth = rng.exponential(1.0 / spec.k_true, m)
    px = np.where(side > 0, mid[idx] + HALF + depth, mid[idx] - HALF - depth)
    qty = rng.exponential(1.5, m) + 0.01


    bursts_ts, bursts_px, bursts_qty, bursts_side = [], [], [], []
    for jt, js in zip(spec.jump_times_s, spec.jump_sigmas):
        i = min(int(jt / dt), n - 1)
        sgn = -1 if js < 0 else 1                 # jump down -> sell cascade
        b = spec.burst_trades
        bt = tick_ts[i] + np.sort(rng.integers(0, 2_000, b))
        sweep = rng.uniform(0.0, abs(js) * spec.sigma0, b)
        bp = mid[i] + sgn * (HALF + sweep)
        bursts_ts.append(bt)
        bursts_px.append(bp)
        bursts_qty.append(rng.exponential(6.0, b) + 0.5)
        bursts_side.append(np.full(b, sgn, dtype=np.int8))
    if bursts_ts:
        trade_ts = np.concatenate([trade_ts, *bursts_ts])
        px = np.concatenate([px, *bursts_px])
        qty = np.concatenate([qty, *bursts_qty])
        side = np.concatenate([side, *bursts_side])
    order = np.argsort(trade_ts, kind="stable")
    trade_ts = trade_ts[order].astype(np.int64)
    px = np.round(px[order] / TICK) * TICK
    qty = qty[order]
    side = side[order]

    # 1 s exogenous series
    n1 = spec.horizon_s
    one_s = t0 + 1000 * np.arange(n1, dtype=np.int64)
    mid_1s = mid[::10][:n1]
    rng_a = np.random.default_rng(seed + 424_243)
    alpha_obs = alpha_true[::10][:n1] + rng_a.normal(0, spec.alpha_noise_sd,
                                                     n1)
    d = np.diff(mid_1s, prepend=mid_1s[0])
    c1 = np.cumsum(d)
    c2 = np.cumsum(d * d)
    W = 60
    var = np.empty(n1)
    for i in range(n1):
        j = max(i - W, 0)
        w = i - j
        if w < 5:
            var[i] = spec.sigma0 ** 2
        else:
            s1 = c1[i] - c1[j]
            s2 = c2[i] - c2[j]
            var[i] = max(s2 / w - (s1 / w) ** 2, 1e-12)
    sigma_hat = np.sqrt(var)

    if spec.funding_ou is not None:
        mu, hl_s, xi, cap, f0 = spec.funding_ou
        rng_f = np.random.default_rng(seed + 999_983)
        phi_f = 0.5 ** (1.0 / hl_s)
        f_path = np.empty(n1)
        f_path[0] = f0
        ef = rng_f.normal(0, xi * math.sqrt(max(1.0 - phi_f * phi_f, 1e-12)),
                          n1)
        for i in range(1, n1):
            f_path[i] = mu + phi_f * (f_path[i - 1] - mu) + ef[i]
        f_path = np.clip(f_path, -cap, cap)
    elif spec.funding_path is not None:
        f_path = np.asarray(spec.funding_path, dtype=np.float64)[:n1]
        if f_path.size < n1:
            f_path = np.pad(f_path, (0, n1 - f_path.size), mode="edge")
    else:
        f_path = np.full(n1, spec.funding_const)

    settle = t0 + 1000 * spec.epoch_s * np.arange(1, spec.n_epochs + 1,
                                                  dtype=np.int64)
    settle[-1] = min(int(settle[-1]), int(tick_ts[-1]))

    day = DayArrays(
        fday=f"{spec.name}#{seed}", tick_ts=tick_ts,
        valid=np.ones(n, dtype=bool),
        bid_p=bid_p, bid_q=bid_q, ask_p=ask_p, ask_q=ask_q,
        trade_ts=trade_ts, trade_px=px, trade_qty=qty, trade_side=side,
        alpha_ts=one_s, alpha_val=alpha_obs,
        alpha_ar_ts=np.empty(0, dtype=np.int64),
        alpha_ar_val=np.empty(0, dtype=np.float64),
        sigma_ts=one_s, sigma_val=sigma_hat,
        intensity_ts=one_s[::300],
        intensity_A=np.full(one_s[::300].size,
                            0.5 * spec.trade_rate
                            * math.exp(spec.k_true * HALF))
        * _window_mult(np.arange(0, n1, 300, dtype=float),
                       spec.liq_windows),
        intensity_k=np.full(one_s[::300].size, spec.k_true),
        funding_ts=one_s, funding_rate=f_path,
        mark_px=mid_1s,
        settle_ts=settle, pause_intervals=[],
    )
    truth = {"sigma_true_1s": sigma_true[::10][:n1], "f_path": f_path,
             "mid_1s": mid_1s, "mid_100ms": mid, "t0": t0}
    return day, truth


def path_metrics(spec: StressSpec, seed: int, res: dict, truth: dict,
                 cfg: SimConfig) -> list[dict]:
    rows = []
    sig_true = truth["sigma_true_1s"]
    t0 = truth["t0"]
    for sid, r in res.items():
        snaps = [s for s in r["snaps"] if not s[14]]
        mtm = np.asarray([s[7] for s in snaps])
        inv = np.asarray([s[3] for s in snaps])
        ts1 = (np.asarray([s[0] for s in snaps]) - t0) // 1000
        ts1 = np.clip(ts1, 0, sig_true.size - 1).astype(np.int64)
        run_max = np.maximum.accumulate(mtm) if mtm.size else np.array([0.0])
        max_dd = float((run_max - mtm).max()) if mtm.size else 0.0
        gamma_dollar = cfg.gamma_eff / cfg.quote_size
        cara = 0.5 * gamma_dollar * inv ** 2 * sig_true[ts1] ** 2
        half_quoted = np.asarray(
            [(s[9] + s[8]) / 2.0 for s in snaps
             if np.isfinite(s[8]) and np.isfinite(s[9])])
        dmid = np.abs(np.diff(truth["mid_1s"]))
        sref_ratio = (float(dmid.mean() / half_quoted.mean())
                      if half_quoted.size and half_quoted.mean() > 0
                      else float("nan"))
        mid100 = truth["mid_100ms"]
        mo_sum, mo_n = 0.0, 0
        for f in r["fills"]:
            if f[8]:                      # is_liquidation
                continue
            i5 = (f[0] + 5000 - t0) // 100
            if i5 < mid100.size:
                mo_sum += f[2] * (mid100[i5] - f[3]) / f[3] * 1e4
                mo_n += 1
        markout5 = mo_sum / mo_n if mo_n else float("nan")
        rows.append({
            "scenario": spec.name, "seed": seed, "strategy": sid,
            "terminal_pnl": r["terminal_pnl"], "n_fills": r["n_fills"],
            "fees": r["fees"], "funding": r["funding"],
            "mean_abs_inv": r["mean_abs_inv"],
            "frac_time_at_cap": r["frac_time_at_cap"],
            "max_drawdown": max_dd,
            "mean_signed_inv": float(inv.mean()) if inv.size else 0.0,
            "cara_corr_mean_usd": float(cara.mean()) if cara.size else 0.0,
            "cara_corr_p99_usd": float(np.quantile(cara, 0.99))
            if cara.size else 0.0,
            "sref_drift_ratio": sref_ratio,
            "markout5_bps": markout5,
            "epoch_s": spec.epoch_s, "n_epochs": spec.n_epochs,
        })
    return rows


def _run_one(args: tuple) -> list[dict]:
    spec, seed, cfg, sids, hook = args
    day, truth = make_stress_day(spec, seed)
    res = simulate_day(day, cfg, list(sids))
    rows = path_metrics(spec, seed, res, truth, cfg)
    if hook is not None:
        extra = hook(spec, seed, res, truth, cfg)  # {sid: {col: val}}
        for r in rows:
            r.update(extra.get(r["strategy"], {}))
    return rows


def run_scenarios(specs: list[StressSpec], n_paths: int, sids: tuple[int, ...],
                  cfg: SimConfig | None = None, processes: int = 1,
                  base_seed: int = 20_260_610,
                  hook=None) -> pl.DataFrame:
    """All (scenario, path) combinations -> tidy per-strategy metric frame."""
    cfg = cfg or SimConfig(lut_min_rebuild_s=60.0)
    jobs = [(spec, base_seed + p, cfg, sids, hook)
            for spec in specs for p in range(n_paths)]
    rows: list[dict] = []
    if processes > 1:
        with ProcessPoolExecutor(max_workers=processes) as ex:
            for out in ex.map(_run_one, jobs, chunksize=1):
                rows.extend(out)
    else:
        for j in jobs:
            rows.extend(_run_one(j))
    return pl.DataFrame(rows)


def cvar(x: np.ndarray, alpha: float = 0.05) -> float:
    """Expected shortfall: mean of the worst alpha-fraction of outcomes."""
    if x.size == 0:
        return float("nan")
    k = max(int(math.ceil(alpha * x.size)), 1)
    return float(np.sort(x)[:k].mean())


def tail_sharpe(x: np.ndarray, alpha: float = 0.05) -> float:
    """Sharpe restricted to the bottom alpha-fraction of paths (per-path,
    NOT annualised: synthetic days have no calendar)."""
    if x.size == 0:
        return float("nan")
    k = max(int(math.ceil(alpha * x.size)), 2)
    tail = np.sort(x)[:k]
    sd = tail.std(ddof=1)
    return float(tail.mean() / sd) if sd > 0 else float("nan")


def scenario_table(df: pl.DataFrame) -> pl.DataFrame:
    """Per (scenario, strategy) summary used by every stress module."""
    recs = []
    have_mo = "markout5_bps" in df.columns
    for (scen, sid), grp in sorted(
            df.group_by("scenario", "strategy"),
            key=lambda kv: (kv[0][0], kv[0][1])):
        x = grp["terminal_pnl"].to_numpy()
        fills = grp["n_fills"].to_numpy()
        mean_fills = float(fills.mean())
        k5 = max(int(math.ceil(0.05 * x.size)), 1)
        tail_idx = np.argsort(x)[:k5]
        recs.append({
            "scenario": scen, "strategy": sid, "n_paths": x.size,
            "mean_pnl": float(x.mean()),
            "sd_pnl": float(x.std(ddof=1)) if x.size > 1 else float("nan"),
            "cvar5": cvar(x), "tail_sharpe5": tail_sharpe(x),
            "worst": float(x.min()),
            "mean_max_dd": float(grp["max_drawdown"].mean()),
            "cap_freq": float((grp["frac_time_at_cap"] > 0).mean()),
            "mean_cap_time": float(grp["frac_time_at_cap"].mean()),
            "mean_funding": float(grp["funding"].mean()),
            "cara_p99": float(grp["cara_corr_p99_usd"].max()),
            "sref_ratio": float(grp["sref_drift_ratio"].mean()),
            "mean_n_fills": mean_fills,
            "pnl_per_fill": (float(x.mean() / mean_fills)
                             if mean_fills > 0 else float("nan")),
            "mean_markout5_bps": (float(grp["markout5_bps"].mean())
                                  if have_mo else float("nan")),
            "tail5_markout5_bps": (
                float(np.nanmean(grp["markout5_bps"].to_numpy()[tail_idx]))
                if have_mo else float("nan")),
        })
    return pl.DataFrame(recs)


def write_scenario_tex(tab: pl.DataFrame, path: Path, caption: str,
                       label: str, footer_lines: list[str] | None = None) -> None:
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{llrrrrrrr}", r"\toprule",
        r"Scenario & Strat. & Mean P\&L & ES$_{5\%}$ & "
        r"Cap (\%) & Funding & Fills & P\&L/fill & Markout (bp) \\",
        r"\midrule",
    ]
    for r in tab.to_dicts():
        mo = r.get("mean_markout5_bps", float("nan"))
        mo_s = f"{mo:+.2f}" if mo == mo else "--"
        lines.append(
            f"{r['scenario'].replace('_', ' ')} & s{r['strategy']} & "
            f"{r['mean_pnl']:,.0f} & {r['cvar5']:,.0f} & "
            f"{100 * r['mean_cap_time']:.2f} & "
            f"{r['mean_funding']:,.1f} & {r['mean_n_fills']:,.0f} & "
            f"{r['pnl_per_fill']:.3f} & {mo_s} \\\\")
    if footer_lines:
        lines.append(r"\midrule")
        lines += [rf"\multicolumn{{9}}{{@{{}}p{{\linewidth}}@{{}}}}{{{ln}}} \\"
                  for ln in footer_lines]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stress-suite shared engine.")
    ap.parse_args()
    print(json.dumps({"modules": ["stress_jump_diffusion",
                                  "stress_funding_ou",
                                  "stress_regime_shift",
                                  "analysis_monte_carlo"]}, indent=2))


if __name__ == "__main__":
    main()
