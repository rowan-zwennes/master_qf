"""Offline-replay backtest driver for the six-strategy ablation."""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import polars as pl

from data_gap_handler import load_pause_intervals, merge_intervals
from fill_core_reference import QueuePositionTracker
from hjb_lut_builder import build_and_write
from hjb_principal_eigenvector import HJBParams
from hjb_riccati_solver import FundingParams
from pipeline_utils import (
    funding_day_bounds,
    funding_day_paths,
    funding_settlement_times,
    load_splits,
)
from sim_quote_engine import (
    STRATEGIES,
    EmaDrift,
    MarketConsts,
    RegimeIIQuoter,
    RegimeIQuoter,
    StrategySpec,
    chi_effective_consts,
    chi_unscale,
    exact_drift_horizon,
    quote_depths,
)

BASE_DEFAULT = Path("/mnt/rowan_thuis/Crypto_recorder/Merged_new/btcusdt")
INTENSITY_DIR_DEFAULT = Path("reports/intensity_qa_rolling")
VOL_DIR_NAME = "volatility"


@dataclass
class SimConfig:
    maker_fee: float = 2.0e-4          # fraction of notional; negative = rebate
    taker_fee: float = 5.0e-4           # used for forced liquidation at day end
    quote_size: float = 0.005
    tick: float = 0.1                   # BTCUSDT perp price tick
    gamma: float = 2.0e-5               # CARA risk aversion (fixed), PER LOT
    Q: int = 10                         # inventory capacity in lots (fixed)
    rho: float = 6.46e-3
    funding_mode: str = "drain_normalized"
    funding_scale: float = 1.0
    delta_min: float = 0.0              # quote-depth floor (sensitivity hook)
    fixed_half_spread: float = 5.0      # strategy 1 naive half-spread (USDT)
    ema_half_life_s: float = 10.0
    # sensitivity scale hooks (±50% protocol, applied multiplicatively)
    gamma_scale: float = 1.0
    sigma_scale: float = 1.0
    k_scale: float = 1.0
    A_scale: float = 1.0
    rho_scale: float = 1.0
    sigma_col: str = "sigma"
    sigma_floor: float = 0.05           # USDT * s^-1/2
    alpha_max_stale_s: float = 2.0      # alpha_ML older than this -> 0
    post_gap_warmup_s: float = 60.0
    queue_init: str = "last"            # "last" | "front"
    order_latency_ms: int = 0
    requote_threshold_ticks: float = 0.0
    chi0: float = 0.0
    chi1: float = 0.0
    chi_fee_rate: float = 0.0
    ml_shift_mode: str = "horizon"      # "horizon" (headline) | "defensive"
    ml_signal_horizon_s: float = 10.0   # h for modes "horizon"/"defensive"
    ml_exact_drift_horizon: bool = True
    alpha_shuffle: bool = False
    lut_du_s: float = 0.1
    lut_ft_rel_trigger: float = 0.20
    lut_ft_abs_trigger: float = 2.0e-5
    lut_sigma_rel_trigger: float = 0.25
    lut_min_rebuild_s: float = 300.0
    lut_linear_drift: bool = True
    lut_q0_ref: float = 0.1            # finite-difference step for the sensitivity
    ml_bake_drift_in_lut: bool = False
    eps_ticks: float = 1.0              # u* tolerance, in ticks
    lut_u_max_floor_s: float = 1800.0   # min LUT horizon (covers observed u* ~1.4e3 s)
    lut_u_max_cap_s: float = 7200.0     # hard cap on the doubling search
    lut_export_dir: str | None = None
    export_regime1: bool = False

    @property
    def gamma_eff(self) -> float:
        return self.gamma * self.gamma_scale

    @property
    def rho_eff(self) -> float:
        return self.rho * self.rho_scale


@dataclass
class DayArrays:
    """Everything simulate_day needs, as plain numpy arrays."""

    fday: str
    # book, 100 ms grid
    tick_ts: np.ndarray          # (n,) i64 ms
    valid: np.ndarray            # (n,) bool
    bid_p: np.ndarray            # (n, L) f64, PRICES MUST BE f64: f32 moves
    bid_q: np.ndarray            # (n, L) f32     them off the 0.1 tick grid,
    ask_p: np.ndarray            # (n, L) f64     which silently breaks the
    ask_q: np.ndarray            # (n, L) f32     at-level fill match
    # trades, true event clock
    trade_ts: np.ndarray         # (m,) i64 ms
    trade_px: np.ndarray         # (m,) f64
    trade_qty: np.ndarray        # (m,) f64
    trade_side: np.ndarray       # (m,) i8, +1 taker buy, -1 taker sell
    # 1 s exogenous series (backward as-of joined inside prepare)
    alpha_ts: np.ndarray         # (a,) i64 ms
    alpha_val: np.ndarray        # (a,) f64 USDT/s
    alpha_ar_ts: np.ndarray      # (b,) i64 ms
    alpha_ar_val: np.ndarray     # (b,) f64 USDT/s
    sigma_ts: np.ndarray         # (s,) i64 ms
    sigma_val: np.ndarray        # (s,) f64
    intensity_ts: np.ndarray     # (k,) i64 ms
    intensity_A: np.ndarray      # (k,) f64
    intensity_k: np.ndarray      # (k,) f64
    funding_ts: np.ndarray       # (f,) i64 ms
    funding_rate: np.ndarray     # (f,) f64
    mark_px: np.ndarray          # (f,) f64
    settle_ts: np.ndarray        # (>=3,) i64 ms
    pause_intervals: list[tuple[int, int]]


def _asof_idx(src_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Backward as-of: index of the most recent src row at or before each
    query ts; -1 where none exists yet.
    # SIMULATION INVARIANT: 'right' - 1 == strictly backward-looking."""
    return np.searchsorted(src_ts, query_ts, side="right") - 1


@dataclass
class TickArrays:
    """Per-100ms-tick derived state, vectorised before the event loop."""

    mid: np.ndarray
    micro: np.ndarray
    alpha: np.ndarray            # alpha_ML;  0 where stale/absent
    alpha_ar: np.ndarray         # alpha_AR;  0 where stale/absent
    sigma: np.ndarray
    A: np.ndarray
    k: np.ndarray
    f_rate: np.ndarray
    u_s: np.ndarray              # seconds to next settlement
    next_settle: np.ndarray      # i64 ms
    snap_mask: np.ndarray        # bool: emit a 1 s snapshot at this tick


def prepare_tick_arrays(day: DayArrays, cfg: SimConfig) -> TickArrays:
    ts = day.tick_ts
    bid0 = day.bid_p[:, 0].astype(np.float64)
    ask0 = day.ask_p[:, 0].astype(np.float64)
    bq0 = day.bid_q[:, 0].astype(np.float64)
    aq0 = day.ask_q[:, 0].astype(np.float64)
    mid = 0.5 * (bid0 + ask0)
    denom = np.maximum(bq0 + aq0, 1e-12)
    micro = (bid0 * aq0 + ask0 * bq0) / denom

    def _gated_drift(src_ts: np.ndarray, src_val: np.ndarray) -> np.ndarray:
        out = np.zeros(ts.size)
        if src_ts.size:
            ai = _asof_idx(src_ts, ts)
            ok = ai >= 0
            age_ok = np.zeros(ts.size, dtype=bool)
            age_ok[ok] = (ts[ok] - src_ts[ai[ok]]) <= cfg.alpha_max_stale_s * 1000
            out[age_ok] = src_val[ai[age_ok]]
        return out

    alpha = _gated_drift(day.alpha_ts, day.alpha_val)
    alpha_ar = _gated_drift(day.alpha_ar_ts, day.alpha_ar_val)

    si = _asof_idx(day.sigma_ts, ts)
    sigma = np.where(si >= 0, day.sigma_val[np.maximum(si, 0)],
                     cfg.sigma_floor)
    sigma = np.maximum(sigma * cfg.sigma_scale, cfg.sigma_floor)

    ki = _asof_idx(day.intensity_ts, ts)
    A = np.where(ki >= 0, day.intensity_A[np.maximum(ki, 0)], np.nan)
    kk = np.where(ki >= 0, day.intensity_k[np.maximum(ki, 0)], np.nan)
    # before the first intensity window of the day, hold the day's first value
    if np.isnan(A).any() and day.intensity_A.size:
        A = np.where(np.isnan(A), day.intensity_A[0], A)
        kk = np.where(np.isnan(kk), day.intensity_k[0], kk)
    A = np.maximum(A * cfg.A_scale, 1e-3)
    kk = np.maximum(kk * cfg.k_scale, 1e-4)

    fi = _asof_idx(day.funding_ts, ts)
    f_rate = np.where(fi >= 0, day.funding_rate[np.maximum(fi, 0)], 0.0)

    # exact u from the canonical settlement calendar (no join needed)
    settle = np.asarray(day.settle_ts, dtype=np.int64)
    nxt = np.searchsorted(settle, ts, side="right")
    nxt = np.minimum(nxt, settle.size - 1)
    next_settle = settle[nxt]
    u_s = np.maximum((next_settle - ts) / 1000.0, 0.0)

    snap_mask = ts % 1000 == 0
    return TickArrays(mid, micro, alpha, alpha_ar, sigma, A, kk, f_rate, u_s,
                      next_settle, snap_mask)


@dataclass
class StratState:
    spec: StrategySpec
    tracker: QueuePositionTracker = field(default_factory=QueuePositionTracker)
    ema: EmaDrift | None = None
    inv: float = 0.0             # BTC, signed
    cash: float = 0.0            # USDT
    fees: float = 0.0            # cumulative fee cost (rebates negative)
    funding: float = 0.0         # cumulative discrete funding P&L (signed)
    bid_id: int = -1
    ask_id: int = -1
    bid_px: float = math.nan
    ask_px: float = math.nan
    bid_alpha_ml: float = 0.0
    ask_alpha_ml: float = 0.0
    # output accumulators
    fills: list = field(default_factory=list)
    snaps: list = field(default_factory=list)
    # regime II
    r2: RegimeIIQuoter | None = None
    # regime I exact f^0 quoter (drift baked); rebuilt on mc roll or <=1 Hz
    r1: RegimeIQuoter | None = None
    r1_mc_gen: int = -1
    last_r1_ts: int = -1
    lut_built_Ft: float = math.nan
    lut_built_sigma: float = math.nan
    lut_built_alpha: float = math.nan   # effective ML drift baked (ml_bake_drift_in_lut)
    lut_built_ts: int = -10**18
    n_lut_builds: int = 0
    r2_latched: bool = False

    def q_lots(self, quote_size: float, Q: int) -> int:
        q = int(round(self.inv / quote_size))
        return max(-Q, min(Q, q))


def _queue_ahead(price: float, prices: np.ndarray, sizes: np.ndarray,
                 side: int, tick: float) -> float:
    m = np.abs(prices - price) < (tick * 0.5)
    if m.any():
        return float(sizes[m][0])
    deepest = prices[-1]
    if (side > 0 and price < deepest and deepest > 0) or \
       (side < 0 and price > deepest and deepest > 0):
        return float(sizes[-1])
    return 0.0


REGIME1_MAGIC = 0x52314631  # 'R1F1' little-endian header tag


def write_regime1_stream(path: Path, ts: list, log_f0: list) -> int:
    """Serialize one Regime-I LOG-f^0 stream (the C++ replay feed)."""
    n_rec = len(ts)
    n_q = len(log_f0[0]) if n_rec else 0
    dt = np.dtype([("ts", "<i8"), ("log_f0", "<f4", (n_q,))])
    arr = np.empty(n_rec, dt)
    if n_rec:
        arr["ts"] = np.asarray(ts, "<i8")
        arr["log_f0"] = np.asarray(log_f0, "<f4")
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", REGIME1_MAGIC, n_q, n_rec, 0))
        arr.tofile(f)
    return n_rec


def write_hdrift_stream(path: Path, ts: list, vals: list) -> int:
    """Serialize the rolling drift-horizon scalar timeline (C++ replay feed)."""
    n_rec = len(ts)
    dt = np.dtype([("ts", "<i8"), ("v", "<f8")])
    arr = np.empty(n_rec, dt)
    if n_rec:
        arr["ts"] = np.asarray(ts, "<i8")
        arr["v"] = np.asarray(vals, "<f8")
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", REGIME1_MAGIC, 0, n_rec, 1))
        arr.tofile(f)
    return n_rec


def _round_down(x: float, tick: float) -> float:
    return math.floor(x / tick + 1e-9) * tick


def _round_up(x: float, tick: float) -> float:
    return math.ceil(x / tick - 1e-9) * tick


def simulate_day(
    day: DayArrays,
    cfg: SimConfig,
    strategy_ids: list[int],
) -> dict[int, dict]:
    """Single chronological pass over one funding day driving every strategy
    against the SAME market, the ablation differences are then strategy
    differences by construction, never replay differences."""
    ta = prepare_tick_arrays(day, cfg)
    n = day.tick_ts.size
    states = {sid: StratState(spec=STRATEGIES[sid]) for sid in strategy_ids}
    chi_on = (cfg.chi0 != 0.0) or (cfg.chi1 != 0.0) \
        or (cfg.chi_fee_rate != 0.0)
    if chi_on:
        bad = [s for s in strategy_ids
               if STRATEGIES[s].naive or STRATEGIES[s].funding
               or STRATEGIES[s].drift != "none"]
        if bad:
            raise ValueError(
                f"chi correction only supports the drift-free, funding-free "
                f"GLT strategy (s2); got strategies {bad}")
    for st in states.values():
        st.tracker.price_eps = cfg.tick * 0.499
        st.tracker.order_latency_ms = int(cfg.order_latency_ms)

    need_funding = [s for s in states.values() if s.spec.funding]
    if cfg.ml_bake_drift_in_lut:
        if [s.spec.sid for s in need_funding] != [6]:
            raise ValueError("ml_bake_drift_in_lut requires exactly strategy 6 "
                             "(run with --strategies 6)")
        if cfg.ml_shift_mode != "horizon":
            raise ValueError("ml_bake_drift_in_lut requires ml_shift_mode='horizon'")
        if cfg.lut_export_dir is not None:
            raise ValueError("ml_bake_drift_in_lut is in-memory only; no LUT export")
    pauses = day.pause_intervals
    pause_i = 0
    paused_prev = False
    warmup_ms = int(cfg.post_gap_warmup_s * 1000)
    warmup_until = -1  # ms; while t < warmup_until the desk is flat (no quoting)
    trade_cursor = 0
    settle_i = 0
    settle_list = list(day.settle_ts)
    # mark price / funding rate exactly at each settlement (as-of)
    settle_mark = []
    for s_ts in settle_list:
        j = int(np.searchsorted(day.funding_ts, s_ts, side="right")) - 1
        settle_mark.append(
            (float(day.mark_px[max(j, 0)]), float(day.funding_rate[max(j, 0)]))
            if j >= 0 else (math.nan, 0.0)
        )

    mc_key = None
    mc: MarketConsts | None = None
    mc_chi: MarketConsts | None = None
    chi0_eff = cfg.chi0                     # fee-aware: refreshed at each mc roll
    mc_gen = 0                              # bumped on every (sigma,A,k) roll
    r1_auto: RegimeIQuoter | None = None    # autonomous f^0 (alpha=0), shared
    r1_chi: RegimeIQuoter | None = None     # exact f^0 on chi-effective consts
    Q = cfg.Q
    qs = cfg.quote_size
    tick = cfg.tick
    downtime_ms = 0
    lut_export_dir = Path(cfg.lut_export_dir) if cfg.lut_export_dir else None
    if lut_export_dir is not None:
        lut_export_dir.mkdir(parents=True, exist_ok=True)
    lut_timeline: list[dict] = []
    # Regime-I exact-f^0 depth streams (opt-in C++ feed): one per drift type.
    export_r1 = bool(cfg.export_regime1 and lut_export_dir is not None)
    r1_rec: dict[str, dict[str, list]] = (
        {s: {"ts": [], "log_f0": []} for s in ("auto", "ar", "ml")}
        if export_r1 else {})

    def record_r1(stream: str, ts: int, qtr: RegimeIQuoter) -> None:
        rec = r1_rec[stream]
        if rec["ts"] and rec["ts"][-1] == ts:   # dedup (e.g. s4 and s6 share "ml")
            return
        rec["ts"].append(ts)
        rec["log_f0"].append(qtr.log_f0)

    hd_rec: dict[str, list] = {"ts": [], "v": []} if export_r1 else {}

    def record_hdrift(ts: int, v: float) -> None:
        if hd_rec["ts"] and hd_rec["ts"][-1] == ts:
            return
        hd_rec["ts"].append(ts)
        hd_rec["v"].append(v)

    cap_ticks = {sid: 0 for sid in strategy_ids}
    absq_sum = {sid: 0.0 for sid in strategy_ids}
    live_ticks = 0  # ticks contributing to absq_sum/cap_ticks (see quoting loop)

    def apply_settlement(idx: int) -> None:
        """# SIMULATION INVARIANT: the EXACT discrete funding charge
        -inv * markPrice * fundingRate at the settlement millisecond; the
        continuous drain never reaches this code path."""
        for st in states.values():
            st.r2_latched = False
        mark, fr = settle_mark[idx]
        if math.isnan(mark):
            return
        for st in states.values():
            fee = st.inv * mark * fr
            st.cash -= fee
            st.funding -= fee

    def cancel_all(st: StratState) -> None:
        st.tracker.cancel_all()
        st.bid_id = st.ask_id = -1
        st.bid_px = st.ask_px = math.nan

    for i in range(n):
        t = int(day.tick_ts[i])

        while trade_cursor < day.trade_ts.size and day.trade_ts[trade_cursor] < t:
            tt = int(day.trade_ts[trade_cursor])
            while settle_i < len(settle_list) and settle_list[settle_i] <= tt:
                apply_settlement(settle_i)
                settle_i += 1
            while pause_i < len(pauses) and tt >= pauses[pause_i][1]:
                pause_i += 1
            if pause_i < len(pauses) and pauses[pause_i][0] <= tt:
                for st in states.values():
                    if st.tracker.active_count():
                        cancel_all(st)
                trade_cursor += 1
                continue
            px = float(day.trade_px[trade_cursor])
            qty = float(day.trade_qty[trade_cursor])
            side = int(day.trade_side[trade_cursor])
            u_fill = ((settle_list[settle_i] - tt) / 1000.0
                      if settle_i < len(settle_list) else float(ta.u_s[i]))
            for sid, st in states.items():
                if st.tracker.active_count() == 0:
                    continue
                for f in st.tracker.on_trade(px, qty, side, tt):
                    notional = f.price * f.qty
                    fee = cfg.maker_fee * notional
                    st.cash += (-f.price * f.qty if f.side > 0 else f.price * f.qty)
                    st.cash -= fee
                    st.fees += fee
                    st.inv += f.side * f.qty
                    if f.side > 0:
                        if abs(f.price - st.bid_px) < tick * 0.5 and \
                           st.tracker.find(st.bid_id) is None:
                            st.bid_id = -1
                            st.bid_px = math.nan
                    else:
                        if abs(f.price - st.ask_px) < tick * 0.5 and \
                           st.tracker.find(st.ask_id) is None:
                            st.ask_id = -1
                            st.ask_px = math.nan
                    st.fills.append((f.ts_ms, sid, f.side, f.price, f.qty,
                                     fee, st.inv, u_fill, False,
                                     f.ts_ms - f.placed_ts_ms, f.swept,
                                     st.bid_alpha_ml if f.side > 0
                                     else st.ask_alpha_ml))
            trade_cursor += 1

        # settlement that falls between the last trade and this tick
        while settle_i < len(settle_list) and settle_list[settle_i] <= t:
            apply_settlement(settle_i)
            settle_i += 1

        while pause_i < len(pauses) and t >= pauses[pause_i][1]:
            pause_i += 1
        in_gap = (pause_i < len(pauses) and pauses[pause_i][0] <= t) \
            or (not bool(day.valid[i]))
        if in_gap:
            if not paused_prev:
                for st in states.values():
                    cancel_all(st)
            paused_prev = True
            downtime_ms += 100
            if ta.snap_mask[i]:
                for sid, st in states.items():
                    st.snaps.append((t, math.nan, math.nan, st.inv, st.cash,
                                     st.fees, st.funding, math.nan, math.nan,
                                     math.nan, math.nan, math.nan,
                                     float(ta.u_s[i]), 0, True,
                                     float(ta.f_rate[i])))
            continue
        if paused_prev:
            paused_prev = False
            warmup_until = t + warmup_ms

        mid = float(ta.mid[i])
        bid0 = float(day.bid_p[i, 0])
        ask0 = float(day.ask_p[i, 0])
        if not (bid0 > 0.0 and ask0 > bid0):
            continue

        key = (ta.sigma[i], ta.A[i], ta.k[i])
        if key != mc_key:
            mc = MarketConsts(cfg.gamma_eff, float(ta.sigma[i]),
                              float(ta.A[i]), float(ta.k[i]))
            chi0_eff = cfg.chi0 + cfg.chi_fee_rate * mid
            mc_chi = (chi_effective_consts(mc, chi0_eff, cfg.chi1)
                      if chi_on else None)
            mc_key = key
            mc_gen += 1
            r1_auto = RegimeIQuoter.build(mc.hjb_params(0.0, Q))
            r1_chi = (RegimeIQuoter.build(mc_chi.hjb_params(0.0, Q))
                      if chi_on else None)
            drift_horizon = (exact_drift_horizon(mc, Q)
                             if cfg.ml_exact_drift_horizon
                             else mc.inv_gs2 * mc.scale)
            if export_r1:
                record_r1("auto", t, r1_auto)
                record_hdrift(t, drift_horizon)

        if need_funding:
            ft = float(ta.f_rate[i])
            Ft = mid * ft
            st0 = states[need_funding[0].spec.sid]
            bake = cfg.ml_bake_drift_in_lut
            bake_alpha = 0.0
            bake_trigger = False
            if bake:
                raw_a = float(ta.alpha[i])
                if raw_a != 0.0:
                    sh = max(-mc.c1, min(mc.c1, raw_a * cfg.ml_signal_horizon_s))
                    bake_alpha = sh / drift_horizon
                in_layer = st0.r2 is not None and float(ta.u_s[i]) <= st0.r2.u_star_s
                bake_trigger = in_layer and not (
                    abs(bake_alpha - st0.lut_built_alpha) <= 1e-15)
            stale = (
                st0.r2 is None
                or abs(Ft - st0.lut_built_Ft)
                > max(cfg.lut_ft_rel_trigger * abs(st0.lut_built_Ft),
                      cfg.lut_ft_abs_trigger * mid)
                or abs(mc.sigma - st0.lut_built_sigma)
                > cfg.lut_sigma_rel_trigger * st0.lut_built_sigma
                or bake_trigger
            )
            min_gap_ms = (1000 if bake_trigger else cfg.lut_min_rebuild_s * 1000)
            if stale and (t - st0.lut_built_ts) >= min_gap_ms:
                p_lut = HJBParams(gamma=cfg.gamma_eff, sigma=mc.sigma,
                                  A=mc.A, k=mc.k,
                                  alpha_ml=(bake_alpha if bake else 0.0), Q=Q)
                fp = FundingParams(F_t=mid * ft, rho=cfg.rho_eff,
                                   mode=cfg.funding_mode)
                u_max = cfg.lut_u_max_floor_s
                q0_ref = (None if bake else
                          cfg.lut_q0_ref if cfg.lut_linear_drift and any(
                              s.spec.drift == "ml" for s in need_funding) else None)
                while True:
                    if lut_export_dir is not None:
                        fname = f"lut_{t}.hftl"
                        lpath = lut_export_dir / fname
                        hdr, _res, u_star = build_and_write(
                            lpath, p_lut, fp, f_t=ft, u_max=u_max,
                            du_ms=int(round(cfg.lut_du_s * 1000.0)),
                            eps_ticks=cfg.eps_ticks, tick=tick, q0_ref=q0_ref)
                        r2 = RegimeIIQuoter.from_file(lpath)
                    else:
                        r2 = RegimeIIQuoter.build(p_lut, fp, u_max=u_max,
                                                  du_s=cfg.lut_du_s,
                                                  eps_ticks=cfg.eps_ticks, tick=tick,
                                                  q0_ref=q0_ref)
                        u_star = r2.u_star_s
                    if (u_star >= u_max - cfg.lut_du_s
                            and u_max < cfg.lut_u_max_cap_s):
                        u_max = min(2.0 * u_max, cfg.lut_u_max_cap_s)
                        continue
                    break
                if lut_export_dir is not None:
                    lut_timeline.append({"activation_ts_ms": int(t),
                                         "file": fname,
                                         "u_star_s": float(u_star),
                                         "du_ms": int(hdr.du_ms),
                                         "n_q": int(hdr.n_q),
                                         "n_u": int(hdr.n_u)})
                for st in need_funding:
                    sst = states[st.spec.sid]
                    sst.r2 = r2
                    sst.lut_built_Ft = mid * ft
                    sst.lut_built_sigma = mc.sigma
                    sst.lut_built_alpha = bake_alpha
                    sst.lut_built_ts = t
                    sst.n_lut_builds += 1

        u = float(ta.u_s[i])
        ml_alpha = float(ta.alpha[i])
        in_warmup = t < warmup_until
        live_ticks += 1
        for sid, st in states.items():
            if in_warmup:
                if st.bid_id >= 0 or st.ask_id >= 0:
                    cancel_all(st)
                absq_sum[sid] += abs(st.inv)
                if abs(st.q_lots(qs, Q)) >= Q:
                    cap_ticks[sid] += 1
                if ta.snap_mask[i]:
                    st.snaps.append((t, mid, float(ta.micro[i]), st.inv, st.cash,
                                     st.fees, st.funding, st.cash + st.inv * mid,
                                     math.nan, math.nan, 0.0, mc.sigma, u, 0, False,
                                     float(ta.f_rate[i])))
                continue
            spec = st.spec
            alpha = (ta.alpha[i] if spec.drift == "ml"
                     else (ta.alpha_ar[i] if spec.drift == "ar" else 0.0))
            if alpha != 0.0 and cfg.ml_shift_mode in ("horizon", "defensive"):
                shift_usd = alpha * cfg.ml_signal_horizon_s
                shift_usd = max(-mc.c1, min(mc.c1, shift_usd))
                alpha = shift_usd / drift_horizon
            q_int = st.q_lots(qs, Q)
            if spec.funding and st.r2 is not None and u <= st.r2.u_star_s:
                st.r2_latched = True
            force_r2 = st.r2_latched
            if alpha == 0.0:
                r1_use = r1_auto
            else:
                if (st.r1 is None or st.r1_mc_gen != mc_gen
                        or t - st.last_r1_ts >= 1000):
                    st.r1 = RegimeIQuoter.build(mc.hjb_params(alpha, Q))
                    st.r1_mc_gen = mc_gen
                    st.last_r1_ts = t
                    if export_r1:
                        record_r1(spec.drift, t, st.r1)
                r1_use = st.r1
            if chi_on:
                db_t, da_t = quote_depths(spec, mc_chi, q_int, 0.0, u, None,
                                          fixed_half_spread=cfg.fixed_half_spread,
                                          r1=r1_chi)
                db = chi_unscale(db_t, chi0_eff, cfg.chi1)
                da = chi_unscale(da_t, chi0_eff, cfg.chi1)
            else:
                db, da = quote_depths(spec, mc, q_int, alpha, u, st.r2,
                                      fixed_half_spread=cfg.fixed_half_spread,
                                      r1=r1_use, force_r2=force_r2,
                                      drift_in_lut=cfg.ml_bake_drift_in_lut)
            if alpha != 0.0 and cfg.ml_shift_mode == "defensive":
                db0, da0 = quote_depths(spec, mc, q_int, 0.0, u, st.r2,
                                        fixed_half_spread=cfg.fixed_half_spread,
                                        r1=r1_auto, force_r2=force_r2)
                db, da = max(db, db0), max(da, da0)

            want_bid = (st.inv + qs <= Q * qs + 1e-9) and math.isfinite(db)
            want_ask = (st.inv - qs >= -Q * qs - 1e-9) and math.isfinite(da)
            bpx = apx = math.nan
            if want_bid:
                bpx = _round_down(mid - max(db, cfg.delta_min), tick)
                bpx = _round_down(min(bpx, ask0 - tick), tick)
                if bpx <= 0.0:
                    want_bid = False
            if want_ask:
                apx = _round_up(mid + max(da, cfg.delta_min), tick)
                apx = _round_up(max(apx, bid0 + tick), tick)

            req_tol = max(tick * 0.5, cfg.requote_threshold_ticks * tick)
            if st.bid_id >= 0 and (not want_bid
                                   or abs(bpx - st.bid_px) > req_tol):
                st.tracker.cancel(st.bid_id)
                st.bid_id = -1
                st.bid_px = math.nan
            if want_bid and st.bid_id < 0:
                qa = (0.0 if cfg.queue_init == "front" else
                      _queue_ahead(bpx, day.bid_p[i],
                                   day.bid_q[i].astype(np.float64), +1, tick))
                st.bid_id = st.tracker.place(+1, bpx, qs, qa, t)
                st.bid_px = bpx
                st.bid_alpha_ml = ml_alpha
            if st.ask_id >= 0 and (not want_ask
                                   or abs(apx - st.ask_px) > req_tol):
                st.tracker.cancel(st.ask_id)
                st.ask_id = -1
                st.ask_px = math.nan
            if want_ask and st.ask_id < 0:
                qa = (0.0 if cfg.queue_init == "front" else
                      _queue_ahead(apx, day.ask_p[i],
                                    day.ask_q[i].astype(np.float64), -1, tick))
                st.ask_id = st.tracker.place(-1, apx, qs, qa, t)
                st.ask_px = apx
                st.ask_alpha_ml = ml_alpha

            absq_sum[sid] += abs(st.inv)
            if abs(q_int) >= Q:
                cap_ticks[sid] += 1

            if ta.snap_mask[i]:
                regime = 2 if force_r2 else 1
                st.snaps.append((t, mid, float(ta.micro[i]), st.inv, st.cash,
                                 st.fees, st.funding,
                                 st.cash + st.inv * mid,
                                 db, da, alpha, mc.sigma, u, regime, False,
                                 float(ta.f_rate[i])))

    last = n - 1
    while last > 0 and not (day.bid_p[last, 0] > 0
                            and day.ask_p[last, 0] > day.bid_p[last, 0]
                            and bool(day.valid[last])):
        last -= 1
    bid0 = float(day.bid_p[last, 0])
    ask0 = float(day.ask_p[last, 0])
    t_end = int(day.tick_ts[last])
    results: dict[int, dict] = {}
    for sid, st in states.items():
        cancel_all(st)
        liq_fee = 0.0
        if abs(st.inv) > 1e-12 and bid0 > 0 and ask0 > bid0:
            px = bid0 if st.inv > 0 else ask0
            notional = abs(st.inv) * px
            st.cash += st.inv * px
            liq_fee = cfg.taker_fee * notional
            st.cash -= liq_fee
            st.fees += liq_fee
            st.fills.append((t_end, sid, -1 if st.inv > 0 else 1, px,
                             abs(st.inv), liq_fee, 0.0, 0.0, True, None, False,
                             0.0))
            st.inv = 0.0
        results[sid] = {
            "terminal_pnl": st.cash,
            "fees": st.fees,
            "funding": st.funding,
            "n_fills": len(st.fills),
            "mean_abs_inv": absq_sum[sid] / max(live_ticks, 1),
            "frac_time_at_cap": cap_ticks[sid] / max(live_ticks, 1),
            "downtime_s": downtime_ms / 1000.0,
            "n_lut_builds": st.n_lut_builds,
            "fills": st.fills,
            "snaps": st.snaps,
        }
    if lut_export_dir is not None:
        (lut_export_dir / "lut_timeline.json").write_text(
            json.dumps({"du_ms": int(round(cfg.lut_du_s * 1000.0)),
                        "n_builds": len(lut_timeline),
                        "luts": lut_timeline}, indent=2))
        (lut_export_dir / "lut_timeline.txt").write_text(
            "".join(f"{e['activation_ts_ms']} {e['file']}\n"
                    for e in lut_timeline))
    if export_r1:
        lines = [f"n_q {2 * Q + 1}"]
        streams: dict[str, dict] = {}
        for s, rec in r1_rec.items():
            if not rec["ts"]:
                continue
            fname = f"regime1_{s}_{day.fday}.r1f"
            n = write_regime1_stream(lut_export_dir / fname,
                                     rec["ts"], rec["log_f0"])
            lines.append(f"{s} {fname} {n}")
            streams[s] = {"file": fname, "n_rec": n}
        if hd_rec["ts"]:
            fname = f"regime1_hdrift_{day.fday}.r1f"
            n = write_hdrift_stream(lut_export_dir / fname,
                                    hd_rec["ts"], hd_rec["v"])
            lines.append(f"hdrift {fname} {n}")
            streams["hdrift"] = {"file": fname, "n_rec": n}
        (lut_export_dir / f"regime1_timeline_{day.fday}.txt").write_text(
            "\n".join(lines) + "\n")
        (lut_export_dir / f"regime1_timeline_{day.fday}.json").write_text(
            json.dumps({"fday": day.fday, "Q": Q, "n_q": 2 * Q + 1,
                        "stream_by_drift": {"none": "auto", "ar": "ar",
                                            "ml": "ml"}, "streams": streams},
                       indent=2))
    return results


MARKOUT_HORIZONS_S = (1, 5, 30)


def _markout_prices(fill_ts: np.ndarray, tick_ts: np.ndarray,
                    mid: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """mid at t_fill + tau, NaN when the target tick is missing or invalid."""
    out = {}
    j0 = np.searchsorted(tick_ts, fill_ts, side="right") - 1
    ok0 = j0 >= 0
    jj0 = np.maximum(j0, 0)
    ok0 &= valid[jj0]
    out["mid_0s"] = np.where(ok0, mid[jj0], np.nan)
    for tau in MARKOUT_HORIZONS_S:
        tgt = fill_ts + tau * 1000
        j = np.searchsorted(tick_ts, tgt, side="left")
        ok = j < tick_ts.size
        jj = np.minimum(j, tick_ts.size - 1)
        # the 100 ms grid is regular within segments; accept a <=200 ms snap
        ok &= (tick_ts[jj] - tgt) <= 200
        ok &= valid[jj]
        px = np.where(ok, mid[jj], np.nan)
        out[f"mid_{tau}s"] = px
    return out


BOOK_LEVELS = 20


def _read_hourly(base: Path, fday: str, stem: str,
                 columns: list[str] | None = None) -> pl.DataFrame:
    frames = []
    for d, h in funding_day_paths(base, fday):
        p = base / d / f"{stem}_{h:02d}h.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p, columns=columns))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def load_day_arrays(base: Path, fday: str, cfg: SimConfig,
                    alpha_parquet: Path | None,
                    intensity_dir: Path,
                    alpha_ar_parquet: Path | None = None) -> DayArrays:
    start_ms, end_ms = funding_day_bounds(fday)

    cols = (["ts_ms", "valid"]
            + [f"bid_p_{i}" for i in range(BOOK_LEVELS)]
            + [f"bid_q_{i}" for i in range(BOOK_LEVELS)]
            + [f"ask_p_{i}" for i in range(BOOK_LEVELS)]
            + [f"ask_q_{i}" for i in range(BOOK_LEVELS)])
    book = (_read_hourly(base, fday, "book20", cols)
            .sort("ts_ms")
            .filter((pl.col("ts_ms") >= start_ms) & (pl.col("ts_ms") < end_ms)))
    if book.is_empty():
        raise FileNotFoundError(f"no book20 data for funding day {fday}")
    tick_ts = book["ts_ms"].to_numpy()
    valid = book["valid"].fill_null(False).to_numpy()

    def lv(prefix: str, dtype) -> np.ndarray:
        arr = np.empty((book.height, BOOK_LEVELS), dtype=dtype)
        for i in range(BOOK_LEVELS):
            arr[:, i] = book[f"{prefix}_{i}"].fill_null(0.0).to_numpy()
        return arr

    # prices f64 (tick-grid exact), sizes f32 (only feed queue_ahead)
    bid_p, ask_p = lv("bid_p", np.float64), lv("ask_p", np.float64)
    bid_q, ask_q = lv("bid_q", np.float32), lv("ask_q", np.float32)

    tr = (_read_hourly(base, fday, "trades",
                       ["EventTime", "id", "Price", "Quantity", "MakerWasBuyer"])
          .drop_nulls()
          .sort(["EventTime", "id"])
          .filter((pl.col("EventTime") >= start_ms)
                  & (pl.col("EventTime") < end_ms)))
    trade_ts = tr["EventTime"].cast(pl.Int64).to_numpy()
    trade_px = tr["Price"].to_numpy().astype(np.float64)
    trade_qty = tr["Quantity"].to_numpy().astype(np.float64)
    # MakerWasBuyer=True -> the passive side bought -> the TAKER sold
    trade_side = np.where(tr["MakerWasBuyer"].to_numpy(), -1, 1).astype(np.int8)

    mk = (_read_hourly(base, fday, "markprice_ffill",
                       ["ts_ms", "MarkPrice", "FundingRate"])
          .sort("ts_ms")
          .filter((pl.col("ts_ms") >= start_ms) & (pl.col("ts_ms") < end_ms)))
    funding_ts = mk["ts_ms"].to_numpy()
    funding_rate = mk["FundingRate"].fill_null(0.0).to_numpy().astype(np.float64)
    if cfg.funding_scale != 1.0:
        funding_rate = funding_rate * cfg.funding_scale
    mark_px = mk["MarkPrice"].to_numpy().astype(np.float64)

    # sigma: per CALENDAR date files covering the funding day
    vol_frames = []
    for d in {d for d, _ in funding_day_paths(base, fday)}:
        p = base / VOL_DIR_NAME / f"volatility_{d}.parquet"
        if p.exists():
            vol_frames.append(pl.read_parquet(
                p, columns=["ts_ms", "sigma", cfg.sigma_col]
                if cfg.sigma_col != "sigma" else ["ts_ms", "sigma"]))
    if vol_frames:
        vol = (pl.concat(vol_frames).sort("ts_ms")
               .filter((pl.col("ts_ms") >= start_ms) & (pl.col("ts_ms") < end_ms)))
        sigma_ts = vol["ts_ms"].to_numpy()
        s_main = vol["sigma"].fill_null(0.0).to_numpy().astype(np.float64)
        if cfg.sigma_col != "sigma":
            s_rob = vol[cfg.sigma_col].fill_null(0.0).to_numpy().astype(np.float64)
            sigma_val = np.where(s_rob > 0.0, s_rob, s_main)
        else:
            sigma_val = s_main
    else:
        raise FileNotFoundError(
            f"no volatility parquet for {fday} under {base / VOL_DIR_NAME}")

    ip = intensity_dir / f"intensity_rolling_{fday}.parquet"
    if ip.exists():
        idf = pl.read_parquet(ip).filter(pl.col("valid")).sort("ts_ms")
        intensity_ts = idf["ts_ms"].to_numpy()
        intensity_A = idf["A"].to_numpy().astype(np.float64) / 2.0
        kt = idf["k_touch"].to_numpy().astype(np.float64)
        kf = idf["k"].to_numpy().astype(np.float64)
        # k_touch is the floor-free headline (calibrate_intensity finding #2)
        intensity_k = np.where(np.isfinite(kt) & (kt > 0), kt, kf)
    else:
        raise FileNotFoundError(f"no rolling intensity parquet: {ip}")

    def _load_drift_stream(path, *, placebo_stream: int | None) -> tuple:
        """Read a 1 Hz drift parquet (alpha_ML or alpha_AR; identical schema)
        and clip it to the day. `placebo_stream` selects the RNG substream for
        the --alpha-shuffle control: None keeps the historical scalar seed used
        by every alpha_ML placebo run to date (byte-exact reproducibility),
        any int selects an independent substream for a second signal."""
        if path is None:
            return np.empty(0, dtype=np.int64), np.empty(0)
        df = pl.read_parquet(path)
        c = "alpha_ml_usdt_s" if "alpha_ml_usdt_s" in df.columns else "y_pred"
        df = (df.select(["ts_ms", c]).drop_nulls().sort("ts_ms")
              .filter((pl.col("ts_ms") >= start_ms) & (pl.col("ts_ms") < end_ms)))
        ts_, val = df["ts_ms"].to_numpy(), df[c].to_numpy().astype(np.float64)
        if cfg.alpha_shuffle and val.size:
            # placebo: destroy timing, keep the marginal distribution
            digits = int(str(fday).replace("-", ""))
            rng = np.random.default_rng(
                digits if placebo_stream is None else [digits, placebo_stream]
            )
            val = rng.permutation(val)
        return ts_, val

    alpha_ts, alpha_val = _load_drift_stream(alpha_parquet, placebo_stream=None)
    alpha_ar_ts, alpha_ar_val = _load_drift_stream(
        alpha_ar_parquet, placebo_stream=1
    )

    cands = sorted({s for d, _ in funding_day_paths(base, fday)
                    for s in funding_settlement_times(d)})
    settles = [s for s in cands if start_ms <= s < end_ms]
    nxt = [s for s in cands if s >= end_ms]
    if nxt:
        settles.append(nxt[0])
    pause = load_pause_intervals(base, fday)

    return DayArrays(
        fday=fday, tick_ts=tick_ts, valid=valid,
        bid_p=bid_p, bid_q=bid_q, ask_p=ask_p, ask_q=ask_q,
        trade_ts=trade_ts, trade_px=trade_px, trade_qty=trade_qty,
        trade_side=trade_side,
        alpha_ts=alpha_ts, alpha_val=alpha_val,
        alpha_ar_ts=alpha_ar_ts, alpha_ar_val=alpha_ar_val,
        sigma_ts=sigma_ts, sigma_val=sigma_val,
        intensity_ts=intensity_ts, intensity_A=intensity_A,
        intensity_k=intensity_k,
        funding_ts=funding_ts, funding_rate=funding_rate, mark_px=mark_px,
        settle_ts=np.asarray(sorted(settles), dtype=np.int64),
        pause_intervals=merge_intervals(pause),
    )


FILL_SCHEMA = ["ts_ms", "strategy", "side", "price", "qty", "fee",
               "inv_after", "u_s", "is_liquidation", "order_age_ms", "swept",
               "alpha_ml"]
SNAP_SCHEMA = ["ts_ms", "mid", "micro", "inv", "cash", "fees_cum",
               "funding_cum", "mtm", "delta_b", "delta_a", "alpha_used",
               "sigma", "u_s", "regime", "paused",
               "f_rate"]


def persist_day(out_dir: Path, fday: str, day: DayArrays,
                results: dict[int, dict]) -> dict[int, dict]:
    mid = (0.5 * (day.bid_p[:, 0].astype(np.float64)
                  + day.ask_p[:, 0].astype(np.float64)))
    summaries = {}
    for sid, res in results.items():
        sdir = out_dir / f"s{sid}"
        sdir.mkdir(parents=True, exist_ok=True)
        fills = pl.DataFrame(
            [list(r) for r in res["fills"]], schema=FILL_SCHEMA, orient="row"
        ) if res["fills"] else pl.DataFrame(schema=FILL_SCHEMA)
        if fills.height:
            mo = _markout_prices(fills["ts_ms"].to_numpy().astype(np.int64),
                                 day.tick_ts, mid, day.valid)
            fills = fills.with_columns(
                [pl.Series(kk, vv) for kk, vv in mo.items()])
        fills.write_parquet(sdir / f"fills_{fday}.parquet")
        snaps = pl.DataFrame(
            [list(r) for r in res["snaps"]], schema=SNAP_SCHEMA, orient="row"
        ) if res["snaps"] else pl.DataFrame(schema=SNAP_SCHEMA)
        snaps.write_parquet(sdir / f"pnl_{fday}.parquet")
        summary = {k: v for k, v in res.items() if k not in ("fills", "snaps")}
        summary["funding_day"] = fday
        (sdir / f"summary_{fday}.json").write_text(json.dumps(summary, indent=2))
        summaries[sid] = summary
    return summaries


def run_one_day(args_tuple) -> tuple[str, dict]:
    (base, fday, cfg, sids, alpha_parquet, intensity_dir, out_dir,
     alpha_ar_parquet) = args_tuple
    t0 = time.time()
    day = load_day_arrays(base, fday, cfg, alpha_parquet, intensity_dir,
                          alpha_ar_parquet)
    results = simulate_day(day, cfg, sids)
    summaries = persist_day(out_dir, fday, day, results)
    return fday, {"elapsed_s": time.time() - t0, "summaries": summaries}


def _make_synthetic_day(seed: int = 7, minutes: float = 12.0) -> DayArrays:
    """A small but complete synthetic day: random-walk mid on the 100 ms grid,
    Poisson trades on the true clock, one funding settlement mid-window, one
    hard pause. Prices on the 0.1 tick grid."""
    rng = np.random.default_rng(seed)
    t0 = 1_770_000_000_000
    n = int(minutes * 600)
    tick_ts = t0 + 100 * np.arange(n, dtype=np.int64)
    mid = 100_000.0 + np.cumsum(rng.normal(0, 0.8, n))
    mid = np.round(mid / 0.1) * 0.1
    half = 0.3  # keeps every synthetic level on the 0.1 tick grid
    L = BOOK_LEVELS
    bid_p = np.empty((n, L), dtype=np.float64)
    ask_p = np.empty((n, L), dtype=np.float64)
    for lv_ in range(L):
        bid_p[:, lv_] = np.round((mid - half - 0.1 * lv_) / 0.1) * 0.1
        ask_p[:, lv_] = np.round((mid + half + 0.1 * lv_) / 0.1) * 0.1
    bid_q = rng.exponential(2.0, (n, L)).astype(np.float32) + 0.1
    ask_q = rng.exponential(2.0, (n, L)).astype(np.float32) + 0.1
    valid = np.ones(n, dtype=bool)

    m = int(minutes * 60 * 6)  # ~6 trades/s
    trade_ts = np.sort(rng.integers(t0, t0 + n * 100, m)).astype(np.int64)
    j = np.searchsorted(tick_ts, trade_ts, side="right") - 1
    j = np.maximum(j, 0)
    side = np.where(rng.random(m) < 0.5, 1, -1).astype(np.int8)
    depth = rng.exponential(1.2, m)
    px = np.where(side > 0, mid[j] + half + depth, mid[j] - half - depth)
    px = np.round(px / 0.1) * 0.1
    qty = rng.exponential(1.5, m) + 0.01

    one_s = t0 + 1000 * np.arange(int(minutes * 60), dtype=np.int64)
    settle = np.asarray([t0 + int(minutes * 60_000 // 2),
                         t0 + int(minutes * 60_000)], dtype=np.int64)
    pause_start = t0 + int(minutes * 60_000 * 0.7)
    pause = [(pause_start, pause_start + 20_000)]

    return DayArrays(
        fday="synthetic", tick_ts=tick_ts, valid=valid,
        bid_p=bid_p, bid_q=bid_q, ask_p=ask_p, ask_q=ask_q,
        trade_ts=trade_ts, trade_px=px, trade_qty=qty, trade_side=side,
        alpha_ts=one_s, alpha_val=rng.normal(0.0, 0.2, one_s.size),
        alpha_ar_ts=one_s, alpha_ar_val=rng.normal(0.0, 0.02, one_s.size),
        sigma_ts=one_s, sigma_val=np.full(one_s.size, 2.5),
        intensity_ts=one_s[::60], intensity_A=np.full(one_s[::60].size, 20.0),
        intensity_k=np.full(one_s[::60].size, 0.145),
        funding_ts=one_s, funding_rate=np.full(one_s.size, 1e-4),
        mark_px=np.interp(one_s, tick_ts, mid),
        settle_ts=settle, pause_intervals=pause,
    )


def _git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, cwd=Path(__file__).parent, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Six-strategy ablation backtest.")
    ap.add_argument("--base", type=Path, default=BASE_DEFAULT)
    ap.add_argument("--split", choices=["sim", "pre_analysis"])
    ap.add_argument("--days", type=str, help="comma-separated funding days")
    ap.add_argument("--strategies", type=str, default="1,2,3,4,5,6")
    ap.add_argument("--alpha-parquet", type=Path,
                    help="ml_predict output (required for strategies 4/6)")
    ap.add_argument("--alpha-ar-parquet", type=Path,
                    help="ml_predict.py --model ar output (strategy 3)")
    ap.add_argument("--intensity-dir", type=Path, default=INTENSITY_DIR_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/ablation"))
    ap.add_argument("--workers", type=int, default=2)
    # economics / hyperparameters
    ap.add_argument("--rho", type=float, default=SimConfig.rho)
    ap.add_argument("--funding-scale", type=float,
                    default=SimConfig.funding_scale,
                    help="multiply the recorded funding rate; 15.1 puts "
                         "mean|f| at 5.5 bp, 123.8 at 45 bp")
    ap.add_argument("--funding-mode", type=str, default=SimConfig.funding_mode,
                    choices=["drain", "drain_normalized", "terminal_jump"])
    ap.add_argument("--gamma", type=float, default=SimConfig.gamma)
    ap.add_argument("--Q", type=int, default=SimConfig.Q)
    ap.add_argument("--quote-size", type=float, default=SimConfig.quote_size)
    ap.add_argument("--maker-fee", type=float, default=SimConfig.maker_fee)
    ap.add_argument("--taker-fee", type=float, default=SimConfig.taker_fee)
    ap.add_argument("--delta-min", type=float, default=SimConfig.delta_min)
    ap.add_argument("--fixed-half-spread", type=float,
                    default=SimConfig.fixed_half_spread)
    ap.add_argument("--sigma-col", type=str, default=SimConfig.sigma_col)
    ap.add_argument("--post-gap-warmup-s", type=float,
                    default=SimConfig.post_gap_warmup_s,
                    help="flat warm-up after a data gap ends (0 disables)")
    ap.add_argument("--chi0", type=float, default=SimConfig.chi0,
                    help="affine per-fill toxicity intercept (USDT); "
                         "pre-analysis calibrated, s2-only")
    ap.add_argument("--chi1", type=float, default=SimConfig.chi1,
                    help="affine per-fill toxicity slope in [0,1); "
                         "pre-analysis calibrated, s2-only")
    ap.add_argument("--chi-fee-rate", type=float,
                    default=SimConfig.chi_fee_rate,
                    help="fee-aware GLT: add eps*S_ref to chi0 at each "
                         "market-constants roll; s2-only")
    ap.add_argument("--order-latency-ms", type=int,
                    default=SimConfig.order_latency_ms,
                    help="an order posted at tick t is fillable from t+L ms; "
                         "cancels stay instant")
    ap.add_argument("--requote-threshold-ticks", type=float,
                    default=SimConfig.requote_threshold_ticks,
                    help="keep a resting order while the new quote is within "
                         "this many ticks of it")
    ap.add_argument("--queue-init", type=str, default=SimConfig.queue_init,
                    choices=["last", "front"],
                    help="queue-position seed")
    ap.add_argument("--ml-shift-mode", type=str,
                    default=SimConfig.ml_shift_mode,
                    choices=["horizon", "defensive"],
                    help="drift coupling: horizon = clip(alpha*h, +-c1), "
                         "defensive = the same applied widen-only")
    ap.add_argument("--ml-signal-horizon", type=float,
                    default=SimConfig.ml_signal_horizon_s)
    ap.add_argument("--ml-exact-drift-horizon", action="store_true",
                    help="rescale alpha*h by the exact-f0 drift horizon "
                         "instead of the Gaussian h_eff")
    ap.add_argument("--ml-bake-drift-in-lut", action="store_true",
                    help="bake the effective ML drift into the Regime-II LUT "
                         "instead of the linear-response superposition; "
                         "needs --strategies 6 and horizon mode")
    ap.add_argument("--alpha-shuffle", action="store_true",
                    help="placebo control: permute alpha_ML in time within "
                         "each day")
    ap.add_argument("--lut-export-dir", type=Path, default=None,
                    help="serialize the Regime-II LUT timeline here for the "
                         "C++ replay")
    ap.add_argument("--save-regime1", action="store_true",
                    help="also serialize the Regime-I exact-f0 depth streams "
                         "(opt-in, ~GB)")
    # sensitivity scale hooks
    for name in ("gamma-scale", "sigma-scale", "k-scale", "A-scale",
                 "rho-scale"):
        ap.add_argument(f"--{name}", type=float, default=1.0)
    args = ap.parse_args()


    cfg = SimConfig(
        maker_fee=args.maker_fee, taker_fee=args.taker_fee,
        quote_size=args.quote_size, gamma=args.gamma, Q=args.Q,
        rho=args.rho, funding_mode=args.funding_mode,
        funding_scale=args.funding_scale,
        delta_min=args.delta_min,
        fixed_half_spread=args.fixed_half_spread,
        sigma_col=args.sigma_col,
        post_gap_warmup_s=args.post_gap_warmup_s,
        queue_init=args.queue_init,
        chi0=args.chi0, chi1=args.chi1, chi_fee_rate=args.chi_fee_rate,
        order_latency_ms=args.order_latency_ms,
        requote_threshold_ticks=args.requote_threshold_ticks,
        ml_shift_mode=args.ml_shift_mode,
        ml_signal_horizon_s=args.ml_signal_horizon,
        ml_exact_drift_horizon=args.ml_exact_drift_horizon,
        ml_bake_drift_in_lut=args.ml_bake_drift_in_lut,
        alpha_shuffle=args.alpha_shuffle,
        lut_export_dir=str(args.lut_export_dir) if args.lut_export_dir else None,
        export_regime1=args.save_regime1,
        gamma_scale=getattr(args, "gamma_scale"),
        sigma_scale=getattr(args, "sigma_scale"),
        k_scale=getattr(args, "k_scale"),
        A_scale=getattr(args, "A_scale"),
        rho_scale=getattr(args, "rho_scale"),
    )
    sids = [int(s) for s in args.strategies.split(",")]
    if args.alpha_parquet and "alpha_ar" in Path(args.alpha_parquet).name:
        raise SystemExit(
            f"--alpha-parquet points at an AR stream ({Path(args.alpha_parquet).name}). "
            "Pass it to --alpha-ar-parquet instead; the schemas are identical "
            "so nothing downstream would catch this.")
    if args.alpha_ar_parquet and "alpha_ml" in Path(args.alpha_ar_parquet).name:
        raise SystemExit(
            f"--alpha-ar-parquet points at an ML stream "
            f"({Path(args.alpha_ar_parquet).name}). Pass it to --alpha-parquet.")
    if any(STRATEGIES[s].drift == "ml" for s in sids) and not args.alpha_parquet:
        raise SystemExit("strategies 4/6 need --alpha-parquet (ml_predict output)")
    if any(STRATEGIES[s].drift == "ar" for s in sids) and not args.alpha_ar_parquet:
        raise SystemExit("strategy 3 needs --alpha-ar-parquet "
                         "(ml_predict.py --model ar output)")

    if args.days:
        days = args.days.split(",")
    elif args.split:
        days = load_splits(args.base)["splits"][args.split]
    else:
        raise SystemExit("provide --days or --split")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "git_sha": _git_sha(), "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": asdict(cfg), "strategies": sids, "days": days,
        "alpha_parquet": str(args.alpha_parquet) if args.alpha_parquet else None,
        "alpha_ar_parquet": (str(args.alpha_ar_parquet)
                             if args.alpha_ar_parquet else None),
        "intensity_dir": str(args.intensity_dir),
        "base": str(args.base),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    jobs = [(args.base, d, cfg, sids, args.alpha_parquet, args.intensity_dir,
             args.out_dir, args.alpha_ar_parquet) for d in days]
    t0 = time.time()
    if args.workers <= 1:
        for jb in jobs:
            fday, info = run_one_day(jb)
            print(f"{fday}: {info['elapsed_s']:.1f}s  "
                  f"pnl={ {s: round(v['terminal_pnl'], 2) for s, v in info['summaries'].items()} }",
                  flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one_day, jb): jb[1] for jb in jobs}
            for fut in as_completed(futs):
                fday, info = fut.result()
                print(f"{fday}: {info['elapsed_s']:.1f}s  "
                      f"pnl={ {s: round(v['terminal_pnl'], 2) for s, v in info['summaries'].items()} }",
                      flush=True)
    print(f"done: {len(days)} days in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
