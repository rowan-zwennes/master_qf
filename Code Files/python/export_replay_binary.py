"""Serialise one funding day into the binary replay format."""
from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np

from run_simulation import (
    BASE_DEFAULT,
    INTENSITY_DIR_DEFAULT,
    DayArrays,
    SimConfig,
    _make_synthetic_day,
    prepare_tick_arrays,
    simulate_day,
)

MAGIC = b"HFTR"
VERSION = 2
HEADER_SIZE = 128
_HEADER = struct.Struct("<4sIIIIIIIqddd")


def write_replay_binary(path: Path, day: DayArrays, cfg: SimConfig) -> dict:
    ta = prepare_tick_arrays(day, cfg)
    n = day.tick_ts.size
    m = day.trade_ts.size
    L = day.bid_p.shape[1]
    settle = np.asarray(day.settle_ts, dtype=np.int64)
    # settlement (mark, rate) pairs exactly as simulate_day computes them
    s_mark = np.empty(settle.size)
    s_rate = np.empty(settle.size)
    for i, s_ts in enumerate(settle):
        j = int(np.searchsorted(day.funding_ts, s_ts, side="right")) - 1
        if j >= 0:
            s_mark[i] = float(day.mark_px[j])
            s_rate[i] = float(day.funding_rate[j])
        else:
            s_mark[i] = math.nan
            s_rate[i] = 0.0
    pauses = np.asarray(
        [x for p in day.pause_intervals for x in p], dtype=np.int64)

    hdr = _HEADER.pack(MAGIC, VERSION, n, m, L, settle.size,
                       len(day.pause_intervals), 0,
                       int(day.tick_ts[0]), cfg.tick, cfg.quote_size, 0.0)
    hdr = hdr + b"\x00" * (HEADER_SIZE - len(hdr))

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(hdr)
        f.write(np.ascontiguousarray(day.tick_ts, dtype=np.int64).tobytes())
        f.write(np.ascontiguousarray(day.valid, dtype=np.uint8).tobytes())
        f.write(np.ascontiguousarray(day.bid_p, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(day.bid_q, dtype=np.float32).tobytes())
        f.write(np.ascontiguousarray(day.ask_p, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(day.ask_q, dtype=np.float32).tobytes())
        for arr in (ta.mid, ta.micro, ta.alpha, ta.alpha_ar, ta.sigma,
                    ta.A, ta.k, ta.f_rate, ta.u_s):
            f.write(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(day.trade_ts, dtype=np.int64).tobytes())
        f.write(np.ascontiguousarray(day.trade_px, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(day.trade_qty,
                                     dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(day.trade_side, dtype=np.int8).tobytes())
        f.write(settle.tobytes())
        f.write(np.ascontiguousarray(s_mark, dtype=np.float64).tobytes())
        f.write(np.ascontiguousarray(s_rate, dtype=np.float64).tobytes())
        f.write(pauses.tobytes())
    tmp.rename(path)
    return {"n_ticks": n, "n_trades": m, "n_levels": L,
            "n_settle": int(settle.size),
            "n_pause": len(day.pause_intervals),
            "bytes": path.stat().st_size}


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        buf = f.read(HEADER_SIZE)
    (magic, version, n, m, L, n_s, n_p, _pad,
     t0, tick, qs, _r) = _HEADER.unpack(buf[:_HEADER.size])
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"bad HFTR header {magic!r} v{version}")
    return {"n_ticks": n, "n_trades": m, "n_levels": L, "n_settle": n_s,
            "n_pause": n_p, "t0_ms": t0, "tick": tick, "quote_size": qs}


def _write_parity_kit(out_dir: Path, cfg: SimConfig, day,
                      sids=(1, 2, 3, 4, 5, 6)) -> dict:
    """Serialise one DayArrays into a full C++ parity kit: day.hftr, the intraday
    Regime-II LUT timeline (out_dir/luts, incl. the <lut>.sens drift companions
    for s6), the Regime-I feed, and the Python expected summaries + per-event
    fill stream. Shared by the synthetic (export_parity_kit) and real-day
    (export_parity_kit_real) callers, which differ only in how `day` is
    built."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = replace(cfg, lut_export_dir=str(out_dir / "luts"),
                  export_regime1=True)
    info = write_replay_binary(out_dir / "day.hftr", day, cfg)
    res = simulate_day(day, cfg, list(sids))
    expected = {
        str(sid): {
            "terminal_pnl": r["terminal_pnl"],
            "fees": r["fees"],
            "funding": r["funding"],
            "n_fills": r["n_fills"],
            "mean_abs_inv": r["mean_abs_inv"],
            "frac_time_at_cap": r["frac_time_at_cap"],
        } for sid, r in res.items()
    }
    fill_lines = ["sid,ts_ms,side,price,qty,fee,inv_after,swept"]
    for sid, r in res.items():
        for f in r["fills"]:
            fill_lines.append(
                f"{sid},{int(f[0])},{int(f[2])},{f[3]:.10f},{f[4]:.10f},"
                f"{f[5]:.10f},{f[6]:.10f},{int(f[10])}")
    (out_dir / "expected_fills.csv").write_text("\n".join(fill_lines) + "\n")
    (out_dir / "expected_summaries.json").write_text(
        json.dumps({"config": {"gamma": cfg.gamma, "Q": cfg.Q,
                               "quote_size": cfg.quote_size,
                               "maker_fee": cfg.maker_fee,
                               "taker_fee": cfg.taker_fee,
                               "tick": cfg.tick,
                               "delta_min": cfg.delta_min,
                               "fixed_half_spread": cfg.fixed_half_spread,
                               },
                    "expected": expected, "replay": info}, indent=2))
    return expected


def export_parity_kit(out_dir: Path, cfg: SimConfig,
                      sids=(1, 2, 3, 4, 5, 6)) -> dict:
    """Synthetic-day parity kit for validate_cpp_parity.py. Sigma is made
    time-varying so several Regime-II LUT rebuilds fire and the swap logic is
    exercised; a short min-rebuild lands several in the u<u* window."""
    out_dir.mkdir(parents=True, exist_ok=True)
    day = _make_synthetic_day(seed=7, minutes=12.0)
    ssig = 2.5 + 1.0 * np.sin(np.arange(day.sigma_ts.size) / 7.0)
    zalpha = day.alpha_val.copy()
    z0 = zalpha.size // 3
    zalpha[z0:z0 + 25] = 0.0
    zalpha_ar = day.alpha_ar_val.copy()
    z1 = 2 * zalpha_ar.size // 3
    zalpha_ar[z1:z1 + 25] = 0.0
    day = replace(day, sigma_val=ssig.astype(np.float64), alpha_val=zalpha,
                  alpha_ar_val=zalpha_ar)
    cfg = replace(cfg, lut_min_rebuild_s=min(cfg.lut_min_rebuild_s, 15.0))
    return _write_parity_kit(out_dir, cfg, day, sids)


def export_parity_kit_real(out_dir: Path, cfg: SimConfig, *, base: Path,
                           fday: str, alpha_parquet: Path | None,
                           intensity_dir: Path,
                           sids=(1, 2, 3, 4, 5, 6),
                           alpha_ar_parquet: Path | None = None) -> dict:
    """Real-day parity kit: identical to export_parity_kit but the DayArrays is
    the recorded funding day `fday` (load_day_arrays), under the PRODUCTION cfg
    (no throttle override), so the parity check runs on real L2 data. The
    linear-response drift companion (<lut>.sens) is written per rebuild and read
    by the C++ replay, so the s6 drift path is validated on real ticks too."""
    from run_simulation import load_day_arrays
    day = load_day_arrays(base, fday, cfg, alpha_parquet, intensity_dir,
                          alpha_ar_parquet)
    return _write_parity_kit(out_dir, cfg, day, sids)


def main() -> None:
    ap = argparse.ArgumentParser(description="HFTR replay-binary exporter.")
    ap.add_argument("--parity-kit", type=Path,
                    help="write synthetic day.hftr + expected_summaries.json")
    ap.add_argument("--fday", type=str, help="real funding day to export")
    ap.add_argument("--base", type=Path, default=BASE_DEFAULT)
    ap.add_argument("--alpha-parquet", type=Path)
    ap.add_argument("--alpha-ar-parquet", type=Path,
                    help="strategy 3's AR(1) drift stream "
                         "(ml_predict.py --model ar)")
    ap.add_argument("--intensity-dir", type=Path,
                    default=INTENSITY_DIR_DEFAULT)
    ap.add_argument("--out", type=Path, default=Path("day.hftr"))
    args = ap.parse_args()
    cfg = SimConfig(lut_min_rebuild_s=60.0)
    if args.parity_kit:
        expected = export_parity_kit(args.parity_kit, cfg)
        print(json.dumps(expected, indent=2))
        return
    if not args.fday:
        raise SystemExit("provide --fday or --parity-kit")
    from run_simulation import load_day_arrays
    day = load_day_arrays(args.base, args.fday, cfg, args.alpha_parquet,
                          args.intensity_dir, args.alpha_ar_parquet)
    info = write_replay_binary(args.out, day, cfg)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
