"""Check that the C++ replay engine reproduces the Python simulator."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ENGINE = HERE / "cpp" / "build" / "hft_engine"


def _ensure_engine() -> None:
    if ENGINE.exists():
        return
    print("building hft_engine ...")
    subprocess.run(["cmake", "-S", str(HERE / "cpp"), "-B",
                    str(HERE / "cpp" / "build"),
                    "-DCMAKE_BUILD_TYPE=Release"], check=True,
                   capture_output=True)
    subprocess.run(["cmake", "--build", str(HERE / "cpp" / "build"),
                    "--target", "hft_engine", "-j", "8"], check=True,
                   capture_output=True)


def _run(args: list[str]) -> dict:
    out = subprocess.run([str(ENGINE), *args], check=True,
                         capture_output=True, text=True)
    return json.loads(out.stdout)


def _read_fills_csv(path: Path) -> dict[str, list[tuple]]:
    """Group fill rows by sid, preserving file order (= per-sid time order
    with the day-end liquidation fill last)."""
    out: dict[str, list[tuple]] = {}
    lines = path.read_text().strip().splitlines()[1:]  # drop header
    for ln in lines:
        sid, ts, side, price, qty, fee, inv, swept = ln.split(",")
        out.setdefault(sid, []).append(
            (int(ts), int(side), float(price), float(qty),
             float(fee), float(inv), int(swept)))
    return out


def _run_and_compare_kit(kit_dir: Path, expected: dict, *, tag: str,
                         what: str) -> None:
    """Run `hft_engine replay` over a parity kit and assert C++ == Python at
    both the day-summary and per-event-fill level. Shared by the synthetic and
    real-day checks (the kit layout is identical)."""
    cpp_fills = kit_dir / "cpp_fills.csv"
    r1_manifest = sorted((kit_dir / "luts").glob("regime1_timeline_*.txt"))
    assert r1_manifest, "parity kit did not emit a Regime-I manifest"
    got = _run(["replay", "--data", str(kit_dir / "day.hftr"),
                "--strategies", "1,2,3,4,5,6",
                "--lut-timeline", str(kit_dir / "luts"),
                "--regime1", str(r1_manifest[0]),
                "--fills-out", str(cpp_fills)])
    for sid, exp in expected.items():
        g = got[sid]
        assert g["n_fills"] == exp["n_fills"], (sid, g, exp)
        for key in ("terminal_pnl", "fees", "funding", "mean_abs_inv",
                    "frac_time_at_cap"):
            a, b = float(g[key]), float(exp[key])
            tol = max(1e-6, 1e-9 * abs(b))
            assert abs(a - b) <= tol, (sid, key, a, b)
    print(f"[{tag}] strategy 1-6 full-day parity ({what}), incl. funding LUT "
          "timeline + s6 drift companion (pnl/fees/funding/fills)  OK")
    print(f"    { {s: round(v['terminal_pnl'], 4) for s, v in expected.items()} }")

    py = _read_fills_csv(kit_dir / "expected_fills.csv")
    cpp = _read_fills_csv(cpp_fills)
    COLS = ("ts_ms", "side", "price", "qty", "fee", "inv_after", "swept")
    total = 0
    for sid in py:
        pf, cf = py[sid], cpp.get(sid, [])
        assert len(pf) == len(cf), (sid, "fill count", len(pf), len(cf))
        for j, (pr, cr) in enumerate(zip(pf, cf)):
            assert pr[0] == cr[0], (sid, j, "ts_ms", pr[0], cr[0])
            assert pr[1] == cr[1], (sid, j, "side", pr[1], cr[1])
            for c in range(2, 6):
                tol = max(1e-6, 1e-9 * abs(pr[c]))
                assert abs(pr[c] - cr[c]) <= tol, (sid, j, COLS[c], pr[c], cr[c])
            # swept is the authoritative fill mechanism, compare exactly
            assert pr[6] == cr[6], (sid, j, "swept", pr[6], cr[6])
            total += 1
    print(f"[{tag}b] per-event fill-stream parity ({total} fills, tol 1e-6)  OK")


def check_replay_parity(td: Path) -> None:
    from export_replay_binary import export_parity_kit
    from run_simulation import SimConfig
    cfg = SimConfig(lut_min_rebuild_s=60.0)
    expected = export_parity_kit(td / "kit", cfg)
    _run_and_compare_kit(td / "kit", expected, tag="1", what="synthetic day")


def check_real_day_parity(td: Path, fday: str, *, base: Path,
                          alpha_parquet: Path | None,
                          intensity_dir: Path,
                          alpha_ar_parquet: Path | None = None) -> None:
    """C++ == Python on a REAL recorded funding day (the first time the C++
    engine touches real L2 data). Same kit + comparison as the synthetic check,
    under the PRODUCTION SimConfig, so it validates the actual OOS quoting path,
    incl. the s6 linear-response drift companion. Heavier (full 24 h day)."""
    from export_replay_binary import export_parity_kit_real
    from run_simulation import SimConfig
    cfg = SimConfig()                       # production defaults, no throttle override
    kit = td / "kit_real"
    expected = export_parity_kit_real(kit, cfg, base=base, fday=fday,
                                      alpha_parquet=alpha_parquet,
                                      intensity_dir=intensity_dir,
                                      alpha_ar_parquet=alpha_ar_parquet)
    _run_and_compare_kit(kit, expected, tag="R", what=f"real day {fday}")


def check_lut_parity(td: Path) -> None:
    from hjb_lut_builder import build_and_write
    from hjb_principal_eigenvector import HJBParams
    from hjb_riccati_solver import FundingParams
    from sim_quote_engine import RegimeIIQuoter

    p = HJBParams(gamma=2e-5, sigma=2.5, A=20.0, k=0.145, alpha_ml=0.0, Q=10)
    fp = FundingParams(F_t=10.0, rho=7e-3, mode="drain_normalized")
    lut_path = td / "parity.lut"
    q0_ref = 0.1
    build_and_write(lut_path, p, fp, f_t=1e-4, u_max=1800.0, du_ms=100,
                    q0_ref=q0_ref)
    r2 = RegimeIIQuoter.from_file(lut_path)
    assert r2.db_sens is not None, "from_file did not load the .sens companion"

    qs = [-9, -3, 0, 4, 9]
    us = [0.0, 0.05, 1.0, 12.34, 59.9, 300.0, 913.7, 1799.9]  # on/between-grid
    q0_probe = 40.0
    got = _run(["lutprobe", "--lut", str(lut_path),
                "--q", ",".join(map(str, qs)),
                "--u", ",".join(f"{u:.6f}" for u in us),
                "--q0", f"{q0_probe:.6f}"])
    assert abs(got["u_star_s"] - r2.u_star_s) < 1e-9, (
        got["u_star_s"], r2.u_star_s)
    assert got["has_sens"] is True, "C++ did not load the .sens companion"
    worst = worst_drift = 0.0
    for pr in got["probes"]:
        db_py, da_py = r2.depths(int(pr["q"]), float(pr["u"]))
        ddb_py, dda_py = r2.depths_drift(int(pr["q"]), float(pr["u"]), q0_probe)
        checks = ((pr["db"], db_py), (pr["da"], da_py))
        drift_checks = ((pr["db_drift"], ddb_py), (pr["da_drift"], dda_py))
        for a, b in checks:
            if math.isnan(b):
                assert math.isnan(a), pr
                continue
            worst = max(worst, abs(a - b))
            assert abs(a - b) < 1e-4, (pr, b)
        for a, b in drift_checks:
            if math.isnan(b):
                assert math.isnan(a), pr
                continue
            worst_drift = max(worst_drift, abs(a - b))
            assert abs(a - b) < 1e-4, ("drift", pr, b)
    print(f"[2] LUT interpolation parity (max |diff| {worst:.2e}, "
          f"u*={got['u_star_s']:.2f}s)  OK")
    print(f"[2b] LUT linear-response drift parity at q0={q0_probe:g} "
          f"(max |diff| {worst_drift:.2e}, sens loaded C++ & Python)  OK")


def check_bench(td: Path) -> None:
    out = _run(["bench", "--out", str(td / "lat.hftb"),
                "--batches", "50", "--batch-size", "256"])
    assert out["stages"] >= 1 and (td / "lat.hftb").exists()
    from analysis_latency import read_hftb
    stages = read_hftb(td / "lat.hftb")
    assert "glt_closed_form" in stages
    med = float(np.median(stages["glt_closed_form"]))
    assert 0.0 < med < 10_000.0, med   # ns/op sanity band
    print(f"[3] bench smoke: glt_closed_form median {med:.1f} ns/op  OK")


def main() -> None:
    import argparse

    from run_simulation import BASE_DEFAULT, INTENSITY_DIR_DEFAULT
    ap = argparse.ArgumentParser(description="C++/Python replay parity checks.")
    ap.add_argument("--real-day", type=str,
                    help="also run full-day parity on this funding day; "
                         "needs --alpha-parquet")
    ap.add_argument("--base", type=Path, default=BASE_DEFAULT)
    ap.add_argument("--alpha-parquet", type=Path)
    ap.add_argument("--alpha-ar-parquet", type=Path,
                    help="strategy 3's AR(1) drift stream "
                         "(ml_predict.py --model ar)")
    ap.add_argument("--intensity-dir", type=Path, default=INTENSITY_DIR_DEFAULT)
    ap.add_argument("--skip-synthetic", action="store_true",
                    help="with --real-day, run ONLY the real-day check")
    args = ap.parse_args()

    _ensure_engine()
    with tempfile.TemporaryDirectory() as tds:
        td = Path(tds)
        if not (args.real_day and args.skip_synthetic):
            check_replay_parity(td)
            check_lut_parity(td)
            check_bench(td)
        if args.real_day:
            check_real_day_parity(td, args.real_day, base=args.base,
                                  alpha_parquet=args.alpha_parquet,
                                  intensity_dir=args.intensity_dir,
                                  alpha_ar_parquet=args.alpha_ar_parquet)
    print("\nALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
