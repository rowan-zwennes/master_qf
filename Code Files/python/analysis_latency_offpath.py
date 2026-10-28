"""Latency of the off-hot-path solvers."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded

from hjb_principal_eigenvector import (
    HJBParams,
    principal_eigvec,
    q_grid,
)
from hjb_riccati_solver import (
    FundingParams,
    solve_backward_reference,
    solve_backward_thomas,
)
from sim_quote_engine import MarketConsts, RegimeIIQuoter

HERE = Path(__file__).parent
ENGINE = HERE / "cpp" / "build" / "hft_engine"

# Production calibration (run_simulation SimConfig / queue-aware grid-30).
GAMMA = 2.0e-5
SIGMA = 4.575
A_INT = 0.2742
K_INT = 0.0900
Q = 10
TICK = 0.1
S_REF = 100_000.0          # BTCUSDT-scale anchor; gamma*S_ref ~ 2 (O(1))
F_T = 10.0                 # S_ref * f_t funding scalar (f_t ~ 1e-4)
RHO = 7.0e-3               # spectral-gap layer rate (project_funding_drain_normalization)
U_MAX = 60.0               # boundary-layer horizon (s)
DU_S = 0.1                 # 100 ms grid (matches Delta t_LUT)
EPOCH_S = 8 * 3600.0       # inter-settlement duration


def pde_backward_solve(
    *, gamma: float = GAMMA, sigma: float = SIGMA, A: float = A_INT,
    k: float = K_INT, Q: int = Q, s_ref: float = S_REF, F_t: float = 0.0,
    rho: float = RHO, u_max: float = U_MAX, du_s: float = DU_S,
    n_s: int = 512, width_sigmas: float = 6.0,
) -> dict:
    """Crank-Nicolson backward solve of the un-reduced HJB on (u, S)."""
    p = HJBParams(gamma=gamma, sigma=sigma, A=A, k=k, alpha_ml=0.0, Q=Q)
    mc = MarketConsts(gamma, sigma, A, k)
    n_q = 2 * Q + 1
    qs = q_grid(Q).astype(float)                      # Q .. -Q
    c1 = mc.c1
    rho_tilde = rho / (1.0 - math.exp(-rho * EPOCH_S)) if rho > 0 else 0.0

    # price grid centred at s_ref, +- width_sigmas * sigma * sqrt(u_max)
    half = width_sigmas * sigma * math.sqrt(max(u_max, du_s))
    zeta = np.linspace(-half, half, n_s)
    dz = float(zeta[1] - zeta[0])
    j0 = int(np.argmin(np.abs(zeta)))                 # S = s_ref node

    # terminal Psi_q(0, zeta) = e^{-gamma q zeta} (f^0_q)^{-gamma/k}
    f0 = principal_eigvec(p)                          # q = Q..-Q, positive
    base = np.power(f0, -gamma / k)                    # (f^0_q)^{-gamma/k}
    Psi = (base[:, None] * np.exp(-gamma * qs[:, None] * zeta[None, :]))

    a_diff = 0.5 * sigma * sigma / (dz * dz)           # 2nd-deriv coefficient
    # tridiagonal of (I - du/2 D) and (I + du/2 D); Neumann (d_zz=0) at edges
    lower = np.full(n_s, -0.5 * du_s * a_diff)
    diag = np.full(n_s, 1.0 + du_s * a_diff)
    upper = np.full(n_s, -0.5 * du_s * a_diff)
    lower[-1] = upper[0] = 0.0
    diag_edge = 1.0
    diL = diag.copy(); diL[0] = diL[-1] = diag_edge
    ab = np.zeros((3, n_s))                            # banded (I - du/2 D)
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diL
    ab[2, :-1] = lower[1:]

    inv_g = 1.0 / gamma
    inv_k = 1.0 / k
    floor = 1e-300
    n_steps = int(round(u_max / du_s))
    for s in range(n_steps):
        u_next = (s + 1) * du_s
        lnPsi = np.log(np.maximum(Psi, floor))

        react = np.zeros_like(Psi)
        gam_kg = gamma / (k + gamma)

        for iq in range(n_q):
            coeff = 0.0

            if iq - 1 >= 0:        # q+1 exists -> bid fills (not at q=+Q)
                db = c1 + zeta + inv_g * (lnPsi[iq - 1] - lnPsi[iq])

                coeff = coeff - A * gam_kg * np.exp(-k * db)

            if iq + 1 < n_q:       # q-1 exists -> ask fills (not at q=-Q)
                da = c1 - zeta + inv_g * (lnPsi[iq + 1] - lnPsi[iq])

                # Plugging optimal da* back yields the symmetric reaction coefficient
                coeff = coeff - A * gam_kg * np.exp(-k * da)

            phi = k * qs[iq] * F_t * rho_tilde * math.exp(-rho * u_next)
            coeff = coeff + (gamma / k) * phi
            react[iq] = coeff

        rhs = np.empty_like(Psi)
        for iq in range(n_q):
            row = Psi[iq]
            expl = row.copy()

            # Explicit diffusion: (du/2)*D*Psi^n
            expl[1:-1] += 0.5 * du_s * a_diff * (row[2:] - 2 * row[1:-1] + row[:-2])

            # Explicit reaction: du * react * Psi^n
            expl += du_s * react[iq] * row
            rhs[iq] = expl

        for iq in range(n_q):
            Psi[iq] = solve_banded((1, 1), ab, rhs[iq])

        # renormalise per q-slice to keep O(1) (cancels in quote log-ratios)
        Psi /= np.max(np.abs(Psi))

    # depths at S = s_ref (zeta=0): same closed form on the converged Psi
    lnPsi0 = np.log(np.maximum(Psi[:, j0], floor))
    db = np.full(n_q, np.nan)
    da = np.full(n_q, np.nan)
    for iq in range(n_q):
        if iq - 1 >= 0:
            db[iq] = c1 + inv_g * (lnPsi0[iq - 1] - lnPsi0[iq])
        if iq + 1 < n_q:
            da[iq] = c1 + inv_g * (lnPsi0[iq + 1] - lnPsi0[iq])
    # reorder to ascending q+Q (match discrete_autonomous_quotes)
    return {"db": db[::-1], "da": da[::-1], "n_s": n_s, "n_q": n_q,
            "n_steps": n_steps, "dz": dz}


def _time(fn, reps: int) -> dict:
    """Wall-clock fn() `reps` times; return p50/p99/min in milliseconds."""
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    a = np.asarray(samples)
    return {"p50_ms": float(np.median(a)), "p99_ms": float(np.percentile(a, 99)),
            "min_ms": float(a.min()), "reps": reps}


def measure_eigensolve(reps: int = 500) -> dict:
    p = HJBParams(gamma=GAMMA, sigma=SIGMA, A=A_INT, k=K_INT, alpha_ml=0.0, Q=Q)
    return _time(lambda: principal_eigvec(p, check=False), reps)


def measure_ode_backsolve(reps: int = 100) -> dict:
    """PURE Regime-II backward solve (Radau reference), apples-to-apples with the
    PDE solve. The full LUT rebuild (this + u* search + quote-grid precompute)
    and the live Thomas-mirror solve are reported separately as context."""
    p = HJBParams(gamma=GAMMA, sigma=SIGMA, A=A_INT, k=K_INT, alpha_ml=0.0, Q=Q)
    fp = FundingParams(F_t=F_T, rho=RHO, mode="drain_normalized")
    u_eval = np.arange(0.0, U_MAX + 1e-9, DU_S)
    return _time(lambda: solve_backward_reference(p, fp, U_MAX, u_eval), reps)


def measure_lut_rebuild_full(reps: int = 100) -> dict:
    """Full Regime-II LUT rebuild = the off-path trigger cost (Radau solve +
    u* search + quote-grid precompute), the number to quote as 'rebuild time'."""
    p = HJBParams(gamma=GAMMA, sigma=SIGMA, A=A_INT, k=K_INT, alpha_ml=0.0, Q=Q)
    fp = FundingParams(F_t=F_T, rho=RHO, mode="drain_normalized")
    return _time(lambda: RegimeIIQuoter.build(
        p, fp, u_max=U_MAX, du_s=DU_S, eps_ticks=1.0, tick=TICK), reps)


def measure_thomas_mirror(reps: int = 100) -> dict:
    """The fixed-step theta/Thomas backward solve the C++ engine mirrors (what a
    LIVE system would rebuild with, vs the offline Radau reference)."""
    p = HJBParams(gamma=GAMMA, sigma=SIGMA, A=A_INT, k=K_INT, alpha_ml=0.0, Q=Q)
    fp = FundingParams(F_t=F_T, rho=RHO, mode="drain_normalized")
    return _time(lambda: solve_backward_thomas(p, fp, U_MAX, DU_S, theta=0.5),
                 reps)


def measure_pde_backsolve(n_s: int = 512, reps: int = 15) -> dict:
    r = _time(lambda: pde_backward_solve(F_t=F_T, n_s=n_s), reps)
    r["n_s"] = n_s
    return r


def read_regime1_stream(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(ts[n_rec], log_f0[n_rec, n_q] f32) from a .r1f stream, i.e. the exact
    bytes Regime1Reader loads (run_simulation.write_regime1_stream)."""
    import struct
    with open(path, "rb") as f:
        magic, n_q, n_rec, _kind = struct.unpack("<IIII", f.read(16))
        if magic != 0x52314631:
            raise ValueError(f"{path}: bad R1F1 magic {magic:#x}")
        dt = np.dtype([("ts", "<i8"), ("log_f0", "<f4", (n_q,))])
        arr = np.fromfile(f, dt, count=n_rec)
    return arr["ts"].copy(), arr["log_f0"].copy()


def build_bench_kit(root: Path) -> dict:
    """Export the synthetic parity kit and return the artifact paths the C++
    bench consumes: the FIRST LUT by name and the `auto` Regime-I stream (the
    stream bench_regime1_read queries). Both benches read these same files."""
    from export_replay_binary import export_parity_kit
    from run_simulation import SimConfig
    kit = root / "kit"
    export_parity_kit(kit, SimConfig(lut_min_rebuild_s=60.0))
    luts = kit / "luts"
    man = sorted(luts.glob("regime1_timeline_*.txt"))[0]
    lut = sorted(luts.glob("lut_*.hftl"))[0]
    r1 = None
    for line in man.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "auto":
            r1 = luts / parts[1]
            break
    if r1 is None or not r1.exists():
        raise FileNotFoundError(f"auto Regime-I stream missing in {man}")
    return {"lut": lut, "r1": r1, "manifest": man}


def measure_python_hotpath(kit: dict, batches: int = 3_000,
                           calls: int = 512) -> dict:
    from sim_quote_engine import RegimeIIQuoter

    r2 = RegimeIIQuoter.from_file(kit["lut"])
    ts_rec, lf_tab = read_regime1_stream(kit["r1"])
    n_rec, n_q = lf_tab.shape
    Q_ = r2.Q
    mc = MarketConsts(GAMMA, SIGMA, A_INT, K_INT)
    c1, inv_k = mc.c1, 1.0 / mc.k
    nan = float("nan")
    u_max_lut = (r2.n_u - 1) * r2.du_s
    cursor = 0

    qs = np.random.default_rng(0).integers(-Q_, Q_ + 1, size=calls)
    us = np.random.default_rng(1).uniform(0.0, u_max_lut, size=calls)
    tq = np.sort(np.random.default_rng(2).integers(
        int(ts_rec[0]), int(ts_rec[-1]) + 1, size=calls).astype(np.int64))

    def regime1_read(i: int):
        """Twin of Regime1Reader::depths: forward-cursor as-of + recompute."""
        nonlocal cursor
        t = tq[i]
        if ts_rec[cursor] <= t:
            while cursor + 1 < n_rec and ts_rec[cursor + 1] <= t:
                cursor += 1
        else:
            cursor = max(int(np.searchsorted(ts_rec, t, side="right")) - 1, 0)
        lf = lf_tab[cursor]
        j = Q_ - int(qs[i])                     # q = Q..-Q row index
        bid = c1 + inv_k * (float(lf[j]) - float(lf[j - 1])) if j >= 1 else nan
        ask = (c1 + inv_k * (float(lf[j]) - float(lf[j + 1]))
               if j + 1 < n_q else nan)
        return bid, ask

    out = {}
    for name, fn in (("regime1_read", regime1_read),
                     ("lut_interp",
                      lambda i: r2.depths(int(qs[i]), float(us[i])))):
        per_call_ns = []
        for _ in range(batches):
            cursor = 0                          # replay restarts each batch
            t0 = time.perf_counter()
            for i in range(calls):
                fn(i)
            per_call_ns.append((time.perf_counter() - t0) * 1e9 / calls)
        a = np.asarray(per_call_ns)
        out[name] = {"p50_ns": float(np.median(a)),
                     "p99_ns": float(np.percentile(a, 99)),
                     "batches": batches, "calls_per_batch": calls}
    out["inputs"] = {"lut": kit["lut"].name, "r1": kit["r1"].name,
                     "n_u": int(r2.n_u), "n_q": int(n_q), "n_rec": int(n_rec),
                     "u_max_lut_s": float(u_max_lut)}
    return out


def hotpath_from_hftb(path: Path) -> dict:
    from analysis_latency import read_hftb
    st = read_hftb(path)
    out = {}
    for name in ("regime1_read", "lut_interp", "glt_closed_form"):
        if name in st:
            v = np.asarray(st[name])
            out[name] = {"p50_ns": float(np.median(v)),
                         "p99_ns": float(np.percentile(v, 99))}
    return out


def try_run_engine_bench(kit: dict, workdir: Path) -> dict:
    """Run `hft_engine bench` on the SAME kit artifacts measure_python_hotpath
    reads, return hot-path ns. Empty dict if the engine binary is missing."""
    if not ENGINE.exists():
        return {}
    out = workdir / "lat.hftb"
    subprocess.run([str(ENGINE), "bench", "--lut", str(kit["lut"]),
                    "--regime1", str(kit["manifest"]), "--out", str(out),
                    "--batches", "3000", "--batch-size", "512"],
                   check=True, capture_output=True)
    return hotpath_from_hftb(out)


def build_report(hot: dict, eig: dict, ode: dict, pde: dict,
                 lut_full: dict, thomas: dict, py_hot: dict | None = None) -> dict:
    # cadence headroom: rebuild time vs how often it is triggered
    eig_period_ms = 1000.0                 # <= 1 Hz rebuild
    rebuild_period_ms = 1000.0             # f_t/sigma trigger; >= 1 s in practice
    return {
        "calibration": {"gamma": GAMMA, "sigma": SIGMA, "A": A_INT, "k": K_INT,
                        "Q": Q, "u_max_s": U_MAX, "du_s": DU_S, "S_ref": S_REF,
                        "F_t": F_T, "rho": RHO},
        "statistic": {
            "hot_path": "p50/p99 ACROSS 512-op batch MEANS (3000 batches), "
                        "both engines; NOT per-quote tail latency",
            "off_path": "p50/p99 across whole-solve repetitions (per-solve)",
        },
        "hot_path_ns": hot,
        "python_hot_path_ns": py_hot or {},
        "eigensolve": eig,
        "ode_backsolve": ode,                 # PURE Radau solve (PDE-comparable)
        "pde_backsolve": pde,                 # PURE PDE solve
        "lut_rebuild_full": lut_full,         # solve + u* + quote grid (trigger cost)
        "thomas_mirror": thomas,              # live engine-mirror solve
        "ratios": {
            "pde_over_ode_solve": pde["p50_ms"] / ode["p50_ms"] if ode["p50_ms"] else None,
            "rebuild_headroom_x": rebuild_period_ms / lut_full["p50_ms"] if lut_full["p50_ms"] else None,
            "eig_headroom_x": eig_period_ms / eig["p50_ms"] if eig["p50_ms"] else None,
        },
    }


def write_tex(r: dict, path: Path) -> None:
    hot = r["hot_path_ns"]
    def hp(name):
        if name not in hot:
            return "n/a & n/a"
        return f"{hot[name]['p50_ns']:.1f} ns & {hot[name]['p99_ns']:.1f} ns"
    lines = [
        "% Auto-generated by analysis_latency_offpath.py, do not edit.",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Stage & Where / cadence & p50 & p99 \\\\",
        "\\midrule",
        f"Regime-I f$^0$ read+compute$^\\dagger$ & C++ hot path / per quote & "
        f"{hp('regime1_read')} \\\\",
        f"Regime-II LUT interpolation$^\\dagger$ & C++ hot path / per quote & "
        f"{hp('lut_interp')} \\\\",
        "\\midrule",
    ]
    py = r.get("python_hot_path_ns") or {}
    def pyp(name):
        if name not in py:
            return None
        v = py[name]
        return (f"{v['p50_ns']/1e3:.2f} $\\mu$s & {v['p99_ns']/1e3:.2f} $\\mu$s")
    if pyp("regime1_read"):
        lines += [
            f"Regime-I f$^0$ read+compute$^\\dagger$ & Python reference / per quote & "
            f"{pyp('regime1_read')} \\\\",
            f"Regime-II LUT interpolation$^\\dagger$ & Python reference / per quote & "
            f"{pyp('lut_interp')} \\\\",
            "\\midrule",
    ]
    lines += [
        f"Regime-I eigensolve & Python off-path / $\\le$1 Hz & "
        f"{r['eigensolve']['p50_ms']:.3f} ms & {r['eigensolve']['p99_ms']:.3f} ms \\\\",
        f"Regime-II ODE backward solve & Python off-path / $f_t$ trigger & "
        f"{r['ode_backsolve']['p50_ms']:.1f} ms & {r['ode_backsolve']['p99_ms']:.1f} ms \\\\",
        f"\\quad full LUT rebuild (+ $u^*$ + quote grid) & off-path / $f_t$ trigger & "
        f"{r['lut_rebuild_full']['p50_ms']:.1f} ms & {r['lut_rebuild_full']['p99_ms']:.1f} ms \\\\",
        f"Raw HJB PDE backward solve ($N_S={r['pde_backsolve']['n_s']}$) & "
        f"avoided by the ansatz & "
        f"{r['pde_backsolve']['p50_ms']:.0f} ms & {r['pde_backsolve']['p99_ms']:.0f} ms \\\\",
        "\\bottomrule",
        "\\addlinespace[2pt]",
        "\\multicolumn{4}{p{0.92\\linewidth}}{\\footnotesize $\\dagger$ Per-quote "
        "operations cost less than the clock read that would time them "
        "individually; these rows report percentiles across the means of "
        "512-operation batches (3000 batches), identically on both engines, and "
        "are therefore not per-quote tail latencies. The off-path rows are "
        "per-solve percentiles.} \\\\",
        "\\end{tabular}",
        ("% Python reference rows run the SAME operation as their C++ twins "
         "(as-of forward cursor + log-ratio depth recompute for Regime I, "
         "u-interpolation for Regime II) at the SAME batch size, so p50/p99 "
         "are the same statistic on both sides."),
        (f"% PDE/ODE solve ratio = {r['ratios']['pde_over_ode_solve']:.0f}x "
         f"(scales with N_S); full-rebuild headroom vs 1 s trigger = "
         f"{r['ratios']['rebuild_headroom_x']:.0f}x; "
         f"live Thomas-mirror solve = {r['thomas_mirror']['p50_ms']:.1f} ms"),
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-s", type=int, default=512, help="PDE price-grid nodes")
    ap.add_argument("--hftb", type=Path, default=None,
                    help="hft_engine bench HFTB dump for hot-path rows")
    ap.add_argument("--report-dir", type=Path, default=HERE / "reports")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tds:
        work = Path(tds)
        kit = build_bench_kit(work)
        hot = (hotpath_from_hftb(args.hftb) if args.hftb
               else try_run_engine_bench(kit, work))
        py_hot = measure_python_hotpath(kit)
    if args.hftb:
        py_hot["inputs"]["warning"] = (
            "--hftb dump supplied: its kit is unknown and may differ from the "
            "kit the Python rows were measured on")
    eig = measure_eigensolve()
    ode = measure_ode_backsolve()
    lut_full = measure_lut_rebuild_full()
    thomas = measure_thomas_mirror()
    pde = measure_pde_backsolve(n_s=args.n_s)
    r = build_report(hot, eig, ode, pde, lut_full, thomas, py_hot)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "latency_offpath.json").write_text(json.dumps(r, indent=2))
    write_tex(r, args.report_dir / "tab_latency.tex")
    print(json.dumps(r, indent=2))
    print(f"-> {args.report_dir / 'latency_offpath.json'}")
    print(f"-> {args.report_dir / 'tab_latency.tex'}")


if __name__ == "__main__":
    main()
