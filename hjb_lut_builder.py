"""Serialise the backward-induction output as the binary lookup table."""
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import time
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from hjb_principal_eigenvector import HJBParams
from hjb_regime_boundary import find_u_star
from hjb_riccati_solver import (
    FundingParams,
    RiccatiResult,
    quotes_from_logomega,
    solve_backward_reference,
)

MAGIC = b"HFTL"
VERSION = 2
HEADER_SIZE = 256
ROW_ALIGN = 64  # bytes; each q-slice starts on a cache line

SENS_MAGIC = b"HFTS"
SENS_VERSION = 1
SENS_HEADER_SIZE = 64
_SENS_HEADER_STRUCT = struct.Struct("<4sIIIIiidI")

_HEADER_STRUCT = struct.Struct("<4sIIIIiiI" + "d" * 9 + "IIQ20s")
assert _HEADER_STRUCT.size == 140, _HEADER_STRUCT.size


@dataclass
class LUTHeader:
    n_q: int
    n_u: int
    du_ms: int
    q_min: int
    q_max: int
    gamma: float
    k: float
    A: float
    rho: float
    f_t: float
    F_t: float
    sigma: float
    alpha_ml: float
    u_star_s: float
    epoch_index: int = 0
    body_crc32: int = 0
    build_ts_ns: int = 0
    git_sha: bytes = b"\x00" * 20

    def pack(self) -> bytes:
        raw = _HEADER_STRUCT.pack(
            MAGIC, VERSION, self.n_q, self.n_u, self.du_ms,
            self.q_min, self.q_max, 0,
            self.gamma, self.k, self.A, self.rho, self.f_t, self.F_t,
            self.sigma, self.alpha_ml, self.u_star_s,
            self.epoch_index, self.body_crc32, self.build_ts_ns, self.git_sha,
        )
        return raw + b"\x00" * (HEADER_SIZE - len(raw))

    @classmethod
    def unpack(cls, buf: bytes) -> "LUTHeader":
        if len(buf) < HEADER_SIZE:
            raise ValueError(f"LUT header truncated: {len(buf)} < {HEADER_SIZE}")
        (magic, version, n_q, n_u, du_ms, q_min, q_max, _pad,
         gamma, k, A, rho, f_t, F_t, sigma, alpha_ml, u_star_s,
         epoch_index, crc, ts, sha) = _HEADER_STRUCT.unpack(buf[:_HEADER_STRUCT.size])
        if magic != MAGIC:
            raise ValueError(f"bad LUT magic {magic!r}")
        if version != VERSION:
            raise ValueError(f"unsupported LUT version {version} (expected {VERSION})")
        return cls(n_q=n_q, n_u=n_u, du_ms=du_ms, q_min=q_min, q_max=q_max,
                   gamma=gamma, k=k, A=A, rho=rho, f_t=f_t, F_t=F_t,
                   sigma=sigma, alpha_ml=alpha_ml, u_star_s=u_star_s,
                   epoch_index=epoch_index, body_crc32=crc, build_ts_ns=ts,
                   git_sha=sha)


def _git_sha() -> bytes:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).parent, timeout=5,
        )
        if out.returncode == 0:
            return bytes.fromhex(out.stdout.strip())[:20].ljust(20, b"\x00")
    except Exception:
        pass
    return b"\x00" * 20


def _padded_row_floats(n_u: int) -> int:
    """Floats per stored row after zero-padding to a 64-byte multiple."""
    row_bytes = n_u * 4 # bytes per float
    padded = ((row_bytes + ROW_ALIGN - 1) // ROW_ALIGN) * ROW_ALIGN
    return padded // 4


def build_log_omega(
    p: HJBParams, fp: FundingParams, u_max: float, du_s: float,
) -> RiccatiResult:
    """High-accuracy backward induction on the exact LUT grid u_i = i * du_s."""
    n_u = int(math.floor(u_max / du_s)) + 1
    u_eval = np.arange(n_u, dtype=np.float64) * du_s
    return solve_backward_reference(p, fp, float(u_eval[-1]), u_eval)


def write_lut(
    path: str | Path,
    p: HJBParams,
    fp: FundingParams,
    res: RiccatiResult,
    *,
    f_t: float,
    du_ms: int,
    u_star_s: float,
    epoch_index: int = 0,
    manifest_extra: dict | None = None,
) -> LUTHeader:
    """Pack header + padded f32 log-omega body, write atomically, emit manifest.

    Atomic write (tmp + rename) so a concurrently-reading engine never sees a
    torn file; the C++ side additionally swaps an atomic pointer.
    """
    path = Path(path)
    n_u = res.u_grid.size
    n_q = 2 * p.Q + 1
    if res.log_omega.shape != (n_u, n_q):
        raise ValueError(f"log_omega shape {res.log_omega.shape} != ({n_u},{n_q})")
    du_s = du_ms / 1000.0
    if n_u > 1 and not (res.u_grid[0] == 0.0
                        and np.allclose(np.diff(res.u_grid), du_s, atol=1e-9)):
        raise ValueError(
            f"res.u_grid is not the uniform grid i*{du_s}s from 0 expected by "
            f"du_ms={du_ms}; header would misdescribe the temporal axis")

    # body: rows are q-slices (q = Q..-Q  ==  state-vector order), columns u.
    row_floats = _padded_row_floats(n_u)
    body = np.zeros((n_q, row_floats), dtype="<f4")
    body[:, :n_u] = res.log_omega.T.astype("<f4")  # (n_q, n_u)
    body_bytes = body.tobytes()

    hdr = LUTHeader(
        n_q=n_q, n_u=n_u, du_ms=du_ms, q_min=-p.Q, q_max=p.Q,
        gamma=p.gamma, k=p.k, A=p.A, rho=fp.rho, f_t=f_t, F_t=fp.F_t,
        sigma=p.sigma, alpha_ml=p.alpha_ml, u_star_s=u_star_s,
        epoch_index=epoch_index, body_crc32=zlib.crc32(body_bytes),
        build_ts_ns=time.time_ns(), git_sha=_git_sha(),
    )

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(hdr.pack())
        fh.write(body_bytes)
    tmp.replace(path)

    manifest = {
        "format": {"magic": "HFTL", "version": VERSION, "body": "log_omega_f32",
                   "row_align_bytes": ROW_ALIGN, "row_order": "q = q_max .. q_min"},
        "params": {
            "gamma": p.gamma, "sigma": p.sigma, "A": p.A, "k": p.k,
            "alpha_ml": p.alpha_ml, "Q": p.Q,
            "alpha": p.alpha, "beta_ml": p.beta_ml, "eta": p.eta,
        },
        "funding": {"rho": fp.rho, "F_t": fp.F_t, "f_t": f_t,
                    "mode": fp.mode, "epoch_s": fp.epoch_s,
                    "drain_scale": fp.drain_scale()},
        "grid": {"n_q": n_q, "n_u": n_u, "du_ms": du_ms,
                 "u_max_s": float(res.u_grid[-1])},
        "u_star_s": u_star_s,
        "epoch_index": epoch_index,
        "solver": res.method,
        "body_crc32": hdr.body_crc32,
        "build_ts_ns": hdr.build_ts_ns,
        "git_sha": hdr.git_sha.hex(),
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return hdr


def read_lut(path: str | Path) -> tuple[LUTHeader, np.ndarray]:
    """Load (header, log_omega) with log_omega shaped (n_q, n_u), q = Q..-Q."""
    raw = Path(path).read_bytes()
    hdr = LUTHeader.unpack(raw)
    row_floats = _padded_row_floats(hdr.n_u)
    expected = HEADER_SIZE + hdr.n_q * row_floats * 4
    if len(raw) != expected:
        raise ValueError(f"LUT size {len(raw)} != expected {expected}")
    body_bytes = raw[HEADER_SIZE:]
    crc = zlib.crc32(body_bytes)
    if crc != hdr.body_crc32:
        raise ValueError(f"LUT body CRC mismatch: {crc:#x} != {hdr.body_crc32:#x}")
    body = np.frombuffer(body_bytes, dtype="<f4").reshape(hdr.n_q, row_floats)
    return hdr, np.ascontiguousarray(body[:, : hdr.n_u], dtype=np.float64)


def sens_path_for(lut_path: str | Path) -> Path:
    """Companion sensitivity file path for a given .hftl path."""
    p = Path(lut_path)
    return p.with_suffix(p.suffix + ".sens")


def _deltas_from_res(res: RiccatiResult, p: HJBParams) -> tuple[np.ndarray, np.ndarray]:
    """(delta_b, delta_a) grids [rows q=Q..-Q, cols u] from f32-quantised
    log-omega -- the exact bytes the .hftl carries (parity with the C++ loader
    and RegimeIIQuoter). alpha_ml does not enter quotes_from_logomega, so p and a
    drift-perturbed p give identical maps for the same log-omega."""
    logw = res.log_omega.astype(np.float32).astype(np.float64).T   # (n_q, n_u)
    n_u = res.u_grid.size
    db = np.empty((2 * p.Q + 1, n_u))
    da = np.empty((2 * p.Q + 1, n_u))
    for j in range(n_u):
        b, a = quotes_from_logomega(logw[:, j], p)
        db[:, j] = b[::-1]
        da[:, j] = a[::-1]
    return db, da


def write_lut_sens(path: str | Path, db_sens: np.ndarray, da_sens: np.ndarray,
                   *, du_ms: int, Q: int, q0_ref: float) -> None:
    """Pack header + two padded f32 sensitivity bodies (db_sens, da_sens),
    written atomically. Row order q = Q..-Q, matching the .hftl body."""
    n_q, n_u = db_sens.shape
    if db_sens.shape != da_sens.shape or n_q != 2 * Q + 1:
        raise ValueError(f"sens shape {db_sens.shape} inconsistent with Q={Q}")
    row_floats = _padded_row_floats(n_u)
    body = np.zeros((2 * n_q, row_floats), dtype="<f4")
    body[:n_q, :n_u] = db_sens.astype("<f4")
    body[n_q:, :n_u] = da_sens.astype("<f4")
    body_bytes = body.tobytes()
    raw = _SENS_HEADER_STRUCT.pack(SENS_MAGIC, SENS_VERSION, n_q, n_u, du_ms,
                                   -Q, Q, float(q0_ref), zlib.crc32(body_bytes))
    hdr = raw + b"\x00" * (SENS_HEADER_SIZE - len(raw))
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(hdr)
        fh.write(body_bytes)
    tmp.replace(path)


def read_lut_sens(path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load (db_sens, da_sens [n_q, n_u], q0_ref). Verifies magic/version/CRC."""
    raw = Path(path).read_bytes()
    (magic, version, n_q, n_u, du_ms, q_min, q_max, q0_ref,
     crc) = _SENS_HEADER_STRUCT.unpack(raw[:_SENS_HEADER_STRUCT.size])
    if magic != SENS_MAGIC:
        raise ValueError(f"bad sens magic {magic!r}")
    if version != SENS_VERSION:
        raise ValueError(f"unsupported sens version {version}")
    row_floats = _padded_row_floats(n_u)
    expected = SENS_HEADER_SIZE + 2 * n_q * row_floats * 4
    if len(raw) != expected:
        raise ValueError(f"sens size {len(raw)} != expected {expected}")
    body_bytes = raw[SENS_HEADER_SIZE:]
    if zlib.crc32(body_bytes) != crc:
        raise ValueError("sens body CRC mismatch")
    body = np.frombuffer(body_bytes, dtype="<f4").reshape(2 * n_q, row_floats)
    db = np.ascontiguousarray(body[:n_q, :n_u], dtype=np.float64)
    da = np.ascontiguousarray(body[n_q:, :n_u], dtype=np.float64)
    return db, da, float(q0_ref)


def verify_on_read(path: str | Path, res: RiccatiResult) -> float:
    """Reload and compare against the source array; returns max |diff| (f32
    quantisation only). Raises if structure or CRC mismatch."""
    hdr, logw = read_lut(path)
    src = res.log_omega.T  # (n_q, n_u)
    if logw.shape != src.shape:
        raise ValueError(f"reloaded shape {logw.shape} != source {src.shape}")
    return float(np.max(np.abs(logw - src.astype(np.float32).astype(np.float64))))


def lut_quotes_at(
    hdr: LUTHeader, logw: np.ndarray, p: HJBParams, u_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    du_s = hdr.du_ms / 1000.0
    x = min(max(u_s / du_s, 0.0), hdr.n_u - 1.0)
    i0 = int(math.floor(x))
    i1 = min(i0 + 1, hdr.n_u - 1)
    a = x - i0
    slice_log = (1.0 - a) * logw[:, i0] + a * logw[:, i1]
    return quotes_from_logomega(slice_log, p)


def build_and_write(
    out_path: str | Path,
    p: HJBParams,
    fp: FundingParams,
    *,
    f_t: float,
    u_max: float,
    du_ms: int,
    eps_ticks: float = 1.0,
    tick: float = 0.1,
    epoch_index: int = 0,
    manifest_extra: dict | None = None,
    q0_ref: float | None = None,
) -> tuple[LUTHeader, RiccatiResult, float]:
    """Solve, locate u*, serialise."""
    res = build_log_omega(p, fp, u_max, du_ms / 1000.0)
    u_star, _ = find_u_star(res, p, eps_ticks=eps_ticks, tick=tick)
    hdr = write_lut(out_path, p, fp, res, f_t=f_t, du_ms=du_ms,
                    u_star_s=u_star, epoch_index=epoch_index,
                    manifest_extra=manifest_extra)
    err = verify_on_read(out_path, res)
    if err > 1e-6:
        raise RuntimeError(f"LUT round-trip beyond f32 quantisation: {err:.2e}")
    if q0_ref is not None:
        gs2 = p.gamma * p.sigma * p.sigma            # q0 -> alpha: alpha = q0*gs2
        res_ref = build_log_omega(replace(p, alpha_ml=q0_ref * gs2), fp,
                                  u_max, du_ms / 1000.0)
        db0, da0 = _deltas_from_res(res, p)
        dbr, dar = _deltas_from_res(res_ref, p)
        write_lut_sens(sens_path_for(out_path), (dbr - db0) / q0_ref,
                       (dar - da0) / q0_ref, du_ms=du_ms, Q=p.Q, q0_ref=q0_ref)
    return hdr, res, u_star


def main() -> None:
    ap = argparse.ArgumentParser(description="Build + serialise the Regime II LUT.")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--u-max", type=float, default=1800.0)
    ap.add_argument("--du-ms", type=int, default=100,
                    help="LUT temporal grid (ms); 10 ms for short horizons, "
                    "coarser for epoch-scale u_max")
    ap.add_argument("--gamma", type=float, default=2e-5)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--A", type=float, default=20.0)
    ap.add_argument("--k", type=float, default=0.145)
    ap.add_argument("--alpha-ml", type=float, default=0.0)
    ap.add_argument("--Q", type=int, default=10)
    ap.add_argument("--rho", type=float, default=1e-6)
    ap.add_argument("--funding-mode", type=str, default="drain_normalized",
                    choices=["drain_normalized", "drain", "terminal_jump"],
                    help="HJB funding-drain mode; drain_normalized is "
                    "production")
    ap.add_argument("--epoch-s", type=float, default=28_800.0,
                    help="inter-settlement duration (s) used by drain_normalized")
    ap.add_argument("--f-t", type=float, default=1e-4,
                    help="funding RATE (dimensionless)")
    ap.add_argument("--s-ref", type=float, default=100_000.0)
    ap.add_argument("--epoch-index", type=int, default=0)
    args = ap.parse_args()


    p = HJBParams(gamma=args.gamma, sigma=args.sigma, A=args.A, k=args.k,
                  alpha_ml=args.alpha_ml, Q=args.Q)
    fp = FundingParams(F_t=args.s_ref * args.f_t, rho=args.rho,
                       mode=args.funding_mode, epoch_s=args.epoch_s)
    hdr, _res, u_star = build_and_write(
        args.out, p, fp, f_t=args.f_t, u_max=args.u_max, du_ms=args.du_ms,
        epoch_index=args.epoch_index,
    )
    print(f"wrote {args.out}  n_u={hdr.n_u}  u*={u_star:.1f}s  "
          f"crc={hdr.body_crc32:#010x}")


if __name__ == "__main__":
    main()
