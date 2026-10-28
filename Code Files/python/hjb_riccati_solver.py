"""Regime II backward induction for omega(t) near settlement."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from hjb_principal_eigenvector import (
    HJBParams,
    as_offdiag_factors,
    build_modified_M_dense,
    principal_eigvec,
    q_grid,
)
from hjb_thomas_solver import thomas_solve

_OMEGA_FLOOR = 1e-300


@dataclass
class FundingParams:
    """The runtime funding scalars that make M(t) non-autonomous."""

    F_t: float
    rho: float
    mode: str = "drain"
    epoch_s: float = 28800.0

    def drain_scale(self) -> float:
        """Multiplier on Phi: converts the legacy per-second drain into one
        that integrates to the exact discrete charge ("drain_normalized"),
        or switches the drain off entirely ("terminal_jump")."""
        if self.mode == "drain":
            return 1.0
        if self.mode == "drain_normalized":
            if self.rho <= 0.0:
                return 1.0 / self.epoch_s          # rho -> 0 limit: flat drain
            return self.rho / (1.0 - math.exp(-self.rho * self.epoch_s))
        if self.mode == "terminal_jump":
            return 0.0
        raise ValueError(f"unknown funding mode: {self.mode!r}")


def terminal_omega(p: HJBParams, fp: FundingParams) -> np.ndarray:
    """Stitching boundary condition omega_q(T_fund) for backward induction."""
    f0 = principal_eigvec(p)
    if fp.mode != "terminal_jump":
        return f0
    w = f0 * np.exp(-p.k * q_grid(p.Q) * fp.F_t)
    return w / w.max()


def calibrated_params() -> tuple[HJBParams, FundingParams]:
    """The (p, fp) used by the stiffness report."""
    p = HJBParams(gamma=2.0e-5, sigma=4.57, A=0.274, k=0.090, alpha_ml=0.0, Q=10)
    fp = FundingParams(F_t=6.0, rho=intrinsic_relaxation_rate(p),
                       mode="drain_normalized", epoch_s=28800.0)
    return p, fp


def stiffness_report(
    p: HJBParams | None = None,
    fp: FundingParams | None = None,
    *,
    u: float = 0.0,
    horizon_s: float = 28800.0,
    rk45_real_bound: float = 3.3,
    dt_lut_s: float = 0.100,
) -> dict:
    """Stiffness of omega'(u) = -M(u) omega(u) at the calibrated parameters.

    The system is linear, so J = -M(u) exactly, and M(u) is symmetric, so its
    eigenvalues are real and each mode evolves as exp(-lambda_i u). Reports the
    stiffness ratio S = max|lambda| / min|lambda| and the explicit-RK45 real-axis
    step bound h <= rk45_real_bound / max|lambda|."""
    from scipy.linalg import eigh_tridiagonal

    if p is None or fp is None:
        cp, cfp = calibrated_params()
        p = p or cp
        fp = fp or cfp

    diag = assemble_M_diag(u, p, fp)        # D_q(u) = alpha q^2 - beta_ML q + Phi
    off = np.full(diag.size - 1, -p.eta)    # constant sub-/super-diagonal
    lam = eigh_tridiagonal(diag, off, eigvals_only=True)

    abslam = np.abs(lam)
    lam_max = float(abslam.max())
    lam_min = float(abslam.min())
    S = lam_max / lam_min if lam_min > 0 else math.inf

    tau_fast = 1.0 / lam_max
    tau_slow = 1.0 / lam_min if lam_min > 0 else math.inf

    h_rk45 = rk45_real_bound / lam_max
    n_rk45 = horizon_s / h_rk45             # stability-forced explicit step count
    n_impl = horizon_s / dt_lut_s           # accuracy-limited implicit step count

    rep = {
        "params": (p, fp),
        "u": u,
        "alpha": p.alpha, "beta_ml": p.beta_ml, "eta": p.eta,
        "lambda_min_signed": float(lam.min()),
        "lambda_max_signed": float(lam.max()),
        "abs_lambda_min": lam_min,
        "abs_lambda_max": lam_max,
        "stiffness_ratio": S,
        "tau_fast_s": tau_fast,
        "tau_slow_s": tau_slow,
        "h_rk45_stability_s": h_rk45,
        "n_steps_rk45": n_rk45,
        "n_steps_implicit": n_impl,
        "rk45_over_implicit": n_rk45 / n_impl,
        "horizon_s": horizon_s,
    }
    return rep


def _print_stiffness_report(rep: dict) -> None:
    p, fp = rep["params"]
    print("=" * 70)
    print("STIFFNESS REPORT  omega'(u) = -M(u) omega(u)   (Jacobian J = -M)")
    print("=" * 70)
    print(f"params: gamma={p.gamma:.2e} sigma={p.sigma} A={p.A} k={p.k} "
          f"alpha_ml={p.alpha_ml} Q={p.Q}")
    print(f"funding: F_t={fp.F_t} rho={fp.rho:.2e} mode={fp.mode}  evaluated at u={rep['u']} s")
    print(f"derived: alpha={rep['alpha']:.4e}  beta_ml={rep['beta_ml']:.4e}  "
          f"eta={rep['eta']:.4e}")
    print("-" * 70)
    print(f"eigenvalues of M (signed): [{rep['lambda_min_signed']:+.4e}, "
          f"{rep['lambda_max_signed']:+.4e}]  (real & symmetric)")
    print(f"|lambda| range           : [{rep['abs_lambda_min']:.4e}, "
          f"{rep['abs_lambda_max']:.4e}]  (1/s)")
    print(f"fastest mode  tau_fast   = {rep['tau_fast_s']:.3g} s")
    print(f"slowest mode  tau_slow   = {rep['tau_slow_s']:.3g} s")
    print(f">>> STIFFNESS RATIO S    = {rep['stiffness_ratio']:.4g}  "
          f"(max|lambda| / min|lambda|)")
    print("-" * 70)
    print(f"explicit RK45 stability step bound  h <= {rep['h_rk45_stability_s']:.3g} s")
    print(f"  -> steps over horizon ({rep['horizon_s']:.0f} s): "
          f"RK45 (stability) ~ {rep['n_steps_rk45']:.3g}   vs   "
          f"implicit (accuracy, 100 ms LUT) ~ {rep['n_steps_implicit']:.3g}")
    print(f"  -> RK45 / implicit step count = {rep['rk45_over_implicit']:.3g}x")
    print("=" * 70)


def stiffness_robustness_sweep(factor: float = 0.5) -> None:
    """Show the stiffness verdict is robust to +/- `factor` mis-calibration of the
    three structural scalars (sigma -> alpha, A and k -> eta), so the modelling
    choice does not hinge on one pre-analysis point estimate."""
    p0, fp0 = calibrated_params()
    print("\nROBUSTNESS SWEEP (+/- {:.0%} on each structural scalar, others held):"
          .format(factor))
    print(f"{'variant':<16}{'S':>10}{'tau_fast':>12}{'tau_slow':>12}"
          f"{'h_RK45(s)':>12}")
    rows = [("baseline", p0)]
    for name, kw in [
        ("sigma low",  {"sigma": p0.sigma * (1 - factor)}),
        ("sigma high", {"sigma": p0.sigma * (1 + factor)}),
        ("A low",      {"A": p0.A * (1 - factor)}),
        ("A high",     {"A": p0.A * (1 + factor)}),
        ("k low",      {"k": p0.k * (1 - factor)}),
        ("k high",     {"k": p0.k * (1 + factor)}),
    ]:
        rows.append((name, replace(p0, **kw)))
    for name, p in rows:
        r = stiffness_report(p, fp0)
        print(f"{name:<16}{r['stiffness_ratio']:>10.1f}{r['tau_fast_s']:>12.2f}"
              f"{r['tau_slow_s']:>12.1f}{r['h_rk45_stability_s']:>12.2f}")


def intrinsic_relaxation_rate(p: HJBParams) -> float:
    """Spectral gap lambda_1 - lambda_0 of the autonomous M (1/s): the rate at
    which any terminal perturbation (e.g. the terminal_jump charge factor)
    relaxes back to the principal eigenmode in backward time. This is the
    model-intrinsic boundary-layer width, the quantity the empirical rho
    proxies for in the drain modes."""
    from scipy.linalg import eigh_tridiagonal
    from hjb_principal_eigenvector import build_autonomous_M

    diag, off = build_autonomous_M(p)
    w = eigh_tridiagonal(diag, off, select="i", select_range=(0, 1),
                         eigvals_only=True)
    return float(w[1] - w[0])


def funding_diag(u: float, p: HJBParams, fp: FundingParams) -> np.ndarray:
    """Phi(u, q) = k q F_t s exp(-rho u) on the q-grid (ordered Q..-Q), with
    s = fp.drain_scale() (1 legacy; rho-normalised; 0 for terminal_jump)."""
    q = q_grid(p.Q)
    return p.k * q * fp.F_t * fp.drain_scale() * math.exp(-fp.rho * u)


def assemble_M_diag(u: float, p: HJBParams, fp: FundingParams) -> np.ndarray:
    """Main diagonal D_q(u) = alpha q^2 - beta_ML q + Phi(u, q), ordered Q..-Q.

    The off-diagonals of M are the constant -eta (both sub and super); only the
    diagonal is time-dependent, because Phi enters only on the diagonal. The
    funding term is delegated to funding_diag (single source for Phi).
    """
    q = q_grid(p.Q)
    return p.alpha * q ** 2 - p.beta_ml * q + funding_diag(u, p, fp)


def dense_M(u: float, p: HJBParams, fp: FundingParams) -> np.ndarray:
    """Dense M(u) (frozen-coefficient oracle and Jacobian cross-checks). The
    off-diagonals carry the fill-conditional AS reweighting, reducing to
    the constant -eta at xi=0; the funding term Phi enters only on the diagonal."""
    M = build_modified_M_dense(p)            # autonomous diag + AS off-diagonals
    np.fill_diagonal(M, np.diagonal(M) + funding_diag(u, p, fp))
    return M


def _logspace_rhs(u: float, y: np.ndarray, p: HJBParams, fp: FundingParams) -> np.ndarray:
    """RHS of d(ln omega)/du = -[M(u) omega]/omega, ordered q = Q..-Q:
        dy_i/du = -D_i(u) + eta [ exp(y_{i-1}-y_i) + exp(y_{i+1}-y_i) ],
    boundary rows dropping the absent neighbour. (-assemble_M_diag returns a fresh
    array, so no in-place aliasing.)"""
    eta = p.eta
    bid, ask = as_offdiag_factors(p)          # AS reweights (both 1 at xi=0)
    dy = -assemble_M_diag(u, p, fp)
    dy[1:] += eta * bid[1:] * np.exp(y[:-1] - y[1:])    # lower neighbour i-1 (bid, q->q+1)
    dy[:-1] += eta * ask[:-1] * np.exp(y[1:] - y[:-1])  # upper neighbour i+1 (ask, q->q-1)
    return dy


def _logspace_jac(u: float, y: np.ndarray, p: HJBParams, fp: FundingParams) -> np.ndarray:
    """Analytic tridiagonal Jacobian of _logspace_rhs. The diagonal D_i(u) has no
    y-dependence and drops out, so the Jacobian depends only on y and eta:
        J_{i,i-1} = eta exp(y_{i-1}-y_i),  J_{i,i+1} = eta exp(y_{i+1}-y_i),
        J_ii      = -(those present neighbours)."""
    n = y.size
    eta = p.eta
    bid, ask = as_offdiag_factors(p)
    J = np.zeros((n, n))
    lo = eta * bid[1:] * np.exp(y[:-1] - y[1:])   # couples i to i-1 (bid, i>=1)
    up = eta * ask[:-1] * np.exp(y[1:] - y[:-1])  # couples i to i+1 (ask, i<=n-2)
    idx = np.arange(n - 1)
    J[idx + 1, idx] = lo                # J[i, i-1]
    J[idx, idx + 1] = up                # J[i, i+1]
    diag = np.zeros(n)
    diag[1:] -= lo
    diag[:-1] -= up
    J[np.arange(n), np.arange(n)] = diag
    return J


def quotes_from_logomega(
    logw: np.ndarray, p: HJBParams
) -> tuple[np.ndarray, np.ndarray]:
    """Optimal (delta_b, delta_a) from ln omega (ordered q = Q..-Q).

    Returned indexed by q in [-Q, Q] (position i = q + Q); the capacity-edge
    quote that does not exist (bid at q=Q, ask at q=-Q) is np.nan. Identical
    formula to hjb_principal_eigenvector.discrete_autonomous_quotes, but consuming
    log-omega directly (avoids an exp round-trip and the associated overflow).
    """
    Q = p.Q
    const = (1.0 / p.gamma) * math.log1p(p.gamma / p.k)

    def y(q: int) -> float:
        return logw[Q - q]  # logw ordered Q..-Q

    delta_b = np.full(2 * Q + 1, np.nan)
    delta_a = np.full(2 * Q + 1, np.nan)
    for qq in range(-Q, Q + 1):
        i = qq + Q
        if qq != Q:
            delta_b[i] = const + (1.0 / p.k) * (y(qq) - y(qq + 1)) + (qq + 1) * p.xi_b
        if qq != -Q:
            delta_a[i] = const + (1.0 / p.k) * (y(qq) - y(qq - 1)) - (qq - 1) * p.xi_a
    return delta_b, delta_a


@dataclass
class RiccatiResult:
    """Backward-induction solution over u in [0, u_max].

    u_grid    : (n_u,) backward-time grid, u = T_fund - t, ascending from 0.
    log_omega : (n_u, n_q) ln omega_q, ordered q = Q..-Q, level-normalised so
                max_q log_omega[i] == 0 at each u (quotes are level-invariant).
    method    : 'radau-logspace', 'crank-nicolson-thomas', or 'theta{theta}-thomas'.
    """

    u_grid: np.ndarray
    log_omega: np.ndarray
    method: str

    def quotes(self, p: HJBParams) -> tuple[np.ndarray, np.ndarray]:
        """(delta_b, delta_a) arrays of shape (n_u, 2Q+1), indexed by q+Q."""
        n_u = self.u_grid.size
        db = np.empty((n_u, 2 * p.Q + 1))
        da = np.empty((n_u, 2 * p.Q + 1))
        for i in range(n_u):
            db[i], da[i] = quotes_from_logomega(self.log_omega[i], p)
        return db, da


def solve_backward_reference(
    p: HJBParams,
    fp: FundingParams,
    u_max: float,
    u_eval: np.ndarray | None = None,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> RiccatiResult:
    """Integrate d(ln omega)/du = -[M omega]/omega backward from u=0 (omega=f^0) to u_max."""
    f0 = terminal_omega(p, fp)
    y0 = np.log(np.maximum(f0, _OMEGA_FLOOR))

    def rhs(u: float, y: np.ndarray) -> np.ndarray:
        return _logspace_rhs(u, y, p, fp)

    def jac(u: float, y: np.ndarray) -> np.ndarray:
        return _logspace_jac(u, y, p, fp)

    if u_eval is None:
        u_eval = np.linspace(0.0, u_max, 256)
    u_eval = np.asarray(u_eval, dtype=np.float64)

    sol = solve_ivp(
        rhs, (0.0, u_max), y0, method="Radau", jac=jac,
        t_eval=u_eval, rtol=rtol, atol=atol, dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"Radau backward induction failed: {sol.message}")

    log_omega = sol.y.T.copy()                       # (n_u, n_q)
    log_omega -= log_omega.max(axis=1, keepdims=True)  # level-normalise
    return RiccatiResult(u_grid=u_eval, log_omega=log_omega, method="radau-logspace")


def solve_backward_thomas(
    p: HJBParams,
    fp: FundingParams,
    u_max: float,
    du: float,
    *,
    theta: float = 0.5,
    u_eval: np.ndarray | None = None,
) -> RiccatiResult:
    """theta-method for d omega/du = -M(u) omega, one Thomas solve per step."""
    Q = p.Q
    n = 2 * Q + 1
    eta = p.eta
    bid, ask = as_offdiag_factors(p)
    lhs_sub = -theta * du * eta * bid[1:]    # A[i+1, i] = -theta du eta bid[i+1]
    lhs_sup = -theta * du * eta * ask[:-1]   # A[i, i+1] = -theta du eta ask[i]
    w_old = (1.0 - theta) * du  # explicit weight on M(u_n)

    omega = terminal_omega(p, fp).astype(np.float64)   # u = 0 (already max-normalised)

    n_steps = int(math.ceil(u_max / du))
    us = [0.0]
    logws = [np.log(np.maximum(omega, _OMEGA_FLOOR))]
    u = 0.0
    for _ in range(n_steps):
        u_new = u + du
        # explicit RHS: (I - (1-theta) du M(u_n)) omega^n  (tridiagonal matvec)
        if w_old > 0.0:
            D_old = assemble_M_diag(u, p, fp)
            rhs = (1.0 - w_old * D_old) * omega
            rhs[:-1] += w_old * eta * ask[:-1] * omega[1:]  # upper neighbour (ask, q->q-1)
            rhs[1:] += w_old * eta * bid[1:] * omega[:-1]   # lower neighbour (bid, q->q+1)
        else:
            rhs = omega
        # implicit LHS solve: (I + theta du M(u_{n+1})) omega^{n+1} = rhs
        diag = 1.0 + theta * du * assemble_M_diag(u_new, p, fp)
        if np.any(diag <= 0.0):
            raise ValueError(
                f"theta-method LHS diagonal lost positivity at u={u_new:.4g}; "
                f"reduce du (need theta du max|D_q| < 1)."
            )
        omega = thomas_solve(lhs_sub, diag, lhs_sup, rhs)
        if np.any(omega <= 0.0):
            raise ValueError(
                f"omega lost positivity at u={u_new:.4g}; reduce du or use theta=1."
            )
        omega = omega / omega.max()
        u = u_new
        us.append(u)
        logws.append(np.log(np.maximum(omega, _OMEGA_FLOOR)))

    u_arr = np.asarray(us)
    log_all = np.asarray(logws)   # each row already level-normalised (max == 0) by
                                  # the per-step omega/omega.max() above
    method = "crank-nicolson-thomas" if theta == 0.5 else f"theta{theta}-thomas"

    if u_eval is not None:
        u_eval = np.asarray(u_eval, dtype=np.float64)
        sel = np.searchsorted(u_arr, u_eval).clip(0, u_arr.size - 1)
        return RiccatiResult(u_eval, log_all[sel], method)
    return RiccatiResult(u_arr, log_all, method)


def _max_interior_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Max |a-b| over entries where both are finite (ignore the nan edges)."""
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "stiffness":
        _print_stiffness_report(stiffness_report())
        stiffness_robustness_sweep()
