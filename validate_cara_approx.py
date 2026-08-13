"""Numerical check of the CARA certainty-equivalent approximation."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from hjb_principal_eigenvector import (
    HJBParams,
    discrete_autonomous_quotes,
    principal_eigvec,
)

for _old, _new in (("trapz", "trapezoid"), ("in1d", "isin"), ("row_stack", "vstack"),
                   ("product", "prod"), ("cumproduct", "cumprod"),
                   ("sometrue", "any"), ("alltrue", "all")):
    if not hasattr(np, _old) and hasattr(np, _new):
        setattr(np, _old, getattr(np, _new))

def _identity_njit(*args, **kwargs):  # type: ignore
    def _wrap(f):
        return f

    if args and callable(args[0]):
        return args[0]
    return _wrap


def _probe_numba():
    try:
        from numba import njit as _njit  # type: ignore

        @_njit(cache=False)
        def _t(x):
            return x + 1

        _t(1)
        return _njit, True
    except Exception:
        return _identity_njit, False


njit, _HAVE_NUMBA = _probe_numba()


SECONDS_PER_EPOCH_DEFAULT = 8 * 3600.0


@dataclass
class Params:
    gamma: float = 2.0e-5
    Q: int = 10  # max abs inventory (PLACEHOLDER: fixed hyperparameter, not finalised)
    nu: float = 0.0  # liquidation impact penalty (unused in the autonomous policy here)
    delta_min: float = -5.0  # quote lower bound in USD from mid (crossed half-spread + taker)

    A: float = 0.8  # base arrival intensity at delta -> 0, fills per second
    k: float = 0.15  # intensity decay per USD of half-spread

    sigma: float = 8.0  # realised vol in USD * s^{-1/2} (micro-price)
    alpha_ml: float = 0.0  # ML drift in USD/s; 0 = the baseline no-funding-correction policy
    s_ref: float = 100_000.0  # reference price level (USD)

    f_p50: float = 3.7e-5  # median |settlement funding rate| (~0.37 bp)
    f_p99: float = 1.5e-4  # near-max |settlement funding rate| (~1.5 bp, extreme carry)
    rho: float = 6.46e-3  # production rho = spectral gap of M at queue-aware (A,k),

    epoch_seconds: float = SECONDS_PER_EPOCH_DEFAULT
    n_paths: int = 4000
    seed: int = 12345
    drain_norm: str = "mean_match"

    def drain_normaliser(self) -> float:
        if self.drain_norm == "mean_match":
            denom = 1.0 - math.exp(-self.rho * self.epoch_seconds)
            return self.rho / denom if denom > 0 else 1.0
        return 1.0

    def hjb(self) -> HJBParams:
        """Project onto the canonical HJB parameter object (the single source of
        truth for f^0, the autonomous quotes, and the alpha/beta_ml/eta derived
        coefficients). The extra Params fields (nu, delta_min, s_ref, f_*, the MC
        knobs) are validation-harness inputs, not HJB state."""
        return HJBParams(
            gamma=self.gamma, sigma=self.sigma, A=self.A, k=self.k,
            alpha_ml=self.alpha_ml, Q=self.Q,
        )

    @property
    def eta(self) -> float:
        return self.hjb().eta

    @property
    def alpha(self) -> float:
        return self.hjb().alpha

    @property
    def beta_ml(self) -> float:
        return self.hjb().beta_ml

    def scaled(self, gamma_mult: float) -> "Params":
        d = asdict(self)
        d["gamma"] = self.gamma * gamma_mult
        return Params(**d)


def principal_eigenvector(p: Params) -> np.ndarray:
    return principal_eigvec(p.hjb())


def optimal_quotes(p: Params, f0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Autonomous quotes for the Gillespie, indexed by q in [-Q, Q] (i = q + Q)."""
    delta_b, delta_a = discrete_autonomous_quotes(p.hjb(), f0)
    # canonical encodes withdrawn edges as nan; the Gillespie needs +inf.
    delta_b = np.where(np.isnan(delta_b), np.inf, delta_b)
    delta_a = np.where(np.isnan(delta_a), np.inf, delta_a)
    # Admissibility: delta >= delta_min (inf stays inf).
    finite_b = np.isfinite(delta_b)
    finite_a = np.isfinite(delta_a)
    delta_b[finite_b] = np.maximum(delta_b[finite_b], p.delta_min)
    delta_a[finite_a] = np.maximum(delta_a[finite_a], p.delta_min)
    return delta_b, delta_a


def fill_intensities(p: Params, delta_b: np.ndarray, delta_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """lambda^{b/a}(q) = A exp(-k delta). Withdrawn quotes (delta = inf) -> 0."""
    with np.errstate(over="ignore"):
        lam_b = p.A * np.exp(-p.k * delta_b)
        lam_a = p.A * np.exp(-p.k * delta_a)
    lam_b[~np.isfinite(delta_b)] = 0.0
    lam_a[~np.isfinite(delta_a)] = 0.0
    return lam_b, lam_a


@njit(cache=True)
def _gillespie_paths(
    lam_b: np.ndarray,
    lam_a: np.ndarray,
    delta_b: np.ndarray,
    delta_a: np.ndarray,
    Q: int,
    F: float,
    rho: float,
    epoch_seconds: float,
    n_paths: int,
    seed: int,
):
    """Simulate the inventory birth-death CTMC for n_paths epochs."""
    np.random.seed(seed)
    term_q = np.empty(n_paths, dtype=np.float64)
    integral = np.empty(n_paths, dtype=np.float64)
    spread_cap = np.empty(n_paths, dtype=np.float64)

    for p in range(n_paths):
        q = 0
        s = 0.0
        acc = 0.0
        cap = 0.0
        while s < epoch_seconds:
            i = q + Q
            rb = lam_b[i]
            ra = lam_a[i]
            R = rb + ra
            if R <= 0.0:
                # both quotes withdrawn (only possible if Q == 0): coast to end
                dt = epoch_seconds - s
            else:
                dt = -math.log(max(np.random.random(), 1e-300)) / R
            if s + dt >= epoch_seconds:
                dt = epoch_seconds - s
                # accumulate the final constant-q interval, then stop
                if rho > 0.0:
                    acc += q * F * (
                        math.exp(-rho * (epoch_seconds - s - dt))
                        - math.exp(-rho * (epoch_seconds - s))
                    ) / rho
                else:
                    acc += q * F * dt
                s = epoch_seconds
                break
            # accumulate the constant-q interval [s, s+dt]
            if rho > 0.0:
                acc += q * F * (
                    math.exp(-rho * (epoch_seconds - s - dt))
                    - math.exp(-rho * (epoch_seconds - s))
                ) / rho
            else:
                acc += q * F * dt
            s += dt
            # execute the jump
            u = np.random.random() * R
            if u < rb:
                cap += delta_b[i]
                q += 1
            else:
                cap += delta_a[i]
                q -= 1
            if q > Q:
                q = Q
            elif q < -Q:
                q = -Q
        term_q[p] = q
        integral[p] = acc
        spread_cap[p] = cap
    return term_q, integral, spread_cap


def cara_ce(x: np.ndarray, gamma: float) -> float:
    """CARA certainty-equivalent  -(1/gamma) ln E[exp(gamma x)], computed in a
    numerically stable way via the log-sum-exp shift."""
    z = gamma * x
    m = z.max()
    lse = m + math.log(np.mean(np.exp(z - m)))
    return -(1.0 / gamma) * lse


def analytic_terminal_inventory_moments(
    lam_b: np.ndarray, lam_a: np.ndarray, Q: int, T: float
) -> tuple[float, float]:
    """Exact E[q_T], Var(q_T) of the inventory CTMC via one matrix exponential of
    the generator, with no Monte Carlo.

    Inventory q in [-Q, Q] (state index i = q + Q) is a birth-death CTMC: from
    state i it jumps up at rate lam_b[i] and down at rate lam_a[i], with the
    capacity edges withdrawn. Starting from q_0 = 0, the law at T is
    p(T) = exp(G^T T) e_Q, from which the first two moments are read off."""
    n = 2 * Q + 1
    G = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        up = lam_b[i] if i + 1 < n else 0.0
        dn = lam_a[i] if i - 1 >= 0 else 0.0
        if i + 1 < n:
            G[i, i + 1] = up
        if i - 1 >= 0:
            G[i, i - 1] = dn
        G[i, i] = -(up + dn)
    p0 = np.zeros(n)
    p0[Q] = 1.0  # q_0 = 0
    pT = expm(G.T * T) @ p0
    pT = np.clip(pT, 0.0, None)
    pT = pT / pT.sum()  # guard tiny negative/normalisation drift from expm
    q = np.arange(-Q, Q + 1, dtype=np.float64)
    eq = float(q @ pT)
    var = float((q * q) @ pT - eq * eq)
    return eq, var


@dataclass
class CaraResult:
    f_label: str
    f_value: float
    mean_discrete: float
    mean_continuous: float
    ce_discrete: float  # noisy log-sum-exp CE (MC-dominated cross-check, see run_one)
    ce_continuous: float
    err_mean: float  # |E[C] - E[I]| (paired point estimate)
    err_mean_se: float  # standard error of E[C-I] using COMMON RANDOM NUMBERS
    err_ce: float  # |J_discrete - J_continuous|, robust 2nd-order analytic estimate
    var_corr_discrete: float  # CARA variance correction -(gamma/2) Var(C), analytic
    var_corr_continuous: float  # CARA variance correction -(gamma/2) Var(I), analytic
    epoch_pnl_scale: float  # gross spread capture per epoch (USD)
    err_ce_frac_pnl: float
    var_qT_mc: float  # Var(q_T) from the Gillespie paths
    var_qT_analytic: float  # Var(q_T) exact, via generator matrix exponential


def run_one(p: Params, f_label: str, f_value: float) -> CaraResult:
    f0 = principal_eigenvector(p)
    delta_b, delta_a = optimal_quotes(p, f0)
    lam_b, lam_a = fill_intensities(p, delta_b, delta_a)

    F = p.s_ref * f_value
    term_q, integral_raw, spread_cap = _gillespie_paths(
        lam_b, lam_a, delta_b, delta_a, p.Q, F, p.rho, p.epoch_seconds, p.n_paths, p.seed
    )

    integral = integral_raw * p.drain_normaliser()

    c_discrete = term_q * F  # discrete settlement charge to cash (depends on q_{T_fund})
    n = c_discrete.size

    diff = c_discrete - integral  # per-path (common random numbers)
    mean_disc = float(np.mean(c_discrete))
    mean_cont = float(np.mean(integral))
    err_mean = float(np.mean(diff))  # == mean_disc - mean_cont, but report its CRN SE
    err_mean_se = float(np.std(diff, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")

    var_disc = float(np.var(c_discrete, ddof=1)) if n > 1 else 0.0
    var_cont = float(np.var(integral, ddof=1)) if n > 1 else 0.0
    vc_disc = -0.5 * p.gamma * var_disc  # CARA variance correction (discrete), analytic
    vc_cont = -0.5 * p.gamma * var_cont  # CARA variance correction (continuous), analytic
    err_ce = err_mean + 0.5 * p.gamma * (var_disc - var_cont)

    # log-sum-exp CEs retained ONLY as a (noise-dominated) cross-check.
    ce_disc = cara_ce(c_discrete, p.gamma)
    ce_cont = cara_ce(integral, p.gamma)
    epoch_pnl = float(np.mean(spread_cap))

    # Exact reference for the MC terminal-inventory variance (no Monte-Carlo).
    var_qT_mc = float(np.var(term_q, ddof=1)) if n > 1 else 0.0
    _eq_an, var_qT_analytic = analytic_terminal_inventory_moments(
        lam_b, lam_a, p.Q, p.epoch_seconds
    )

    return CaraResult(
        f_label=f_label,
        f_value=f_value,
        mean_discrete=mean_disc,
        mean_continuous=mean_cont,
        ce_discrete=ce_disc,
        ce_continuous=ce_cont,
        err_mean=err_mean,
        err_mean_se=err_mean_se,
        err_ce=abs(err_ce),
        var_corr_discrete=vc_disc,
        var_corr_continuous=vc_cont,
        epoch_pnl_scale=epoch_pnl,
        err_ce_frac_pnl=abs(err_ce) / epoch_pnl if epoch_pnl > 0 else float("nan"),
        var_qT_mc=var_qT_mc,
        var_qT_analytic=var_qT_analytic,
    )


def run_sweep(base: Params, gamma_mults=(0.5, 1.0, 1.5)) -> list[tuple[float, CaraResult]]:
    rows: list[tuple[float, CaraResult]] = []
    for gm in gamma_mults:
        p = base.scaled(gm)
        for f_label, f_value in (("p50", base.f_p50), ("p99", base.f_p99)):
            res = run_one(p, f_label, f_value)
            rows.append((gm, res))
    return rows


def to_latex(rows: list[tuple[float, CaraResult]]) -> str:
    lines = [
        r"% Auto-generated by validate_cara_approx.py",
        r"% Continuous-decay approximation error (Justification II).",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"$\gamma$ & $|f|$ & $|\mathbb{E}C-\mathbb{E}I|$ (USDT/BTC) & "
        r"$|J_{\text{disc}}-J_{\text{cont}}^{CE}|$ (USDT/BTC) & var.\ corr. (USDT/BTC) & "
        r"\% epoch P\&L \\",
        r"\midrule",
    ]
    for gm, r in rows:
        lines.append(
            f"${gm:g}\\gamma_0$ & {r.f_label} & {r.err_mean:.3e} & {r.err_ce:.3e} & "
            f"{r.var_corr_continuous:.3e} & {100*r.err_ce_frac_pnl:.3f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def make_figure(rows: list[tuple[float, CaraResult]], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    # error vs gamma, split by |f| level
    for f_label in ("p50", "p99"):
        sub = [(gm, r) for gm, r in rows if r.f_label == f_label]
        gms = [gm for gm, _ in sub]
        errs = [r.err_ce for _, r in sub]
        ax0.plot(gms, errs, marker="o", label=f"|f|={f_label}")
    ax0.set_xlabel(r"$\gamma$ multiplier")
    ax0.set_ylabel(r"$|J_{disc}-J_{cont}^{CE}|$ (USDT/BTC)")
    ax0.set_title("CE error vs risk aversion")
    ax0.set_yscale("log")
    ax0.legend()
    ax0.grid(True, which="both", alpha=0.3)

    # error vs |f| at base gamma
    base_rows = [r for gm, r in rows if abs(gm - 1.0) < 1e-9]
    fs = [r.f_value for r in base_rows]
    errs = [r.err_ce for r in base_rows]
    ax1.plot(fs, errs, marker="s", color="C3")
    ax1.set_xlabel(r"$|f|$ (funding rate)")
    ax1.set_ylabel(r"$|J_{disc}-J_{cont}^{CE}|$ (USDT/BTC)")
    ax1.set_title(r"CE error vs $|f|$ (base $\gamma$)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def print_report(rows: list[tuple[float, CaraResult]]) -> None:
    print("\n=== Continuous-decay approximation error (Justification II) ===")
    print(
        f"{'gamma':>8} {'|f|':>5} {'E[C]':>12} {'E[I]':>12} "
        f"{'E[C-I]':>11} {'|err_CE|':>11} {'%epochPnL':>10}"
    )
    for gm, r in rows:
        print(
            f"{gm:>7g}x {r.f_label:>5} {r.mean_discrete:>12.4e} {r.mean_continuous:>12.4e} "
            f"{r.err_mean:>11.3e} {r.err_ce:>11.3e} {100*r.err_ce_frac_pnl:>9.3f}%"
        )
    base = [r for gm, r in rows if abs(gm - 1.0) < 1e-9 and r.f_label == "p50"]
    if base:
        r = base[0]
        print("\nHeadline (base gamma, |f|=p50):")
        print(
            f"  mean / timing difference  E[C-I] = {r.err_mean:.3e} +/- {r.err_mean_se:.2e} USD "
            f"(common random numbers)"
        )
        print(f"  CARA variance correction  -(g/2)Var(C) = {r.var_corr_discrete:.3e} USD (analytic)")
        print(f"                            -(g/2)Var(I) = {r.var_corr_continuous:.3e} USD (analytic)")
        rel = (abs(r.var_qT_mc - r.var_qT_analytic) / r.var_qT_analytic
               if r.var_qT_analytic > 0 else float("nan"))
        print(
            f"  [cross-check] Var(q_T): MC={r.var_qT_mc:.4f} vs exact(expm)="
            f"{r.var_qT_analytic:.4f}  (rel diff {100*rel:.2f}%)"
        )


def load_calib(path: Path, base: Params) -> Params:
    """Override placeholder params from a calibration JSON. Unknown keys ignored;
    only fields present in Params are applied."""
    raw = json.loads(path.read_text())
    d = asdict(base)
    applied: list[str] = []
    for key, val in raw.items():
        if key in d:
            d[key] = val
            applied.append(key)
    if not applied:
        print(
            f"WARN: --calib {path} matched no Params fields; expected a flat "
            f"JSON with top-level keys like 'A', 'k', 'sigma', 'rho'."
        )
    else:
        print(f"Loaded {len(applied)} calibrated field(s) from {path}: {', '.join(sorted(applied))}")
    return Params(**d)


def build_params(args: argparse.Namespace) -> Params:
    p = Params()
    calib = args.calib
    if calib is None:
        default_calib = Path("reports/cara_calib.json")
        if default_calib.exists():
            calib = str(default_calib)
        else:
            print(
                f"WARN: no --calib given and {default_calib} not found; "
                f"using placeholder defaults."
            )
    if calib:
        cp = Path(calib)
        if cp.exists():
            p = load_calib(cp, p)
        else:
            print(f"WARN: --calib {cp} not found; keeping current (placeholder) defaults.")
    for field in ("gamma", "Q", "A", "k", "sigma", "alpha_ml", "s_ref", "rho", "nu", "n_paths", "seed"):
        v = getattr(args, field, None)
        if v is not None:
            setattr(p, field, v)
    if getattr(args, "drain_norm", None) is not None:
        p.drain_norm = args.drain_norm
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", type=str, default=None,
                    help="flat JSON calibration checkpoint (Params fields)")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--Q", type=int, default=None)
    ap.add_argument("--A", type=float, default=None)
    ap.add_argument("--k", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--alpha_ml", type=float, default=None)
    ap.add_argument("--s_ref", type=float, default=None)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--nu", type=float, default=None)
    ap.add_argument("--n_paths", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--drain-norm",
        choices=("mean_match", "none"),
        default=None,
        help="drain normalisation (see DECISION LOG); default mean_match",
    )
    ap.add_argument("--outdir", type=str, default="reports")
    args = ap.parse_args()

    if not _HAVE_NUMBA:
        print("WARN: numba unavailable, running the pure-python Gillespie fallback (slower).")

    p = build_params(args)
    print("Parameters:")
    for kk, vv in asdict(p).items():
        print(f"  {kk:>14} = {vv}")
    print(f"  derived: eta={p.eta:.4e}  alpha={p.alpha:.4e}  beta_ml={p.beta_ml:.4e}")

    rows = run_sweep(p)
    print_report(rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tex = to_latex(rows)
    (outdir / "cara_approx_table.tex").write_text(tex + "\n")
    make_figure(rows, outdir / "cara_approx.png")
    print(f"\nWrote {outdir/'cara_approx_table.tex'} and {outdir/'cara_approx.png'}.")


if __name__ == "__main__":
    main()
