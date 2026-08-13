"""Shared statistical tests used by the analysis and validation scripts."""
from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tools.tools import add_constant
from scipy import stats


class ADFResult(NamedTuple):
    statistic: float
    p_value: float
    lags: int
    n_obs: int
    critical_values: dict[str, float]
    stationary: bool

class KPSSResult(NamedTuple):
    statistic: float
    p_value: float
    lags: int
    n_obs: int
    critical_values: dict[str, float]
    stationary: bool
    p_value_capped: bool

class ARCHResult(NamedTuple):
    statistic: float
    p_value: float
    f_statistic: float
    f_p_value: float
    heteroskedastic: bool

class DMResult(NamedTuple):
    statistic: float
    p_value: float
    mean_diff: float
    hac_std_err: float
    superior: bool

class JKMResult(NamedTuple):
    z_statistic: float
    p_value: float
    sharpe_diff: float


def adf_test(series: np.ndarray | pd.Series, regression: str = "c", maxlag: int = 20) -> ADFResult:
    """
    Augmented Dickey-Fuller unit root test for stationarity.
    H0: The series has a unit root (non-stationary).
    H1: The series is stationary.
    """
    if isinstance(series, pd.Series):
        series = series.dropna().to_numpy()

    res = adfuller(series, regression=regression, autolag="AIC", maxlag=maxlag)

    return ADFResult(
        statistic=float(res[0]),
        p_value=float(res[1]),
        lags=int(res[2]),
        n_obs=int(res[3]),
        critical_values=res[4],
        stationary=bool(res[1] < 0.05)
    )


def kpss_test(
    series: np.ndarray | pd.Series,
    regression: str = "c",
    nlags: int | str = "auto",
) -> KPSSResult:
    """Kwiatkowski-Phillips-Schmidt-Shin test for stationarity."""
    if isinstance(series, pd.Series):
        series = series.dropna().to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value, lags, crit = kpss(series, regression=regression, nlags=nlags)

    upper_crit = float(crit.get("1%", float("inf")))
    lower_crit = float(crit.get("10%", float("-inf")))
    capped = bool(stat > upper_crit or stat < lower_crit)

    return KPSSResult(
        statistic=float(stat),
        p_value=float(p_value),
        lags=int(lags),
        n_obs=int(len(series)),
        critical_values={k: float(v) for k, v in crit.items()},
        stationary=bool(p_value >= 0.05),
        p_value_capped=capped,
    )


def arch_test(returns: np.ndarray | pd.Series, lags: int = 10) -> ARCHResult:
    """
    Engle's ARCH test for Autoregressive Conditional Heteroskedasticity.
    H0: Residuals are homoskedastic (no volatility clustering).
    H1: ARCH effects are present (volatility clusters).
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna().to_numpy()

    resids = returns - np.mean(returns)

    lm, lmpval, fval, fpval = het_arch(resids, nlags=lags)

    return ARCHResult(
        statistic=float(lm),
        p_value=float(lmpval),
        f_statistic=float(fval),
        f_p_value=float(fpval),
        heteroskedastic=bool(lmpval < 0.05)
    )


def vif_check(X: pd.DataFrame) -> pd.Series:
    """
    Compute VIF for a feature matrix. VIF > 10 indicates high multicollinearity.
    Used to show that disjoint bands reduce redundancy.
    """
    # Drop rows with any NaN before VIF computation
    X_clean = X.dropna()
    if X_clean.empty:
        return pd.Series(index=X.columns, dtype=float)

    Xc = add_constant(X_clean, has_constant="add")
    vif_data = pd.Series(
        [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])],
        index=Xc.columns,
        name="VIF",
    ).drop("const")
    return vif_data


def diebold_mariano_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray, horizon: int = 1) -> DMResult:
    """
    Diebold-Mariano test for predictive accuracy.
    Tests if Model B is significantly more accurate than Model A.
    Uses squared-error loss and Newey-West HAC standard errors.
    """
    e_a = (y_true - y_pred_a)**2
    e_b = (y_true - y_pred_b)**2
    d = e_a - e_b

    mean_d = np.mean(d)
    n = len(d)

    schwert = int(np.floor(4 * (n / 100)**(2/9)))
    nlags = max(horizon - 1, schwert)

    df_d = pd.DataFrame({'d': d})
    df_d['const'] = 1

    import statsmodels.api as sm
    ols = sm.OLS(df_d['d'], df_d['const']).fit()

    hac_results = ols.get_robustcov_results(cov_type='HAC', maxlags=nlags)
    hac_std_err = hac_results.bse[0]

    dm_stat = mean_d / hac_std_err if hac_std_err > 0 else 0.0
    p_val = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return DMResult(
        statistic=float(dm_stat),
        p_value=float(p_val),
        mean_diff=float(mean_d),
        hac_std_err=float(hac_std_err),
        superior=bool(p_val < 0.05 and dm_stat > 0)
    )


def sharpe_ratio_test(returns_i: np.ndarray, returns_j: np.ndarray) -> JKMResult:
    """
    Jobson-Korkie-Memmel test for the equality of two Sharpe Ratios.
    H0: SR_i == SR_j
    H1: SR_i != SR_j
    """
    # Requires returns to be from the same time period (paired)
    mu_i, mu_j = np.mean(returns_i), np.mean(returns_j)
    sig_i, sig_j = np.std(returns_i), np.std(returns_j)
    sig_ij = np.cov(returns_i, returns_j, ddof=0)[0, 1]
    n = len(returns_i)

    theta = sig_j * mu_i - sig_i * mu_j

    v = (1/n) * (2 * sig_i**2 * sig_j**2 - 2 * sig_i * sig_j * sig_ij +
                 0.5 * mu_i**2 * sig_j**2 + 0.5 * mu_j**2 * sig_i**2 -
                 (mu_i * mu_j / (sig_i * sig_j)) * sig_ij**2)

    z = theta / np.sqrt(v) if v > 0 else 0.0
    p_val = 2 * (1 - stats.norm.cdf(np.abs(z)))

    sr_i = mu_i / sig_i if sig_i > 0 else 0.0
    sr_j = mu_j / sig_j if sig_j > 0 else 0.0

    return JKMResult(
        z_statistic=float(z),
        p_value=float(p_val),
        sharpe_diff=float(sr_i - sr_j)
    )


class BootstrapCI(NamedTuple):
    point: float
    lo: float
    hi: float
    block_len: float
    n_boot: int


class MWUResult(NamedTuple):
    statistic: float
    p_value: float
    rank_biserial: float   # effect size in [-1, 1]; >0 means a tends larger
    n_a: int
    n_b: int


def sharpe_ratio(daily_pnl: np.ndarray, ann_factor: float = np.sqrt(365.0)) -> float:
    """Annualised Sharpe of a daily P&L series (crypto trades 365 days/yr).
    Zero-variance degenerate series returns 0.0 rather than inf."""
    x = np.asarray(daily_pnl, dtype=float)
    s = x.std(ddof=1)
    return float(ann_factor * x.mean() / s) if s > 0 else 0.0


def empirical_block_length(x: np.ndarray, max_frac: float = 0.25) -> int:
    """Noise-band block length: the smallest lag L at which |acf(L)| drops below
    the 1.96/sqrt(n) band, capped at max_frac * n. Returns >= 1.

    Robust fallback for ppw_block_length, which is the default block-length
    estimator used by the bootstrap CIs."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return 1
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        return 1
    band = 1.96 / np.sqrt(n)
    cap = max(1, int(max_frac * n))
    for lag in range(1, cap + 1):
        acf = float(np.dot(xc[:-lag], xc[lag:])) / denom
        if abs(acf) < band:
            return lag
    return cap


def ppw_block_length(x: np.ndarray, max_frac: float = 0.25) -> float:
    """MSE-optimal mean block length for the *stationary* bootstrap: Politis-White (2004)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    try:
        eps = x - x.mean()
        if n < 16 or float(eps @ eps) <= 0.0:
            raise ValueError("series too short or degenerate for PPW")
        b_max = float(np.ceil(min(3.0 * np.sqrt(n), n / 3.0))) # maximum block length
        kn = max(5, int(np.log10(n))) # number of consecutive insignificant autocorrelations to define m
        m_max = int(np.ceil(np.sqrt(n))) + kn
        cv = 2.0 * np.sqrt(np.log10(n) / n) # critical value for insignificant autocorrelation
        acv = np.zeros(m_max + 1)
        abs_acorr = np.zeros(m_max + 1)
        opt_m: int | None = None
        r0 = float(eps @ eps)  # = n * R_hat(0)
        for i in range(m_max + 1):
            cross = float(eps[i:] @ eps[: n - i])
            acv[i] = cross / n
            abs_acorr[i] = abs(cross) / r0 if r0 > 0.0 else 0.0
            if i >= kn and opt_m is None and bool(np.all(abs_acorr[i - kn:i] < cv)):
                opt_m = i - kn
        m = 2 * max(opt_m, 1) if opt_m is not None else m_max
        m = min(m, m_max)
        g = 0.0
        lr_acv = acv[0]
        for k in range(1, m + 1):
            lam = 1.0 if k / m <= 0.5 else 2.0 * (1.0 - k / m)
            g += 2.0 * lam * k * acv[k]
            lr_acv += 2.0 * lam * acv[k]
        d_sb = 2.0 * lr_acv**2
        if d_sb <= 0.0:
            raise ValueError("non-positive long-run variance")
        b_sb = ((2.0 * g**2) / d_sb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
        b_sb = min(b_sb, b_max)
        if not np.isfinite(b_sb) or b_sb <= 0.0:
            raise ValueError("degenerate block length")
        return float(b_sb)
    except Exception:
        return float(empirical_block_length(x, max_frac=max_frac))


def _stationary_bootstrap_indices(n: int, mean_block: float,
                                  rng: np.random.Generator) -> np.ndarray:
    """One resample of indices via Politis-Romano stationary bootstrap: geometric block."""
    p = 1.0 / max(mean_block, 1.0) # probability parameter of a geometric distribution
    # each block has length >= 1, so n blocks always suffice
    lengths = rng.geometric(p, size=n).astype(np.int64)
    starts = rng.integers(0, n, size=n)
    csum = np.cumsum(lengths)
    nb = int(np.searchsorted(csum, n)) + 1  # blocks needed to reach n
    lengths = lengths[:nb].copy()
    starts = starts[:nb]
    total_before_last = int(csum[nb - 2]) if nb > 1 else 0
    lengths[-1] = n - total_before_last     # truncate the final block
    # idx = concat_b [starts[b] + arange(lengths[b])] mod n, fully vectorised
    block_id = np.repeat(np.arange(nb), lengths)
    offsets = np.arange(n) - np.repeat(
        np.concatenate(([0], np.cumsum(lengths[:-1]))), lengths)
    return (starts[block_id] + offsets) % n


def block_bootstrap_ci(
    x: np.ndarray,
    stat_fn=sharpe_ratio,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    block_len: float | None = None,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile CI for stat_fn(x) under the stationary block bootstrap."""
    x = np.asarray(x, dtype=float)
    bl = float(block_len) if block_len is not None else ppw_block_length(x)
    rng = np.random.default_rng(seed)
    stats_b = np.empty(n_boot)
    for b in range(n_boot):
        stats_b[b] = stat_fn(x[_stationary_bootstrap_indices(x.size, bl, rng)])
    lo, hi = np.quantile(stats_b, [alpha / 2, 1 - alpha / 2])
    return BootstrapCI(float(stat_fn(x)), float(lo), float(hi), bl, n_boot)


def block_bootstrap_diff_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn=sharpe_ratio,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    block_len: float | None = None,
    seed: int = 0,
) -> BootstrapCI:
    """CI for stat_fn(x) - stat_fn(y) on PAIRED daily series (same days):
    one index resample is applied to BOTH series, preserving their
    cross-sectional dependence, the right design for ablation gaps
    (strategy 4 vs 6, 5 vs 6) measured on identical market days."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("paired bootstrap needs equal-length daily series")
    if block_len is None:
        block_len = max(ppw_block_length(x), ppw_block_length(y))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(x.size, float(block_len), rng)
        diffs[b] = stat_fn(x[idx]) - stat_fn(y[idx])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return BootstrapCI(float(stat_fn(x) - stat_fn(y)), float(lo), float(hi),
                       float(block_len), n_boot)


class InteractionResult(NamedTuple):
    point: float          # delta = (stat(a1)-stat(a0)) - (stat(b1)-stat(b0))
    lo: float
    hi: float
    p_value: float        # two-sided bootstrap p for H0: delta == 0
    eff_treated: float    # stat(a1) - stat(a0): the ML effect WITH funding
    eff_control: float    # stat(b1) - stat(b0): the ML effect WITHOUT funding
    block_len: float
    n_boot: int


def block_bootstrap_interaction_ci(
    a1: np.ndarray, a0: np.ndarray,
    b1: np.ndarray, b0: np.ndarray,
    stat_fn=sharpe_ratio,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    block_len: float | None = None,
    seed: int = 0,
) -> InteractionResult:
    """Difference-in-differences INTERACTION on four PAIRED daily series (same days, equal."""
    a1 = np.asarray(a1, dtype=float)
    a0 = np.asarray(a0, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    b0 = np.asarray(b0, dtype=float)
    n = a1.size
    if not (a0.size == n and b1.size == n and b0.size == n):
        raise ValueError("interaction bootstrap needs four equal-length series")
    if block_len is None:
        block_len = max(ppw_block_length(a1), ppw_block_length(a0),
                        ppw_block_length(b1), ppw_block_length(b0))
    rng = np.random.default_rng(seed)

    def _interaction(i1, i0, j1, j0) -> float:
        return (stat_fn(i1) - stat_fn(i0)) - (stat_fn(j1) - stat_fn(j0))

    point = _interaction(a1, a0, b1, b0)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stationary_bootstrap_indices(n, float(block_len), rng)
        deltas[b] = _interaction(a1[idx], a0[idx], b1[idx], b0[idx])
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    p_two = 2.0 * min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0)))
    return InteractionResult(
        float(point), float(lo), float(hi), min(p_two, 1.0),
        float(stat_fn(a1) - stat_fn(a0)), float(stat_fn(b1) - stat_fn(b0)),
        float(block_len), n_boot,
    )


def mann_whitney_test(a: np.ndarray, b: np.ndarray,
                      alternative: str = "two-sided") -> MWUResult:
    """Mann-Whitney U on two independent samples (e.g. per-fill markouts of
    two strategies). rank_biserial = 2U/(n_a n_b) - 1, the probability that a
    random a-draw exceeds a random b-draw, rescaled to [-1, 1]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    res = stats.mannwhitneyu(a, b, alternative=alternative)
    rbc = 2.0 * float(res.statistic) / (a.size * b.size) - 1.0
    return MWUResult(float(res.statistic), float(res.pvalue), rbc,
                     a.size, b.size)


def benjamini_hochberg(p_values, q: float = 0.05):
    """BH step-up procedure. Returns (reject_mask, adjusted_p) in the input
    order. Used across the 6-strategy x 3-horizon markout test family and the
    pairwise Sharpe comparisons (analysis_pnl / analysis_markout)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adj
    return adjusted <= q, adjusted


class PSRResult(NamedTuple):
    psr: float            # P(true SR > sr_benchmark)
    sr_hat: float         # per-observation (NON-annualised) Sharpe
    skew: float           # sample skewness of returns
    kurt: float           # Pearson kurtosis of returns (normal = 3)
    n: int
    sr_benchmark: float


class DSRResult(NamedTuple):
    dsr: float            # P(true SR > expected max under the null of N trials)
    sr0: float            # expected max Sharpe under the null (per-observation)
    sr_hat: float
    n_trials: int
    var_trials: float     # variance of the trial Sharpes (per-observation)
    psr_zero: float       # PSR against 0, for context (no deflation)
    n: int


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> PSRResult:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012): the
    probability that the TRUE Sharpe exceeds sr_benchmark, given the sample
    Sharpe, length, and the third/fourth moments. All Sharpes here are
    per-observation (NON-annualised); annualisation must not be applied or the
    moment correction is mis-scaled. With normal returns the denominator
    reduces to sqrt(1 + SR^2/2)."""
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    s = x.std(ddof=1) if n > 1 else 0.0
    if n < 3 or s == 0.0:
        return PSRResult(float("nan"), float("nan"), float("nan"),
                         float("nan"), n, sr_benchmark)
    sr = float(x.mean() / s)
    g3 = float(stats.skew(x, bias=False))
    g4 = float(stats.kurtosis(x, fisher=False, bias=False))   # Pearson (normal=3)
    var = max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr, 1e-12)
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(var)
    return PSRResult(float(stats.norm.cdf(z)), sr, g3, g4, n, sr_benchmark)


def expected_max_sharpe(var_trials: float, n_trials: int) -> float:
    """Bailey-Lopez de Prado expected MAXIMUM Sharpe under the null that all
    n_trials configurations have a true Sharpe of zero and their estimates have
    variance var_trials (per-observation). This is the benchmark the selected
    strategy must beat to survive selection bias. Returns 0 for <=1 trial or
    zero dispersion (no selection effect)."""
    if n_trials <= 1 or var_trials <= 0.0:
        return 0.0
    g = 0.5772156649015329          # Euler-Mascheroni
    z1 = float(stats.norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return math.sqrt(var_trials) * ((1.0 - g) * z1 + g * z2)


def deflated_sharpe_ratio(returns, trial_sharpes=None, *,
                          n_trials: int | None = None,
                          var_trials: float | None = None) -> DSRResult:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): the PSR of the selected strategy."""
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if trial_sharpes is not None:
        ts = np.asarray(trial_sharpes, dtype=float)
        ts = ts[np.isfinite(ts)]
        if var_trials is None:
            var_trials = float(ts.var(ddof=1)) if ts.size > 1 else 0.0
        if n_trials is None:
            n_trials = int(ts.size)
    if n_trials is None or var_trials is None:
        raise ValueError("provide trial_sharpes, or both n_trials and var_trials")
    sr0 = expected_max_sharpe(var_trials, n_trials)
    psr = probabilistic_sharpe_ratio(x, sr_benchmark=sr0)
    psr0 = probabilistic_sharpe_ratio(x, sr_benchmark=0.0)
    return DSRResult(psr.psr, sr0, psr.sr_hat, int(n_trials),
                     float(var_trials), psr0.psr, psr.n)


