"""Thomas-algorithm tridiagonal solver."""
from __future__ import annotations

import numpy as np


def _identity_njit(*args, **kwargs):
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


njit, HAVE_NUMBA = _probe_numba()


@njit(cache=True, fastmath=False)
def thomas_solve_inplace(a, b, c, d):  # pragma: no cover - numba/pure-python kernel
    """Solve the tridiagonal system for x, where for row i: a[i]*x[i-1] + b[i]*x[i] +."""
    n = b.shape[0]
    x = np.empty(n, dtype=np.float64)
    # forward elimination
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]
    # back substitution
    x[n - 1] = d[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def thomas_solve(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Non-destructive convenience wrapper around `thomas_solve_inplace`."""
    n = diag.shape[0]
    if not (sub.shape[0] == sup.shape[0] == n - 1 and rhs.shape[0] == n):
        raise ValueError("shape mismatch: need len(sub)=len(sup)=n-1, len(rhs)=n")
    a = np.empty(n, dtype=np.float64)
    c = np.empty(n, dtype=np.float64)
    a[0] = 0.0
    a[1:] = sub
    c[: n - 1] = sup
    c[n - 1] = 0.0
    b = diag.astype(np.float64)  # astype(copy=True) already returns a fresh array
    d = rhs.astype(np.float64)   # so b, d are scratch copies without a second copy
    return thomas_solve_inplace(a, b, c, d)


def _dense(sub, diag, sup):
    return np.diag(diag) + np.diag(sup, 1) + np.diag(sub, -1)


