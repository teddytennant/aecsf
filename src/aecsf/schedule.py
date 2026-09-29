"""Differentiable Gaussian noising schedule and reverse-SDE coefficients.

Paper (49)–(51), Appendix B.5:

    a_τ = 1 − (1 − ε_a) τ
    σ_τ² = ε_b + (1 − ε_b) τ
    f_τ = ∂_τ log a_τ
    r_τ² = ∂_τ σ_τ² − 2 f_τ σ_τ²
    λ(τ) = log(a_τ² / σ_τ²)   (uniform logSNR grid)
"""

from __future__ import annotations

import numpy as np


def a_tau(tau, eps_a: float = 1e-3):
    return 1.0 - (1.0 - eps_a) * tau


def sigma2_tau(tau, eps_b: float = 0.025):
    return eps_b + (1.0 - eps_b) * tau


def logsnr(tau, eps_a: float = 1e-3, eps_b: float = 0.025):
    a = a_tau(tau, eps_a)
    s2 = sigma2_tau(tau, eps_b)
    return 2.0 * np.log(np.maximum(a, 1e-30)) - np.log(np.maximum(s2, 1e-30))


def f_tau(tau, eps_a: float = 1e-3):
    """∂_τ log a_τ = −(1 − ε_a) / a_τ."""
    a = a_tau(tau, eps_a)
    return -(1.0 - eps_a) / np.maximum(a, 1e-30)


def r2_tau(tau, eps_a: float = 1e-3, eps_b: float = 0.025):
    """Diffusion rate r_τ² = ∂_τ σ_τ² − 2 f_τ σ_τ² ≥ 0 under (50)."""
    ds2 = 1.0 - eps_b
    return ds2 - 2.0 * f_tau(tau, eps_a) * sigma2_tau(tau, eps_b)


def _invert_logsnr(target, eps_a: float, eps_b: float, n_iter: int = 80) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if logsnr(mid, eps_a, eps_b) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def logsnr_grid(n_tau: int, eps_a: float = 1e-3, eps_b: float = 0.025) -> np.ndarray:
    """Times τ_i, i = 0..N_τ, with τ_0 = 0, τ_{N_τ} = 1, uniform in logSNR (51)."""
    if n_tau < 1:
        raise ValueError("n_tau must be >= 1")
    lam0 = logsnr(0.0, eps_a, eps_b)
    lam1 = logsnr(1.0, eps_a, eps_b)
    taus = np.empty(n_tau + 1, dtype=np.float64)
    taus[0] = 0.0
    taus[n_tau] = 1.0
    for i in range(1, n_tau):
        # λ(τ_i) = λ(1) + (N_τ − i)/N_τ [λ(0) − λ(1)]
        target = lam1 + (n_tau - i) / n_tau * (lam0 - lam1)
        taus[i] = _invert_logsnr(target, eps_a, eps_b)
    return taus


def reverse_coefficients(tau, eps_a: float = 1e-3, eps_b: float = 0.025):
    """Pack (a, σ², f, r²) at a reverse-time node."""
    return {
        "a": float(a_tau(tau, eps_a)),
        "sigma2": float(sigma2_tau(tau, eps_b)),
        "f": float(f_tau(tau, eps_a)),
        "r2": float(r2_tau(tau, eps_a, eps_b)),
    }
