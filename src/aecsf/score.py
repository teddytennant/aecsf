"""Conditional Tweedie identity and ensemble conditional-mean / score estimator.

Paper (6), (11)–(12), (25), (32), Appendix B.1–B.2, B.6:

    s^a_{n,τ}(x; y) = −x / σ_τ² + (a_τ / σ_τ²) D_{n,τ}(x; y)
    D_{n,τ}(x; y) = E[Z | X_τ = x, Y_n = y]

    b_{j,m} = log W_m − ||x^{(j)} − a z^m||² / (2 σ²) + (1 − ρ) log g(y | z^m)
    c_{j,m} = softmax_m(b_{j,m})
    D̂^{(j)} = Σ_m c_{j,m} z^m
"""

from __future__ import annotations

import jax.numpy as jnp


def pairwise_sqdist(x, a, z):
    """||x_j − a z_m||² via the expansion in Appendix B.6. x:(J,d), z:(M,d) → (J,M)."""
    x2 = jnp.sum(x * x, axis=-1, keepdims=True)
    z2 = jnp.sum(z * z, axis=-1)
    return x2 + (a * a) * z2 - 2.0 * a * (x @ z.T)


def log_sum_exp_normalize(log_w, axis: int = -1):
    """Log-sum-exp softmax of (52)–(54)."""
    c = jnp.max(log_w, axis=axis, keepdims=True)
    unnorm = jnp.exp(log_w - c)
    return unnorm / jnp.sum(unnorm, axis=axis, keepdims=True)


def conditional_log_weights(x, z, log_W, a, sigma2, rho, log_g_z):
    """Particle-specific log coefficients b_{j,m} in (32).

    x:(J,d), z:(M,d), log_W:(M,), log_g_z:(M,) → b:(J,M)
    """
    sq = pairwise_sqdist(x, a, z)
    return log_W[None, :] - sq / (2.0 * sigma2) + (1.0 - rho) * log_g_z[None, :]


def conditional_weights(x, z, log_W, a, sigma2, rho, log_g_z):
    """Normalized c_{j,m} in (11). Rows sum to 1."""
    return log_sum_exp_normalize(
        conditional_log_weights(x, z, log_W, a, sigma2, rho, log_g_z), axis=-1
    )


def conditional_mean(x, z, log_W, a, sigma2, rho, log_g_z):
    """Weighted moment D̂ in (11)–(12), (31). → (J,d)."""
    c = conditional_weights(x, z, log_W, a, sigma2, rho, log_g_z)
    return c @ z


def tweedie_score_from_mean(x, d_mean, a, sigma2):
    """Noisy posterior score from the conditional Tweedie identity (6)/(25)."""
    return -x / sigma2 + (a / sigma2) * d_mean


def ensemble_score(x, z, log_W, a, sigma2, rho, log_g_z):
    """Score estimator (12)/(33): Tweedie identity with ensemble D̂."""
    d_mean = conditional_mean(x, z, log_W, a, sigma2, rho, log_g_z)
    return tweedie_score_from_mean(x, d_mean, a, sigma2), d_mean
