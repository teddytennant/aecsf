"""Shared adaptive weighted proposal ensemble (13)–(16), Appendix B.3.

Incremental likelihood weights, reverse-particle-guided ancestor probabilities,
stratified resampling with importance correction, and tempered Langevin movement.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aecsf.score import log_sum_exp_normalize, pairwise_sqdist


def incremental_log_weights(log_W, log_g_z, rho_i, rho_minus):
    """log ω_m = log W_m + (ρ_i − ρ^−) log g(y | z_m) from (13)."""
    return log_W + (rho_i - rho_minus) * log_g_z


def incremental_weights(log_W, log_g_z, rho_i, rho_minus):
    """Normalized ω in (13)."""
    return log_sum_exp_normalize(incremental_log_weights(log_W, log_g_z, rho_i, rho_minus))


def ancestor_selection(x, z, log_omega, a, sigma2):
    """Guided ancestor probabilities (14).

    r_{j,m} ∝ K_i(x^{(j)}, z_m) ω_m,   s_m = J^{−1} Σ_j r_{j,m}.
    Returns s:(M,) and r:(J,M).
    """
    sq = pairwise_sqdist(x, a, z)
    log_r = log_omega[None, :] - sq / (2.0 * sigma2)
    r = log_sum_exp_normalize(log_r, axis=-1)
    s = jnp.mean(r, axis=0)
    return s, r


def stratified_resample(key, s):
    """Stratified inverse-CDF ancestors of Appendix B.3.

    U_ℓ ∼ Unif((ℓ−1)/M, ℓ/M),  A_ℓ = min{m : Σ_{r≤m} s_r ≥ U_ℓ}.
    """
    m = s.shape[0]
    cdf = jnp.cumsum(s)
    cdf = cdf / jnp.maximum(cdf[-1], 1e-30)
    u = (jnp.arange(m, dtype=s.dtype) + jax.random.uniform(key, (m,), dtype=s.dtype)) / m
    ancestors = jnp.searchsorted(cdf, u, side="left")
    return jnp.clip(ancestors, 0, m - 1)


def importance_correction(omega, s, ancestors):
    """Descendant weights (15): v_m = ω_{A_m} / s_{A_m}, then normalize."""
    s_a = jnp.maximum(s[ancestors], 1e-30)
    v = omega[ancestors] / s_a
    return v / jnp.maximum(jnp.sum(v), 1e-30)


def resample_proposals(key, z, omega, s):
    """Stratified ancestors plus importance correction (15). Returns z^+, W^+, A."""
    ancestors = stratified_resample(key, s)
    z_new = z[ancestors]
    w_new = importance_correction(omega, s, ancestors)
    return z_new, w_new, ancestors


def forecast_score(z, mean, var):
    """Score of the fitted diagonal-Gaussian forecast (D.3)."""
    return -(z - mean) / var


def langevin_move(key, z, rho, mean, var, grad_log_g, h, n_steps, w_f=1.0, w_g=1.0):
    """Tempered overdamped Langevin (16), L inner steps of size h."""

    def body(z_cur, k):
        drift = w_f * forecast_score(z_cur, mean, var) + w_g * rho * grad_log_g(z_cur)
        xi = jax.random.normal(k, z_cur.shape, dtype=z_cur.dtype)
        z_next = z_cur + 0.5 * h * drift + jnp.sqrt(h) * xi
        return z_next, None

    keys = jax.random.split(key, n_steps)
    z_out, _ = jax.lax.scan(lambda zc, k: body(zc, k), z, keys)
    return z_out


def adapt_proposals(
    key,
    x,
    z,
    log_W,
    log_g_z,
    a,
    sigma2,
    rho_i,
    rho_minus,
    mean,
    var,
    grad_log_g,
    do_resample: bool,
    do_move: bool,
    h: float,
    n_langevin: int,
    w_f: float = 1.0,
    w_g: float = 1.0,
):
    """One adaptive-proposal stage of Algorithm 1.

    Weight update (13) always; optional reverse-particle-guided resampling (14)–(15)
    and Langevin movement (16).
    """
    log_omega = incremental_log_weights(log_W, log_g_z, rho_i, rho_minus)
    omega = log_sum_exp_normalize(log_omega)
    s, r = ancestor_selection(x, z, log_omega, a, sigma2)

    def _resample(args):
        k, z_in, om, sel = args
        z_r, w_r, _ = resample_proposals(k, z_in, om, sel)
        return z_r, w_r

    def _no_resample(args):
        _, z_in, om, _ = args
        return z_in, om

    key, k_res, k_move = jax.random.split(key, 3)
    z_mid, w_mid = jax.lax.cond(
        do_resample, _resample, _no_resample, (k_res, z, omega, s)
    )

    def _move(args):
        k, z_in = args
        return langevin_move(
            k, z_in, rho_i, mean, var, grad_log_g, h, n_langevin, w_f, w_g
        )

    z_out = jax.lax.cond(do_move, _move, lambda args: args[1], (k_move, z_mid))
    log_W_out = jnp.log(jnp.maximum(w_mid, 1e-30))
    return z_out, log_W_out, omega, s, r
