"""Reverse-diffusion analysis update (Algorithm 1).

Forecast ensemble → shared adaptive proposal + reverse particles → analysis ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from aecsf.proposal import adapt_proposals
from aecsf.schedule import a_tau, logsnr_grid, sigma2_tau
from aecsf.score import ensemble_score


@dataclass
class AECSFConfig:
    n_tau: int = 8
    eps_a: float = 1e-3
    eps_b: float = 0.025
    n_langevin: int = 3
    langevin_h: float = 0.02
    w_f: float = 1.0
    w_g: float = 1.0
    resample: bool = True
    move: bool = True


def fit_diagonal_forecast(ensemble, jitter: float = 1e-5, floor: float = 1e-3):
    """Diagonal Gaussian forecast fit with the variance regularizer of D.3.

    v_q = max( max(M^{−1} Σ (X_{m q} − m_q)², 10^{−12}) + 10^{−5}, 10^{−3} )
    """
    mean = jnp.mean(ensemble, axis=0)
    var = jnp.mean((ensemble - mean) ** 2, axis=0)
    var = jnp.maximum(jnp.maximum(var, 1e-12) + jitter, floor)
    return mean, var


def init_reverse_particles(key, n_particles: int, dim: int, dtype=jnp.float32):
    """Standardized N(0, I) reverse-particle start of Appendix B.5.

    Independent standard-normal entries, then componentwise zero mean and unit
    sample standard deviation across the J particles.
    """
    x = jax.random.normal(key, (n_particles, dim), dtype=dtype)
    mean = jnp.mean(x, axis=0, keepdims=True)
    std = jnp.std(x, axis=0, keepdims=True, ddof=1)
    return (x - mean) / jnp.maximum(std, 1e-8)


def reverse_euler(x, score, tau, delta, eps_a, eps_b, key):
    """One Euler–Maruyama reverse step (17).

    x_{i−1} = x_i − Δ_i [f_{τ_i} x_i − r_{τ_i}² ŝ] + √(Δ_i r_{τ_i}²) ξ
    """
    a = 1.0 - (1.0 - eps_a) * tau
    s2 = eps_b + (1.0 - eps_b) * tau
    f = -(1.0 - eps_a) / jnp.maximum(a, jnp.array(1e-30, dtype=x.dtype))
    r2 = (1.0 - eps_b) - 2.0 * f * s2
    drift = f * x - r2 * score
    xi = jax.random.normal(key, x.shape, dtype=x.dtype)
    return x - delta * drift + jnp.sqrt(jnp.maximum(delta * r2, 0.0)) * xi


def _stage_flags(i: int, n_tau: int, resample: bool, move: bool):
    """Resampling / movement schedule of Appendix D.3.

    Remaining T−1 stages numbered from 0. Resampling at even stages; movement
    at stages 0, …, T−3. Final score stage (i=1) resamples but does not move.
    """
    # i runs N_τ, …, 1. Adaptation is skipped at i = N_τ.
    stage = n_tau - 1 - i
    do_resample = bool(resample) and (stage % 2 == 0)
    do_move = bool(move) and (stage <= n_tau - 3)
    return do_resample, do_move


def analysis_update(
    key,
    forecast,
    log_likelihood,
    grad_log_likelihood,
    config: AECSFConfig | None = None,
    proposals=None,
):
    """One assimilation cycle of Algorithm 1.

    forecast:(J,d) is the forecast ensemble (also the initial proposal if
    ``proposals`` is None). ``log_likelihood(z)`` and ``grad_log_likelihood(z)``
    take a (M,d) batch and return (M,) and (M,d).
    """
    if config is None:
        config = AECSFConfig()
    forecast = jnp.asarray(forecast)
    j, dim = forecast.shape
    z = forecast if proposals is None else jnp.asarray(proposals)
    m = z.shape[0]
    log_W = jnp.full((m,), -jnp.log(m), dtype=forecast.dtype)
    mean, var = fit_diagonal_forecast(forecast)

    taus = logsnr_grid(config.n_tau, config.eps_a, config.eps_b)
    key, k_init = jax.random.split(key)
    x = init_reverse_particles(k_init, j, dim, dtype=forecast.dtype)

    rho_minus = 0.0
    n_tau = config.n_tau
    for step in range(n_tau):
        i = n_tau - step
        tau_i = float(taus[i])
        tau_im1 = float(taus[i - 1])
        delta = tau_i - tau_im1
        a = float(a_tau(tau_i, config.eps_a))
        sigma2 = float(sigma2_tau(tau_i, config.eps_b))
        rho_i = 1.0 - tau_i

        key, k_adapt, k_rev = jax.random.split(key, 3)
        if i < n_tau:
            do_resample, do_move = _stage_flags(
                i, n_tau, config.resample, config.move
            )
            z, log_W, _, _, _ = adapt_proposals(
                k_adapt,
                x,
                z,
                log_W,
                log_likelihood(z),
                a,
                sigma2,
                rho_i,
                rho_minus,
                mean,
                var,
                grad_log_likelihood,
                do_resample,
                do_move,
                config.langevin_h,
                config.n_langevin,
                config.w_f,
                config.w_g,
            )
            rho_minus = rho_i

        score, _ = ensemble_score(
            x, z, log_W, a, sigma2, rho_i, log_likelihood(z)
        )
        x = reverse_euler(
            x, score, tau_i, delta, config.eps_a, config.eps_b, k_rev
        )
    return x, z, log_W


def analysis_mean(key, forecast, log_likelihood, grad_log_likelihood, config=None):
    x, _, _ = analysis_update(
        key, forecast, log_likelihood, grad_log_likelihood, config
    )
    return jnp.mean(x, axis=0)
