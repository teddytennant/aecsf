"""Reverse-diffusion endpoint on a 1D Gaussian observation (17), Algorithm 1."""

import jax
import jax.numpy as jnp
import numpy as np

from aecsf.dynamics import gaussian_observation_fns
from aecsf.filter import AECSFConfig, analysis_update, reverse_euler
from aecsf.score import tweedie_score_from_mean


def test_reverse_euler_finite_and_uses_score():
    key = jax.random.PRNGKey(6)
    x = jnp.array([[1.0], [0.0], [-1.0]])
    # Score of N(0.5, 1): s = -(x - 0.5)
    score = -(x - 0.5)
    x_new = reverse_euler(x, score, tau=0.8, delta=0.1, eps_a=1e-3, eps_b=0.025, key=key)
    assert x_new.shape == x.shape
    assert jnp.all(jnp.isfinite(x_new))
    x_zero = reverse_euler(
        x, jnp.zeros_like(score), tau=0.8, delta=0.1, eps_a=1e-3, eps_b=0.025, key=key
    )
    assert not np.allclose(x_new, x_zero, atol=1e-8)


def test_tweedie_score_pulls_toward_conditional_mean():
    x = jnp.array([[0.0]])
    d_mean = jnp.array([[2.0]])
    a, sigma2 = 0.8, 0.5
    s = tweedie_score_from_mean(x, d_mean, a, sigma2)
    # s = -x/σ² + (a/σ²) D; at x=0 this has the sign of D
    assert s[0, 0] > 0.0


def test_reverse_sampler_endpoint_closer_to_posterior_than_prior():
    """1D N(0,1) prior, y = z + N(0, 0.25); posterior mean 1.6, prior mean 0."""
    key = jax.random.PRNGKey(7)
    y = jnp.array([2.0])
    r = 0.25
    ma = (1.0 / r) * 2.0 / (1.0 + 1.0 / r)
    assert abs(ma - 1.6) < 1e-12

    j = 24
    key, k_ens = jax.random.split(key)
    forecast = jax.random.normal(k_ens, (j, 1))
    H = jnp.array([[1.0]])
    log_g, grad_log_g = gaussian_observation_fns(y, H, r)
    cfg = AECSFConfig(n_tau=8, n_langevin=2, langevin_h=0.05, eps_a=1e-3, eps_b=0.025)

    x_a, _, _ = analysis_update(key, forecast, log_g, grad_log_g, cfg)
    est = float(jnp.mean(x_a))
    rmse_post = abs(est - ma)
    rmse_prior = abs(0.0 - ma)
    assert rmse_post < rmse_prior
    assert jnp.all(jnp.isfinite(x_a))
