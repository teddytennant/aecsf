"""Conditional Tweedie identity (6)/(25) and exactness of the weighted estimator."""

import jax
import jax.numpy as jnp
import numpy as np

from aecsf.score import (
    conditional_mean,
    ensemble_score,
    pairwise_sqdist,
    tweedie_score_from_mean,
)


def _gauss_posterior(m0, v0, y, r, h=1.0):
    prec = 1.0 / v0 + (h * h) / r
    va = 1.0 / prec
    ma = va * (m0 / v0 + h * y / r)
    return ma, va


def _tempered_gauss(m0, v0, y, r, rho, h=1.0):
    prec = 1.0 / v0 + rho * (h * h) / r
    v = 1.0 / prec
    m = v * (m0 / v0 + rho * h * y / r)
    return m, v


def test_tweedie_identity_matches_analytic_gaussian_score():
    m0, v0, y, r = 0.0, 1.0, 1.5, 0.25
    a, sigma2 = 0.6, 0.4
    ma, va = _gauss_posterior(m0, v0, y, r)
    x = jnp.array([[0.2], [0.8], [-0.4]])

    # E[Z | X=x, Y=y]: Kalman update of N(ma, va) with obs x = a Z + N(0, σ²)
    prec_d = 1.0 / va + (a * a) / sigma2
    vd = 1.0 / prec_d
    d_mean = vd * (ma / va + a * x[:, 0] / sigma2)
    d_mean = d_mean[:, None]

    s_tweedie = tweedie_score_from_mean(x, d_mean, a, sigma2)
    var_x = a * a * va + sigma2
    s_analytic = -(x - a * ma) / var_x
    np.testing.assert_allclose(s_tweedie, s_analytic, rtol=1e-10, atol=1e-10)


def test_ensemble_score_matches_analytic_when_proposal_is_tempered():
    """Monte Carlo from π_ρ ≈ true tempered law; score ≈ analytic noisy posterior."""
    key = jax.random.PRNGKey(0)
    m0, v0, y, r = 0.0, 1.0, 1.2, 0.36
    a, sigma2, rho = 0.5, 0.5, 0.5
    ma, va = _gauss_posterior(m0, v0, y, r)
    m_rho, v_rho = _tempered_gauss(m0, v0, y, r, rho)

    m = 4000
    z = m_rho + jnp.sqrt(v_rho) * jax.random.normal(key, (m, 1))
    log_W = jnp.full((m,), -jnp.log(m))
    log_g = -0.5 * ((y - z[:, 0]) ** 2) / r
    x = jnp.array([[0.1], [0.4], [-0.2]])

    s_hat, d_hat = ensemble_score(x, z, log_W, a, sigma2, rho, log_g)

    prec_d = 1.0 / va + (a * a) / sigma2
    vd = 1.0 / prec_d
    d_true = (vd * (ma / va + a * x[:, 0] / sigma2))[:, None]
    var_x = a * a * va + sigma2
    s_true = -(x - a * ma) / var_x

    np.testing.assert_allclose(d_hat, d_true, rtol=0.08, atol=0.08)
    np.testing.assert_allclose(s_hat, s_true, rtol=0.08, atol=0.08)


def test_true_conditional_proposal_recovers_exact_discrete_mean():
    """If W_m ∝ π^f g^ρ on a finite support, D̂ is the exact discrete E[Z|X,Y]."""
    z = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.5])[:, None]
    m0, v0, y, r = 0.0, 1.0, 1.0, 0.5
    a, sigma2, rho = 0.7, 0.3, 0.4
    x = jnp.array([[0.35], [-0.1]])

    log_prior = -0.5 * (z[:, 0] - m0) ** 2 / v0
    log_g = -0.5 * (y - z[:, 0]) ** 2 / r
    log_W = log_prior + rho * log_g
    log_W = log_W - jax.nn.logsumexp(log_W)

    d_hat = conditional_mean(x, z, log_W, a, sigma2, rho, log_g)

    # Exact discrete posterior: p(z|x,y) ∝ π^f(z) φ_σ²(x − a z) g(y|z)
    sq = pairwise_sqdist(x, a, z)
    log_post = log_prior[None, :] - sq / (2.0 * sigma2) + log_g[None, :]
    w = jax.nn.softmax(log_post, axis=-1)
    d_exact = w @ z
    np.testing.assert_allclose(d_hat, d_exact, rtol=1e-10, atol=1e-10)
