"""Adaptive proposal update uses reverse particles and is not a no-op (13)–(16)."""

import jax
import jax.numpy as jnp
import numpy as np

from aecsf.proposal import (
    ancestor_selection,
    incremental_weights,
    langevin_move,
    adapt_proposals,
)


def test_incremental_weights_change_with_likelihood_tempering():
    log_W = jnp.log(jnp.array([0.2, 0.3, 0.5]))
    log_g = jnp.array([0.0, -2.0, 1.0])
    w0 = incremental_weights(log_W, log_g, rho_i=0.0, rho_minus=0.0)
    np.testing.assert_allclose(w0, jnp.exp(log_W), rtol=1e-10, atol=1e-10)

    w1 = incremental_weights(log_W, log_g, rho_i=0.6, rho_minus=0.0)
    assert not np.allclose(w1, w0, atol=1e-8)
    np.testing.assert_allclose(jnp.sum(w1), 1.0, rtol=1e-10, atol=1e-10)


def test_ancestor_probabilities_depend_on_reverse_particles():
    z = jnp.array([[-2.0], [0.0], [2.0]])
    log_omega = jnp.zeros(3)
    a, sigma2 = 0.5, 0.4
    x_near_right = jnp.array([[a * 2.0], [a * 2.0]])
    x_near_left = jnp.array([[a * -2.0], [a * -2.0]])

    s_right, r_right = ancestor_selection(x_near_right, z, log_omega, a, sigma2)
    s_left, r_left = ancestor_selection(x_near_left, z, log_omega, a, sigma2)

    assert s_right[2] > s_right[0]
    assert s_left[0] > s_left[2]
    assert not np.allclose(s_right, s_left, atol=1e-6)
    np.testing.assert_allclose(jnp.sum(r_right, axis=-1), jnp.ones(2), atol=1e-10)
    np.testing.assert_allclose(jnp.sum(s_right), 1.0, atol=1e-10)


def test_adapt_proposals_changes_weights_using_reverse_particles():
    key = jax.random.PRNGKey(4)
    z = jnp.array([[-2.0], [0.0], [2.0]])
    log_W = jnp.zeros(3) - jnp.log(3.0)
    log_g = jnp.array([-0.5, 0.0, 1.5])
    a, sigma2, rho_i, rho_minus = 0.6, 0.35, 0.4, 0.0
    x = jnp.array([[a * 2.0], [a * 1.8], [a * 2.2]])
    mean = jnp.array([0.0])
    var = jnp.array([1.0])

    def grad_log_g(zz):
        return jnp.zeros_like(zz)

    z_out, log_W_out, omega, s, r = adapt_proposals(
        key,
        x,
        z,
        log_W,
        log_g,
        a,
        sigma2,
        rho_i,
        rho_minus,
        mean,
        var,
        grad_log_g,
        do_resample=True,
        do_move=False,
        h=0.02,
        n_langevin=1,
    )
    w0 = jnp.exp(log_W)
    w_out = jnp.exp(log_W_out)
    # (13) changes the target weights; (14) tilts s_m toward reverse particles.
    assert not np.allclose(omega, w0, atol=1e-8)
    assert not np.allclose(s, omega, atol=1e-6), "guided s_m must use reverse particles"
    assert s[2] > s[0]
    np.testing.assert_allclose(jnp.sum(w_out), 1.0, atol=1e-10)
    np.testing.assert_allclose(jnp.sum(omega), 1.0, atol=1e-10)
    assert z_out.shape == z.shape
    # Stratified ancestors from s_m copy the reverse-particle-favored locations.
    assert float(jnp.mean(z_out)) > float(jnp.mean(z))


def test_langevin_move_is_not_identity():
    key = jax.random.PRNGKey(5)
    z = jnp.array([[0.0], [0.5], [-0.5]])
    mean = jnp.array([1.0])
    var = jnp.array([0.25])

    def grad_log_g(zz):
        return -(zz - 1.5) / 0.25

    z_new = langevin_move(key, z, rho=0.8, mean=mean, var=var, grad_log_g=grad_log_g, h=0.05, n_steps=3)
    assert z_new.shape == z.shape
    assert not np.allclose(z_new, z, atol=1e-8)
