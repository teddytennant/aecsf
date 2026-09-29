"""Particle-specific conditional weights (11), (32) are a softmax and sum to 1."""

import jax
import jax.numpy as jnp
import numpy as np

from aecsf.score import (
    conditional_log_weights,
    conditional_weights,
    log_sum_exp_normalize,
)


def test_conditional_weights_are_softmax_and_sum_to_one():
    key = jax.random.PRNGKey(2)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (5, 3))
    z = jax.random.normal(k2, (7, 3))
    log_W = jnp.log(jax.random.dirichlet(k3, jnp.ones(7)))
    log_g = jnp.array([-0.2, 0.1, -1.3, 0.4, 0.0, -0.7, 0.2])
    a, sigma2, rho = 0.55, 0.42, 0.3

    b = conditional_log_weights(x, z, log_W, a, sigma2, rho, log_g)
    c = conditional_weights(x, z, log_W, a, sigma2, rho, log_g)
    c_softmax = jax.nn.softmax(b, axis=-1)
    c_lse = log_sum_exp_normalize(b, axis=-1)

    np.testing.assert_allclose(c, c_softmax, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(c, c_lse, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(jnp.sum(c, axis=-1), jnp.ones(5), rtol=1e-10, atol=1e-10)
    assert jnp.all(c >= 0.0)
