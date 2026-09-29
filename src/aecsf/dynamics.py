"""Tiny Lorenz-63 dynamics, linear-Gaussian observations, and an EnKF baseline."""

from __future__ import annotations

import jax
import jax.numpy as jnp


LORENZ_SIGMA = 10.0
LORENZ_RHO = 28.0
LORENZ_BETA = 8.0 / 3.0


def lorenz63_f(state):
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    dx = LORENZ_SIGMA * (y - x)
    dy = x * (LORENZ_RHO - z) - y
    dz = x * y - LORENZ_BETA * z
    return jnp.stack([dx, dy, dz], axis=-1)


def rk4_step(state, dt):
    k1 = lorenz63_f(state)
    k2 = lorenz63_f(state + 0.5 * dt * k1)
    k3 = lorenz63_f(state + 0.5 * dt * k2)
    k4 = lorenz63_f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def lorenz63_step(state, dt, n_sub: int = 1):
    def body(s, _):
        return rk4_step(s, dt), None

    out, _ = jax.lax.scan(body, state, None, length=n_sub)
    return out


def gaussian_observation_fns(y, H, obs_var):
    """log g(y|z) and ∇_z log g for y = H z + N(0, obs_var I), constants dropped."""
    y = jnp.asarray(y)
    H = jnp.asarray(H)

    def log_g(z):
        resid = y - z @ H.T
        return -0.5 * jnp.sum(resid * resid, axis=-1) / obs_var

    def grad_log_g(z):
        resid = y - z @ H.T
        return (resid / obs_var) @ H

    return log_g, grad_log_g


def enkf_update(key, ensemble, y, H, obs_var, inflation: float = 1.0):
    """Stochastic EnKF with identity (or general H) observations."""
    j, _ = ensemble.shape
    x = ensemble
    mean = jnp.mean(x, axis=0)
    anom = (x - mean) * inflation
    p = (anom.T @ anom) / jnp.maximum(j - 1, 1)
    r = obs_var * jnp.eye(H.shape[0], dtype=x.dtype)
    s = H @ p @ H.T + r
    k_gain = jnp.linalg.solve(s, H @ p).T
    noise = jax.random.normal(key, (j, H.shape[0]), dtype=x.dtype) * jnp.sqrt(obs_var)
    innov = y[None, :] + noise - x @ H.T
    return x + innov @ k_gain.T
