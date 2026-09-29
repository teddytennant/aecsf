"""CPU smoke: linear-Gaussian and Lorenz-63 DA, AECSF vs EnKF RMSE."""

from aecsf._host import preload

preload()

import jax
import jax.numpy as jnp

from aecsf.dynamics import enkf_update, gaussian_observation_fns, lorenz63_step
from aecsf.filter import AECSFConfig, analysis_update


def _rmse(ens, truth):
    return float(jnp.sqrt(jnp.mean((jnp.mean(ens, axis=0) - truth) ** 2)))


def linear_gaussian_smoke(key):
    a_dyn, q_var, r_var = 0.8, 0.15, 0.25
    n_cycles, j = 6, 16
    cfg = AECSFConfig(n_tau=16, n_langevin=2, langevin_h=0.04)
    H = jnp.array([[1.0]])

    key, k_t, k_e = jax.random.split(key, 3)
    truth = jax.random.normal(k_t, ())
    ens_a = jax.random.normal(k_e, (j, 1))
    ens_e = ens_a
    err_a, err_e = [], []
    for _ in range(n_cycles):
        key, k_q, k_r, k_a, k_enkf, k_qa, k_qe = jax.random.split(key, 7)
        truth = a_dyn * truth + jnp.sqrt(q_var) * jax.random.normal(k_q)
        y = truth + jnp.sqrt(r_var) * jax.random.normal(k_r)
        ens_a = a_dyn * ens_a + jnp.sqrt(q_var) * jax.random.normal(k_qa, ens_a.shape)
        ens_e = a_dyn * ens_e + jnp.sqrt(q_var) * jax.random.normal(k_qe, ens_e.shape)
        log_g, grad_g = gaussian_observation_fns(jnp.array([y]), H, r_var)
        ens_a, _, _ = analysis_update(k_a, ens_a, log_g, grad_g, cfg)
        ens_e = enkf_update(k_enkf, ens_e, jnp.array([y]), H, r_var)
        tvec = jnp.array([truth])
        err_a.append(_rmse(ens_a, tvec))
        err_e.append(_rmse(ens_e, tvec))
    return float(jnp.mean(jnp.array(err_a))), float(jnp.mean(jnp.array(err_e)))


def lorenz63_smoke(key):
    dt, n_sub, n_cycles, j = 0.02, 4, 4, 8
    obs_var = 2.0
    H = jnp.eye(3)
    cfg = AECSFConfig(n_tau=12, n_langevin=2, langevin_h=0.02, eps_a=5e-4, eps_b=0.0125)

    key, k_t, k_e = jax.random.split(key, 3)
    truth = jnp.array([1.0, 1.0, 1.0])
    for _ in range(40):
        truth = lorenz63_step(truth, dt, n_sub=1)
    ens_a = truth + 0.5 * jax.random.normal(k_e, (j, 3))
    ens_e = ens_a
    err_a, err_e = [], []
    for _ in range(n_cycles):
        key, k_r, k_a, k_enkf = jax.random.split(key, 4)
        truth = lorenz63_step(truth, dt, n_sub=n_sub)
        y = truth + jnp.sqrt(obs_var) * jax.random.normal(k_r, (3,))
        ens_a = jax.vmap(lambda s: lorenz63_step(s, dt, n_sub=n_sub))(ens_a)
        ens_e = jax.vmap(lambda s: lorenz63_step(s, dt, n_sub=n_sub))(ens_e)
        log_g, grad_g = gaussian_observation_fns(y, H, obs_var)
        ens_a, _, _ = analysis_update(k_a, ens_a, log_g, grad_g, cfg)
        ens_e = enkf_update(k_enkf, ens_e, y, H, obs_var, inflation=1.05)
        err_a.append(_rmse(ens_a, truth))
        err_e.append(_rmse(ens_e, truth))
    return float(jnp.mean(jnp.array(err_a))), float(jnp.mean(jnp.array(err_e)))


def main():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    lg_a, lg_e = linear_gaussian_smoke(k1)
    l63_a, l63_e = lorenz63_smoke(k2)
    print(f"linear-Gaussian AECSF RMSE {lg_a:.4f}  EnKF RMSE {lg_e:.4f}")
    print(f"Lorenz-63 AECSF RMSE {l63_a:.4f}  EnKF RMSE {l63_e:.4f}")


if __name__ == "__main__":
    main()
