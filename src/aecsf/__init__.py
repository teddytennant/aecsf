"""AECSF: adaptive ensemble conditional score filtering (arXiv:2609.32411)."""

from aecsf._host import preload

preload()

from aecsf.dynamics import enkf_update, gaussian_observation_fns, lorenz63_step
from aecsf.filter import AECSFConfig, analysis_update, fit_diagonal_forecast, reverse_euler
from aecsf.proposal import ancestor_selection, incremental_weights, langevin_move
from aecsf.schedule import a_tau, logsnr_grid, r2_tau, sigma2_tau
from aecsf.score import (
    conditional_mean,
    conditional_weights,
    ensemble_score,
    tweedie_score_from_mean,
)

__all__ = [
    "AECSFConfig",
    "a_tau",
    "analysis_update",
    "ancestor_selection",
    "conditional_mean",
    "conditional_weights",
    "enkf_update",
    "ensemble_score",
    "fit_diagonal_forecast",
    "gaussian_observation_fns",
    "incremental_weights",
    "langevin_move",
    "logsnr_grid",
    "lorenz63_step",
    "preload",
    "r2_tau",
    "reverse_euler",
    "sigma2_tau",
    "tweedie_score_from_mean",
]
