JAX kernel of AECSF adaptive ensemble conditional score filtering (arXiv:2609.32411).

## Install

```
python3 -m venv .venv
.venv/bin/pip install -e ".[tests]"
```

## Run

```
.venv/bin/python -m pytest -q
.venv/bin/python train.py
```

## What does not match the paper

The paper's high-dimensional Gaussian, Lorenz-96, and Kuramoto-Sivashinsky tables are not reproduced. Reverse-step counts, ensemble sizes, and Langevin settings here are tiny CPU values, not Table 4. The forecast score is the diagonal-Gaussian fit of Appendix D.3 only; no full-covariance or localization. No GPU kernels, no weather models, and no learned score network (the method is training-free). Endpoint reverse particles retain the schedule's residual noise σ_0² = ε_b rather than a denoising projection.
