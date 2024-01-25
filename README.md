<div align="center">
<img alt="MCCube logo" src="https://raw.githubusercontent.com/tttc3/MCCube/main/docs/_static/logo.svg"/>
<h1>
    <strong>MCCube</strong></br>
    <em>Markov chain cubature via JAX</em>
</h1>
</div>

<!-- Add the badges in here -->
[![Documentation Status](https://readthedocs.org/projects/mccube/badge/?version=latest)](https://mccube.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/tttc3/MCCube/actions/workflows/tests.yml/badge.svg)](https://github.com/tttc3/MCCube/actions/workflows/tests.yml/)
[![pypi version](https://img.shields.io/pypi/v/mccube.svg)](https://pypi.org/project/mccube/)

MCCube provides the tools for performing Markov chain cubature in [diffrax](https://github.com/patrick-kidger/diffrax).

**Key features:**

- Custom terms, paths, and solvers that provide a painless means to perform MCC in diffrax.
- A small library of recombination kernels, convential cubature formulae, and metrics.

## Installation
To install the base pacakge:
```bash
pip install mccube
```
Requires Python 3.12+, Diffrax 0.5.0+, and Equinox 0.11.3+.

By default, a CPU only version of JAX will be installed. To make use of other JAX/XLA 
compatible accelerators (GPUs/TPUs) please follow [these installation instructions](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier).
Windows support for JAX is currently experimental; WSL2 is the recommended approach for 
using JAX on Windows.

## Documentation
Available at [https://mccube.readthedocs.io/](https://mccube.readthedocs.io/).

## What is Markov chain cubature?
MCC is an approach to constructing a [Cubature on Wiener Space](https://www.jstor.org/stable/4143098) 
which does not suffer from exponential scaling in time (particle count explosion), 
thanks to the utilization of (partitioned) recombination in the (approximate) cubature 
kernel.

### Example
```Python
import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import multivariate_normal

from mccube import (
    GaussianRegion,
    Hadamard,
    LocalLinearCubaturePath,
    MCCSolver,
    MCCTerm,
    MonteCarloKernel,
    gaussian_wasserstein_metric,
)

key = jr.PRNGKey(42)
n, d = 512, 10
t0 = 0.0
epochs = 512
dt0 = 0.05
t1 = t0 + dt0 * epochs
y0 = jnp.ones((n, d))

target_mean = 2 * jnp.ones(d)
target_cov = 3 * jnp.eye(d)


def logdensity(p):
    return multivariate_normal.logpdf(p, mean=target_mean, cov=target_cov)


ode = diffrax.ODETerm(lambda t, p, args: jax.vmap(jax.grad(logdensity))(p))
cde = diffrax.WeaklyDiagonalControlTerm(
    lambda t, p, args: jnp.sqrt(2.0),
    LocalLinearCubaturePath(Hadamard(GaussianRegion(d))),
)
terms = MCCTerm(ode, cde)
solver = MCCSolver(diffrax.Euler(), MonteCarloKernel(n, key=key))

sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0)
res_mean = jnp.mean(sol.ys[-1], axis=0)
res_cov = jnp.cov(sol.ys[-1], rowvar=False)
metric = gaussian_wasserstein_metric((target_mean, res_mean), (target_cov, res_cov))

print(f"Result 2-Wasserstein distance: {metric}")
```

## Citation
Please cite this repository if it has been useful in your work:
```bibtex
@software{mccube2023github,
    author={},
    title={{MCC}ube: Markov chain cubature via {JAX}},
    url={},
    version={<insert current release tag>},
    year={2023},
}
```

## See Also
Some other Python/JAX packages that you may find interesting:

- [Markov-Chain-Cubature](https://github.com/james-m-foster/markov-chain-cubature) A PyTorch implementation of Markov Chain Cubature.
- [PySR](https://github.com/MilesCranmer/PySR) High-Performance Symbolic Regression in Python and Julia.
- [Equinox](https://github.com/patrick-kidger/equinox) A JAX library for parameterised functions.
- [Diffrax](https://github.com/patrick-kidger/diffrax) A JAX library providing numerical differential equation solvers.
- [Lineax](https://github.com/google/lineax) A JAX library for linear solves and linear least squares.
- [OTT-JAX](https://github.com/ott-jax/ott) A JAX library for optimal transport.
