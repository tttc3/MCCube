<div align="center">
<img alt="MCCube logo" src="docs/_static/logo.svg"/>
<h1>
    <strong>MCCube</strong></br>
    <small><em>Markov chain cubature via JAX</em></small>
</h1>
</div>

<!-- Add the badges in here -->

MCCube is a [JAX](https://jax.readthedocs.io) library for constructing Markov Chain 
Cubatures (MCCs) that (weakly) solve certain SDEs, and thus, can be used for performing 
Bayesian inference.

The core features of MCCube are:
- Approximate Bayesian inference of JAX transformable functions (support for PyTorch, Tensorflow and Numpy functions is provided via [Ivy](https://unify.ai/docs/ivy/compiler/transpiler.html));
- A simple Markov chain cubature inference loop;
- A Component framework for constructing Cubature steps as a composition of a Propagator and Recombinator;
- Trace-time component validation that ensures components obey certain expected mathematical properties, with minimal runtime overhead;
- Visualization tools for evaluating and debugging inference/solving performance;
- A [Blackjax](https://blackjax.readthedocs.io/en/latest/)-like interface provided by `mccube.extensions` (**Coming Soon**);
- A custom solver for using MCC in [Diffrax](https://docs.kidger.site/diffrax/), also provided by`mccube.extensions` (**Coming Soon**). 

In addition, like the samplers in [Blackjax](https://blackjax.readthedocs.io/en/latest/), 
MCCube can easily be integrated with probabilistic programming languages (PPLs), as long 
as they can provide a (potentially unnormalized) log-density function.

> [!warning]\
> This package is currently a work-in-progress/experimental. Expect bugs, API instability, and treat all results with a healthy degree of skepticism.

## Who should use MCCube?
MCCube should appeal to:
- Users of [Blackjax](https://github.com/blackjax-devs/blackjax#who-should-use-blackjax) (people who need/want modular GPU/TPU capable samplers);
- Users of [Diffrax](https://github.com/patrick-kidger/diffrax) (people who need to solve SDEs/CDEs);
- Markov chain cubature researchers/developers.

## Installation
To install the base pacakge:
```bash
pip install mccube
```
If you want all the extras provided in `mccube.extensions`:
```bash
pip install mccube[extras]
```

Requires Python 3.9+, JAX 0.4.11+, and Equinox 0.10.5+.

By default, a CPU only version of JAX will be installed. To make use of other JAX/XLA 
compatible accelerators (GPUs/TPUs) please follow [these installation instructions](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier).
Windows support for JAX is currently experimental; WSL2 is the recommended approach for 
using JAX on Windows.

## Documentation
**Coming soon** at [https://mccube.readthedocs.io/](https://mccube.readthedocs.io/).

## What is Markov chain cubature?
MCC is an approach to constructing a [Cubature on Wiener Space](https://www.jstor.org/stable/4143098) which does not suffer from exponential scaling in time (particle count explosion), thanks to the utilization of (partitioned) recombination in the Cubature step/transition kernel.

## Quick Example
The below toy example demonstrates MCCube for inferring the moments of a ten dimensional 
Gaussian, with mean two and diagonal covariance six, given its logdensity function.
More **in-depth examples are coming soon**. 

```Python
import jax
import numpy as np
from jax.scipy.stats import multivariate_normal

from mccube import MCCubatureStep, mccubaturesolve, minimal_cubature_formula
from mccube.components import LangevinDiffusionPropagator, MonteCarloRecombinator
from mccube.metrics import cubature_target_error

# Setup the problem.
n_particles = 8192
target_dimension = 2

rng = np.random.default_rng(42)
prior_particles = rng.uniform(size=(n_particles, target_dimension))
target_mean = 2 * np.ones(target_dimension)
target_cov = 3 * np.diag(target_mean)


# MCCube expects the logdensity to have call signature (t, p(t), args), allowing the
# density to be time dependant, or to rely on some other generic args.
# Note: You can obtain significantly better performance by defining a custom jvp here.
def target_logdensity(t, p, args):
    return multivariate_normal.logpdf(p, target_mean, target_cov)


# Setup the MCCubature.
recombinator_key = jax.random.PRNGKey(42)
cfv = minimal_cubature_formula(target_dimension, degree=3).vectors
cs = MCCubatureStep(
    propagator=LangevinDiffusionPropagator(cfv),
    recombinator=MonteCarloRecombinator(recombinator_key),
)

# Construct the MCCubature/solve for the MCCubature paths.
mccubature_paths = mccubaturesolve(
    logdensity=target_logdensity,
    transition_kernel=cs,
    initial_particles=prior_particles,
)

# Compare mean and covariance of the inferred cubature to the target.
posterior_particles = mccubature_paths.particles[-1, :, :]
mean_err, cov_err = cubature_target_error(posterior_particles, target_mean, target_cov)
print(f"Mean Error: {mean_err}\n", f"Cov Error: {cov_err}")
```

Note that `mccubaturesolve` returns the cubature paths, but does not return any other 
intermediate step information. If such information is required, a 'visualizer' callback
can be used, for example:

```python
from mccube.extensions.visualizers import TensorboardVisualizer

with TensorboardVisualizer() as tbv:
    cubature = mccubaturesolve(..., visualization_callback=tbv)
```

To make use of the Tensorboard visualization suite remember to run the following command
either during/after each experimental run:

```bash
tensorboard --logdir=experiments
```

## Citation
Please cite this repository if it has been useful in your work:
```
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

- [PySR](https://github.com/MilesCranmer/PySR) High-Performance Symbolic Regression in Python and Julia.
- [Equinox](https://github.com/patrick-kidger/equinox) A JAX library for parameterised functions.
- [Diffrax](https://github.com/patrick-kidger/diffrax) A JAX library providing numerical differential equation solvers.
- [Lineax](https://github.com/google/lineax) A JAX library for linear solves and linear least squares.
- [OTT-JAX](https://github.com/ott-jax/ott) A JAX library for optimal transport.