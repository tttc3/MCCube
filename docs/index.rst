MCCube: Markov Chain Cubature via JAX
=====================================

MCCube is a `JAX <https://jax.readthedocs.io>`_ library for constructing Markov Chain 
Cubatures (MCCs) that (weakly) solve certain SDEs, and thus, can be used for performing 
Bayesian inference.

The core features of MCCube are:

- Approximate Bayesian inference of JAX transformable functions (support for PyTorch, Tensorflow and Numpy functions is provided via `Ivy <https://unify.ai/docs/ivy/compiler/transpiler.html>`_);
- A simple Markov chain cubature inference loop;
- A Component framework for constructing Cubature steps as a composition of a Propagator and Recombinator;
- Trace-time component validation that ensures components obey certain expected mathematical properties, with minimal runtime overhead;
- Visualization tools for evaluating and debugging inference/solving performance;
- A `Blackjax <https://blackjax.readthedocs.io/en/latest/>`_-like interface provided by :mod:`mccube.extensions` (**Coming Soon**);
- A custom solver for using MCC in `Diffrax <https://docs.kidger.site/diffrax/>`_, also provided by :mod:`mccube.extensions` (**Coming Soon**). 

In addition, like the samplers in `Blackjax`_, MCCube can easily be integrated with 
probabilistic programming languages (PPLs), as long as they can provide a (potentially 
unnormalized) log-density function.

.. admonition:: Compatibility with non-JAX transformable functions
    :class: tip

    MCCube natively supports operating on JAX transformable functions. If you have 
    functions written in a different automatic differentiation framework, such as 
    PyTorch or Tensorflow, compatibility with MCCube can be achived by transpiling the 
    funtion to JAX with the `Ivy`_ transpiler.

Who should use MCCube?
----------------------
MCCube should appeal to:

- Users of `Blackjax`_ (people who need/want modular GPU/TPU capable samplers);
- Users of `Diffrax`_ (people who need to solve SDEs/CDEs);
- Markov chain cubature researchers/developers.

Installation
------------
.. tab-set::

    .. tab-item:: CPU
        To install the base package:

        .. code-block:: bash

            pip install mccube
        
        If you want all the extras provided in :mod:`mccube.extensions`:

        .. code-block:: bash

            pip install "mccube[extras]"

    .. tab-item:: GPU/TPU

        .. warning::

            By default, a CPU only version of JAX will be installed. To make use of 
            other JAX/XLA compatible accelerators (GPUs/TPUs) please follow 
            `these installation instructions <https://github.com/google/jax#installation>`_.
        
    .. tab-item:: Developers

        .. code-block:: bash

            git clone https://github.com/tttc3/mccube
            cd mccube
            pip install -e ".[dev]"

            # If making changes that you wish to commit
            pre-commit install

            # To verify that all tests pass
            pytest

            # To build the documentation
            pip install -r docs/requirements-docs.txt
            sphinx-build -b html docs docs/_build/html -j auto

Requires Python 3.9+, JAX 0.4.11+, and Equinox 0.10.5+.

Windows support for JAX is currently experimental; WSL2 is the recommended approach for 
using JAX on Windows.

What is Markov chain cubature?
------------------------------
MCC is an approach to constructing a `Cubature on Wiener Space <https://www.jstor.org/stable/4143098>`_ :cite:p:`lyons2004` 
which does not suffer from exponential scaling in time (particle count explosion), thanks 
to the utilization of (partitioned) recombination in the Cubature step/transition kernel.

Quick Example
-------------
The below toy example demonstrates MCCube for inferring the moments of a ten dimensional 
Gaussian, with mean two and diagonal covariance six, given its logdensity function.
More **in-depth examples are coming soon**. 

.. code-block:: python

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

Note that :func:`mccubaturesolve` returns the cubature paths, but does not return any other 
intermediate step information. If such information is required, a 'visualizer' callback
can be used, for example:

.. code-block:: python

    from mccube.visualizers import TensorboardVisualizer

    with TensorboardVisualizer() as tbv:
        cubature = mccubaturesolve(..., visualizer_callback=tbv)

To make use of the Tensorboard visualization suite remember to run the following command
either during/after each experimental run:

.. code-block:: bash

    tensorboard --logdir=experiments

Citation
--------
Please cite this repository if it has been useful in your work:

.. code-block:: bibtex

    @software{mccube2023github,
        author={},
        title={{MCC}ube: Markov chain cubature via {JAX}},
        url={},
        version={<insert current release tag>},
        year={2023},
    }


See Also
--------
Some other Python/JAX packages that you may find interesting:

- `Markov-Chain-Cubature <https://github.com/james-m-foster/markov-chain-cubatur>`_ A PyTorch implementation of Markov Chain Cubature.
- `PySR <https://github.com/MilesCranmer/PySR>`_ High-Performance Symbolic Regression in Python and Julia.
- `Equinox <https://github.com/patrick-kidger/equinox>`_ A JAX library for parameterised functions.
- `Diffrax`_ A JAX library providing numerical differential equation solvers.
- `Lineax <https://github.com/google/lineax>`_ A JAX library for linear solves and linear least squares.
- `OTT-JAX <https://github.com/ott-jax/ott>`_ A JAX library for optimal transport.

.. toctree::
    :hidden:
    :caption: Getting Started

    MCC From Scratch <fromscratch.md>

.. toctree::
    :hidden:
    :caption: Further Resources

    API Reference <api/index.rst>
    Contributing <CONTRIBUTING.md>
    bibliography.rst