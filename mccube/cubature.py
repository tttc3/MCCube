"""Markov Chain Cubature (MCC) construction/inference/solving.

This module provides the tools for constructing SDE cubatures via MCC.

An SDE cubature (alternatively called a Cubature on Wiener Space :cite:p:`lyons2004`) 
is an approximate discrete-time integrator that exactly integrates weak-solutions 
(moments up to degree $N$) for some class of SDEs. This works by reducing the problem 
of integrating the SDE with respect to its infinitely many strong solution paths, to 
obtain some expected path, to one of taking discrete sums of finitely many, so called 
cubature paths.

In measure terminology, an SDE cubature replaces the SDE's Wiener measure with a 
discrete measure, supported on finitely many bounded variation paths, called a cubature 
measure/cubature paths (essentially reducing the problem to one of solving many ODEs 
instead of a single SDE). See :cite:t:`crisan2017cubature` for a nice introduction to 
the method, and :cite:t:`lyons2004` for a comprehensive investigation.

The salient limitation of SDE cubatures constructed as per :cite:t:`lyons2004`
is that the paths count scales exponentially with the number of discrete time steps 
($\mathcal{O}(n^{m})$, where $n$ is the propagator expansion factor, and $m$ is the 
number of time-integration steps). MCC solves this problem by constructing the 
collection of paths as a markov chain, wherethe :class:`MCCubatureStep` acts as a 
transition kernel that employs recombination to maintain the path/particle count at 
every time step. Note that in MCCube the paths are usually interpreted as particle 
trajectories, as this provides a consistent physically analogy.
"""

from __future__ import annotations

from typing import Callable

import chex
import equinox as eqx
import jax
from jax.experimental.host_callback import id_tap
from jaxtyping import Array, ArrayLike, Float, PyTree

from mccube.components import (
    AbstractPropagator,
    AbstractRecombinator,
    WrappedPropagator,
    WrappedRecombinator,
)
from mccube.utils import no_operation


class MCCubatureState(eqx.Module):
    time: float
    particles: PyTree[Float[Array, "n d"]]
    args: PyTree


class MCCubatureStep(AbstractPropagator):
    r"""Shape preserving (non-expanding) MCCubature step/transition kernel.

    `MCCubatureStep` automatically computes the recombination factor such that the
    composition of Propagator and Recombinator results in no change to particle count
    or dimension. Also implements a concrete validate method to enforce the following
    properties of the composed transformation:

    1. Particle PyTree structure is conserved.
    2. Particle count, rank, and dimensionality are conserved.

    Attributes:
        propagator: a Propagator component.
        recombinator: a Recombinator component.
    """
    propagator: AbstractPropagator = WrappedPropagator()
    recombinator: AbstractRecombinator = WrappedRecombinator()

    def transform(
        self,
        logdensity: Callable[[float, PyTree, PyTree], PyTree],
        time: float,
        particles: PyTree[Float[ArrayLike, "n d"]],
        args: PyTree,
    ) -> PyTree[Float[Array, "n d"]]:
        propagated_particles = self.propagator(logdensity, time, particles, args)
        recombination_factor = propagated_particles.shape[0] // particles.shape[0]
        recombinant_particles = self.recombinator(
            recombination_factor, time, propagated_particles, args
        )
        return recombinant_particles

    def validate(self, transform):
        r"""Validate the transform.

        Ensures that the transform obeys the properties:

        1. The transformation **must conserve the particle PyTree structure** $P$.
        2. The transformation **must conserve particle count and dimensionality,
        $p(t) \in \mathbb{R}^{n \times d}$ and $p^{\prime}(t) \in \mathbb{R}^{n \times d}$**.

        Args:
            transform: transform to validate, $h(g, t, p(t), args)$.

        Returns:
            Valid transform $(v \circ f)(g, t, p(t), args)$.
        """  # noqa: E501

        def valid_step(
            logdensity: Callable[[float, PyTree, PyTree], PyTree],
            time: float,
            particles: PyTree,
            args: PyTree,
        ) -> PyTree:
            updated_particles = transform(logdensity, time, particles, args)
            chex.assert_trees_all_equal_shapes(
                updated_particles,
                particles,
                custom_message=(
                    "Cubature step must return a particle PyTree with the same shape as"
                    " the input particle PyTree."
                ),
            )
            return updated_particles

        return valid_step


def mccubaturesolve(
    logdensity: Callable[[float, PyTree, PyTree], PyTree],
    transition_kernel: MCCubatureStep | AbstractPropagator,
    initial_particles: Float[Array, "n d"],
    args: PyTree | None = None,
    epochs: int = 500,
    *,
    t0: float = 0.0,
    dt: float = 1.0,
    visualization_callback: Callable[[MCCubatureState], None] | None = None,
) -> PyTree[MCCubatureState]:
    r"""Fixed-timestep MCCubature construction/inference/solution loop.

    Constructs and evaluates an MCCubature by iterative application of a
    `transition_kernel`/:class:`MCCubatureStep`.

    For a suitable SDE (Propagator), :func:`mccubaturesolve` generates a collection of
    paths as a discrete-time Markov chain, where the `MCCubatureStep` defines the state
    transition kernel $h(g, t, p(t), args)$, `logdensity` defines the (unnormalized)
    density of the distribution $g(t, p(t), args)$ that parametrizes the SDE, `epochs`
    defines the number of discrete time steps $dt$, and `prior_samples` defines the
    chain starting state $p(t0)$.

    When used for Bayesian inference (for example when the SDE/Propagator is the
    :class:`LangevinDiffusionPropagator`), the initial state of the chain $p(t_0)$
    is the set of prior samples, the logdensity $g$ is consistent with the posterior
    (the distribution one wishes to sample from), and the final state of the chain
    $p(t_0 + dt * \text{epochs})$ is the set of posterior samples.W

    See :mod:`mccube.extensions.solvers` for a more general approach to solving
    SDEs via MCCubature.

    Args:
        logdensity: the log-density of the distribution that parametrizes the
            SDE/Propagator (in the case of Bayesian inference this is the distribution
            one wishes to sample from).
        transition_kernel: the MCCubature/integration step.
        epochs: the number of cubature steps (discrete timesteps) to take.
        initial_particles: the particles to use as the initial state of the Markov chain
            (in the case of Bayesian inference these are samples from the prior).
        args: additional static arguments passed to the log-density function.
        t0: the timestamp of the seed event for the Markov Chain.
        dt: the fixed-timestep between events in the Markov Chain.
        visualization_callback: can be used to perform in-process visualisation of the
            `CubatureState` at each epoch.

    Returns:
        The :class:`MCCubatureState` at every timestep (collection of evaluated
        cubature paths).

        Those familiar with cubature formulae may wish to think of the particles in
        each :class:`MCCubatureState` as a collection of cubature vectors $v_i$, whose
        cubature coefficients are all $B_i = 1/n$. For weak quantities such as the mean,
        \int x dP(x) \approx \frac{1}{n} \sum_i^{n}, the discrete estimator is identical
        to the implied cubature formula $\sum_i^{n} B_i f(v)$, for $f(v) = v$.
    """
    initial_particles = MCCubatureState(time=t0, particles=initial_particles, args=args)
    visualization_callback = visualization_callback or no_operation  # Handle None case
    id_tap(visualization_callback, initial_particles)

    def body_fun(state, _):
        time = state.time
        particles = state.particles
        updated_particles = transition_kernel(logdensity, time, particles, args)
        updated_state = eqx.tree_at(
            lambda state: [state.time, state.particles],
            state,
            [time + dt, updated_particles],
        )
        id_tap(visualization_callback, updated_state)
        return updated_state, updated_state

    _, final_state = jax.lax.scan(body_fun, initial_particles, None, length=epochs)
    return final_state
