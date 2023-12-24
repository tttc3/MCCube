"""Tools for performing basic Markov-chain cubature inference without `diffrax`."""

import functools
from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.tree_util as jtu
from jax.experimental.host_callback import id_tap
from jaxtyping import PyTree

from ._custom_types import Args, P, RealScalarLike
from ._kernels import AbstractKernel
from ._utils import nop


class MCCubatureState(eqx.Module):
    """Markov-chain cubature state-information container.

    Attributes:
        time: state timecode.
        particles: state particles.
        args: additional arguments passed to the transition kernel.
    """

    time: RealScalarLike
    particles: P
    args: Args


def mccubaturesolve(
    transition_kernels: Sequence[
        AbstractKernel | Callable[[RealScalarLike, RealScalarLike, P, Args], P]
    ],
    initial_particles: P,
    args: Args = None,
    epochs: int = 500,
    t0: RealScalarLike = 0.0,
    dt: RealScalarLike = 0.01,
    callbacks: PyTree[Callable[[MCCubatureState], None]] | None = None,
) -> PyTree[MCCubatureState]:
    """Constant step-size Markov-chain cubature inference loop.

    Args:
        transition_kernels: A Sequence of kernels which are composited from left to
            right to form a single composite Markov-chain transition kernel (a
            presummed approximate cubature kernel).
        initial_particles: The initial condition/state from which to start the chain.
        args: Any additional arguments required by the transition_kerenels.
        epochs: The number of transitions/steps to perform.
        t0: The time origin for the chain (initial time).
        dt: The constant time step-size to take on each transition/step.
        callbacks: A PyTree of callables that are executed at the end of every step,
            including the initialization step. The callables operate on the current
            chain state and should ideally be non-blocking.

    Returns:
        The :class:`MCCubatureState` at every time-step/epoch.
    """
    initial_particles = MCCubatureState(time=t0, particles=initial_particles, args=args)
    composed_kernels = _compose_kernels(transition_kernels)
    _callbacks = callbacks if callbacks is not None else nop
    jtu.tree_map(lambda x: id_tap(x, initial_particles), _callbacks)

    def body_fun(state, _):
        t0 = state.time
        t1 = t0 + dt
        particles = state.particles
        updated_particles = composed_kernels(t0, t1, particles, args)
        updated_state = eqx.tree_at(
            lambda state: [state.time, state.particles],
            state,
            [t1, updated_particles],
        )
        jtu.tree_map(lambda x: id_tap(x, updated_state), _callbacks)
        return updated_state, updated_state

    _, final_state = jax.lax.scan(body_fun, initial_particles, None, length=epochs)
    return final_state


def _compose_kernels(
    kernels: Sequence[
        AbstractKernel | Callable[[RealScalarLike, RealScalarLike, P, Args], P]
    ],
) -> Callable[[RealScalarLike, RealScalarLike, P, Args], P]:
    def _compose(f, g):
        return lambda t0, t1, p, args: g(t0, t1, f(t0, t1, p, args), args)

    if isinstance(kernels, Sequence) and len(kernels) > 1:
        return functools.reduce(_compose, kernels)
    return kernels[0]
