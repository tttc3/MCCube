"""Discrete time-integration step for specific diffusion processes (fixed form SDEs)."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from mccube.components.propagators.base import AbstractPropagator


class LangevinDiffusionPropagator(AbstractPropagator):
    r"""An overdamped Langevin diffusion SDE particle Propagator.

    Permits a stationary ergodic solution, that is proportional to $\exp{-f}$. Hence
    is useful for Bayesian inference/sampling from $f$.

    Equivalent to a single step of the Euler-Maruyama time-integration scheme for the
    overdamped Langevin equation $Y^{i}_{t+1} := Y_t -\nabla f(Y_t) h + \sqrt{2h} e_i$,
    where $\{e_i\}$ are elements of the matrix of cubature_vectors $M$, $h$ is the
    Euler-Maruyama `step_size`, and $f$ is a `logdensity`, parameterising the particle
    interaction potential, $Y_t := p(t)$, and $Y^{i}_{t+1} := p^{\prime}(t)$.

    Note that the dimension of the vectors $Y^i$ and the cubature vectors $e_i$ must be
    identical and that for any collection of `cubature_matrix` with $k > 1$,
    $\text{card}(\{Y^{i}_{t+1}\}) > \text{card}(\{Y_t\})$.

    Attributes:
        cubature_matrix: matrix of $d\text{-dimensional}$ cubature vectors.
        step_size: Euler-Maruyama step size $h$.
    """

    cubature_matrix: Float[Array, "k d"]
    step_size: float = 0.1

    def transform(
        self,
        logdensity: Callable[[float, PyTree, PyTree], PyTree],
        time: float,
        particles: PyTree,
        args: PyTree,
    ) -> PyTree:
        dim = particles.shape[-1]
        diffusion = jnp.sqrt(2.0 * self.step_size) * self.cubature_matrix
        drift = jax.vmap(jax.grad(logdensity, argnums=1), [None, 0, None])(
            time, particles, args
        )
        drift *= self.step_size
        return (particles + drift + diffusion[:, None, :]).reshape(-1, dim)
