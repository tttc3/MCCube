from collections.abc import Callable

import jax.numpy as jnp
import jax.tree_util as jtu

from .._custom_types import Args, CubaturePoints, P, RealScalarLike, XP
from .base import AbstractKernel


class OverdampedLangevinKernel(AbstractKernel):
    r"""Overdamped Langevin SDE step with Euler-Maruyama cubature discretisation.

    Equivalent to a single step of the Euler-Maruyama time-integration scheme for the
    overdamped Langevin equation $Y^{i}_{t+1} := Y_t -\nabla f(Y_t) h + \sqrt{2h}\ e_i$,
    where $\{e_i\}$ are elements of the matrix of cubature_vectors $M$, $h = t_1 - t_0$
    is the Euler-Maruyama step_size, and $-\nabla f$ is the `grad_logdensity_fn`.

    The Overdamped Langevin SDE permits a stationary ergodic solution, that is
    proportional to $\exp{-f}$. Hence it is useful for Bayesian inference/sampling.

    Attributes:
        grad_logdensity_fn: the gradient of the logdensity function given particles $p(t_0)$.
        cubature_paths: callable which returns a matrix $M$ with $n$ rows of
            $d\text{-dimensional}$ cubature vectors.
    """

    grad_logdensity_fn: Callable[[RealScalarLike, P, Args], P]
    cubature_paths: Callable[[RealScalarLike, RealScalarLike], CubaturePoints]

    def transform(
        self,
        t0: RealScalarLike,
        t1: RealScalarLike,
        particles: P,
        args: Args,
    ) -> XP:
        diffusion = jnp.sqrt(2.0) * self.cubature_paths(t0, t1)

        def _update(_particles):
            drift = self.grad_logdensity_fn(t0, particles[:, 0, :], args)[None, ...]
            drift *= t1 - t0
            return particles + drift + diffusion[None, ...]

        return jtu.tree_map(_update, particles)


OverdampedLangevinKernel.__init__.__doc__ = r"""Args:
    grad_logdensity_fn: the gradient of the logdensity function given particles $p(t_0)$.
    cubature_paths: callable which returns a matrix $M$ with $n$ rows of 
            $d\text{-dimensional}$ cubature vectors.
"""
