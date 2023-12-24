from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from diffrax.misc import force_bitcast_convert_type, split_by_tree
from jaxtyping import PRNGKeyArray

from .._custom_types import Args, RealScalarLike, P, XP
from .._utils import nop
from .base import AbstractRecombinationKernel


class MonteCarloKernel(AbstractRecombinationKernel):
    """Monte-Carlo particle sub-sampling/recombination.

    Attributes:
        recombined_count: the particle count that :meth:`transform` should yield.
        key: the base PRNGKey required for Monte Carlo sampling.
        weighting_function: a function, which given a set of particles, generates a set
            of importance-sampling weights. Uniform importance weighting by default.
        with_replacement: if to perform sub-sampling with or without replacement.
    """

    key: PRNGKeyArray
    weighting_function: Callable[[XP], XP] = nop
    with_replacement: bool = False

    def transform(
        self, t0: RealScalarLike, t1: RealScalarLike, particles: XP, args: Args
    ) -> P:
        t0_ = force_bitcast_convert_type(t0, jnp.int32)
        t1_ = force_bitcast_convert_type(t1, jnp.int32)
        key = jr.fold_in(self.key, t0_)
        key = jr.fold_in(key, t1_)
        keys = split_by_tree(key, particles)

        def _choice(_key, _a, _shape):
            _a = jnp.reshape(_a, (-1,) + jnp.shape(_a)[2:])
            _weights = self.weighting_function(_a)
            choice = jr.choice(_key, _a, (_shape[0],), self.with_replacement, _weights)
            return jnp.broadcast_to(choice[:, None, :], _shape)

        return jtu.tree_map(_choice, keys, particles, self.recombined_shape)


MonteCarloKernel.__init__.__doc__ = """Args:
    recombined_count: the particle count that :meth:`transform` should yield.
    key: the base PRNGKey required for Monte Carlo sampling.
    weighting_function: a function, which given a set of particles, generates a set
        of importance-sampling weights. Uniform importance weighting by default.
    with_replacement: if to perform sub-sampling with or without replacement.
"""
