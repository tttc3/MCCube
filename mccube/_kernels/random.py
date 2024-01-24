from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from diffrax._misc import force_bitcast_convert_type, split_by_tree
from jaxtyping import PRNGKeyArray

from .._custom_types import (
    Args,
    Particles,
    PartitionedParticles,
    RealScalarLike,
    RecombinedParticles,
    Weights,
)
from .._utils import nop, unpack_particles
from .base import AbstractPartitioningKernel, AbstractRecombinationKernel


class MonteCarloKernel(AbstractRecombinationKernel):
    r"""Monte Carlo particle sub-sampling/recombination.

    Example:
        ```python
        import jax.numpy as jnp
        import jax.random as jr

        key = jr.PRNGKey(42)
        kernel = mccube.MonteCarloKernel({"y": 3}, key=key)
        y0 = {"y": jnp.ones((10,2))}
        result = kernel(..., y0, ...)
        # {"y": jnp.ones((3,2))}
        ```

    Attributes:
        recombination_count: indicates the requested size of the recombined dimension.
        with_replacement: if to perform sub-sampling with or without replacement.
        weighting_function: allows particle weights to be transformed. If the transform
            returns :code:`None` then the weights are assumed/implicitly uniform.
        key: the base PRNGKey required for Monte Carlo sampling.
    """

    with_replacement: bool = False
    weighting_function: Callable[[Weights], Weights | None] = nop
    key: PRNGKeyArray = eqx.field(kw_only=True)

    def __call__(
        self,
        t: RealScalarLike,
        particles: Particles,
        args: Args,
        weighted: bool = False,
    ) -> RecombinedParticles | PartitionedParticles:
        _t = force_bitcast_convert_type(t, jnp.int32)
        key = jr.fold_in(self.key, _t)
        keys = split_by_tree(key, particles)

        def _choice(key, p, count):
            _, weights = unpack_particles(p, weighted)
            if weighted:
                weights = self.weighting_function(weights)
            count = (*count,) if isinstance(count, tuple) else (count,)
            choice = jr.choice(key, p, count, self.with_replacement, weights)
            return choice

        return jtu.tree_map(_choice, keys, particles, self.recombination_count)


class MonteCarloPartitioningKernel(AbstractPartitioningKernel):
    r"""Monte carlo particle resampling/partitioning.

    Rather than using the Monte Carlo method to reduce/recombine the particles, as in
    [`mccube.MonteCarloKernel`][], here the method is used simply to assign particles
    to $m$ equally sized partitions.

    Example:
        ```python
        import jax.numpy as jnp
        import jax.random as jr

        key = jr.PRNGKey(42)
        kernel = mccube.MonteCarloKernel(..., key=key)
        partitioning_kernel = mccube.MonteCarloPartitioningKernel(4, kernel)
        y0 = jnp.ones((12,2))
        result = partitioning_kernel(..., y0, ...)
        # jnp.ones((4,3,2))
        ```

    Attributes:
        partition_count: indicates the requested number of partitions, $m$.
        monte_carlo_kernel: the base monte carlo kernel used for random partition
            assignment, with arbitrary `recombination_count` (as this count is
            overriden based on the `partition_count`).
    """

    monte_carlo_kernel: MonteCarloKernel

    def __call__(
        self,
        t: RealScalarLike,
        particles: Particles,
        args: Args,
        weighted: bool = False,
    ) -> PartitionedParticles:
        recombination_count = jtu.tree_map(
            lambda p, c: (c, p.shape[0] // c), particles, self.partition_count
        )
        kernel = eqx.tree_at(
            lambda x: x.recombination_count,
            self.monte_carlo_kernel,
            recombination_count,
        )
        return jtu.tree_map(lambda p: kernel(t, p, args, weighted), particles)
