from collections.abc import Callable

import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike, Shaped

from .._custom_types import Args, Particles, PartitionedParticles, RealScalarLike
from .._metrics import center_of_mass
from .._utils import unpack_particles
from .base import AbstractPartitioningKernel


class StratifiedPartitioningKernel(AbstractPartitioningKernel):
    r"""Norm based particle stratification/partitioning.

    Computes the norm of each particle, sorts by the norm, and then partitions the
    sorted particles into $m$ equal-size strata/partitions.

    Attributes:
        partition_count: indicates the requested number of partitions, $m$.
        norm: the norm used for sorting the particles. If [`None`][], the particles
            remain unsorted and the kernel is equivalent to `particles.reshape`.
    """

    norm: Callable[
        [Shaped[ArrayLike, "n d"]], Shaped[ArrayLike, " n"]
    ] | None = jtu.Partial(jnp.linalg.norm, axis=-1)

    def __call__(
        self,
        t: RealScalarLike,
        particles: Particles,
        args: Args,
        weighted: bool = False,
    ) -> PartitionedParticles:
        def _stratified_partitioning(_p, _count):
            if self.norm is None:
                return _p.reshape(_count, -1, _p.shape[-1])
            _p_unpacked, _w = unpack_particles(_p, weighted)
            com = center_of_mass(_p_unpacked, _w)
            com_centred_p = _p_unpacked - com
            norm_vector = self.norm(com_centred_p)
            norm_sorted_indices = jnp.argsort(norm_vector)
            partition_indices = norm_sorted_indices.reshape(_count, -1)
            return _p[partition_indices]

        return jtu.tree_map(_stratified_partitioning, particles, self.partition_count)
