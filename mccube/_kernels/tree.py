from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import BallTree, KDTree

from .._custom_types import Args, Particles, PartitionedParticles, RealScalarLike
from .._utils import unpack_particles
from .base import AbstractPartitioningKernel

trees = {"KDTree": KDTree, "BallTree": BallTree}


class BinaryTreePartitioningKernel(AbstractPartitioningKernel):
    """Binary tree based particle partitioning.

    !!! warning

        This kernel is based on Sci-Kit learn with JAX JIT compatibility provided
        through the use of [`jax.pure_callback`][].

    Attributes:
        mode: if to use [`sklearn.neighbors.KDTree`][] or [`sklearn.neighbors.BallTree`][]
            binary tree partitioning modes.
        metric: what metric to use in the construciton of the tree.
        metric_kwargs: key word arguments taken by the metric.
    """

    mode: Literal["KDTree", "BallTree"] = "BallTree"
    metric: str | DistanceMetric = "minkowski"
    metric_kwargs: dict = eqx.field(default_factory=dict)

    def __call__(
        self,
        t: RealScalarLike,
        particles: Particles,
        args: Args,
        weighted: bool = False,
    ) -> PartitionedParticles:
        def _tree_fn(p, leaf_size):
            t = trees[self.mode](p, leaf_size, self.metric, **self.metric_kwargs)
            indices = t.get_arrays()[1].reshape(-1, leaf_size)
            return indices.astype(np.int32)

        def _tree_partitioning(p, count):
            leaf_size = p.shape[0] // count
            shape = (count, leaf_size)
            dtype = jnp.int32
            result_shape_dtype = jax.ShapeDtypeStruct(shape, dtype)
            p_unpacked, _ = unpack_particles(p, weighted)
            indices = jax.pure_callback(  # type: ignore
                _tree_fn, result_shape_dtype, p_unpacked, leaf_size
            )
            return p[indices]

        return jtu.tree_map(_tree_partitioning, particles, self.partition_count)
