import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

import mccube
from mccube._custom_types import (
    Args,
    Particles,
    PartitionedParticles,
    RealScalarLike,
    RecombinedParticles,
)
from mccube._kernels.random import MonteCarloKernel


# _kernels/base.py
def test_partitioning_recombination_kernel():
    class PartitioningKernel(mccube.AbstractPartitioningKernel):
        def __call__(
            self,
            t: RealScalarLike,
            particles: Particles,
            args: Args,
            weighted: bool = False,
        ) -> PartitionedParticles:
            return jtu.tree_map(
                lambda p, c: p.reshape(-1, c, p.shape[-1]),
                particles,
                self.partition_count,
            )

    class RecombinationKernel(mccube.AbstractRecombinationKernel):
        def __call__(
            self,
            t: RealScalarLike,
            particles: Particles,
            args: Args,
            weighted: bool = False,
        ) -> RecombinedParticles:
            return jtu.tree_map(lambda p, c: p[:c], particles, self.recombination_count)

    y0 = jnp.array([[2.0, 4.0, 6.0, 8.0]]).T
    y1 = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]).T
    y_tree = [y0, y1]
    partitioning_kernel = PartitioningKernel([2, 3])
    recombination_kernel = RecombinationKernel([2, 3])
    kernel = mccube.PartitioningRecombinationKernel(
        partitioning_kernel, recombination_kernel
    )
    values = kernel(0.0, y_tree, None)
    assert eqx.tree_equal(values, y_tree)

    # Weighted
    y_tree = jtu.tree_map(
        lambda x: mccube.pack_particles(x, jnp.ones(x.shape[0])), y_tree
    )
    values = kernel(0.0, y_tree, None, weighted=True)
    assert eqx.tree_equal(values, y_tree)


# _kernels/_random.py
@pytest.mark.skip("Not yet implemented; requires statistical testing.")
def test_monte_carlo_kernel():
    ...


def test_monte_carlo_partitioning_kernel():
    n_parts = 4

    y0 = jnp.array([[1.0, 0.01], [2.0, 1.0], [3.0, 100.0], [4.0, 10000.0]])

    key = jr.key(42)
    mc_kernel = MonteCarloKernel(None, key=key)
    kernel = mccube.MonteCarloPartitioningKernel(n_parts, mc_kernel)
    values = kernel(0.0, y0, ...)
    assert values.shape == (n_parts, y0.shape[0] // n_parts, y0.shape[-1])
    assert eqx.tree_equal(
        jnp.unique(values, return_counts=True), jnp.unique(y0, return_counts=True)
    )

    key = jr.key(42)
    mc_kernel = MonteCarloKernel(None, weighting_function=lambda x: x, key=key)
    kernel = mccube.MonteCarloPartitioningKernel(n_parts, mc_kernel)
    values = kernel(0.0, y0, ..., weighted=True)
    assert values.shape == (n_parts, y0.shape[0] // n_parts, y0.shape[-1])
    assert eqx.tree_equal(
        jnp.unique(values, return_counts=True), jnp.unique(y0, return_counts=True)
    )
    sorted_idx = jnp.argsort(y0[:, -1], axis=0)[::-1]
    assert eqx.tree_equal(values, y0[sorted_idx[:, None]])


# _kernels/_stratified.py
def test_stratified_partitioning_kernel():
    n_parts = 4

    y0 = jnp.array([[-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]]).T
    kernel = mccube.StratifiedPartitioningKernel(n_parts)
    values = kernel(0.0, y0, ...)
    assert values.shape == (n_parts, y0.shape[0] // n_parts, y0.shape[-1])
    # fmt: off
    expected_partitioning = jnp.array(
        [
            [[-1.0], [1.0]], 
            [[-2.0], [2.0]], 
            [[-3.0], [3.0]], 
            [[-4.0], [4.0]]
        ]
    )
    # fmt: on
    assert eqx.tree_equal(expected_partitioning, values)

    y0_matrix = jnp.tile(y0, 3)
    expected_partitioning = jnp.tile(expected_partitioning, 3)
    expected_shape = (n_parts, y0_matrix.shape[0] // n_parts, y0_matrix.shape[-1])
    values = kernel(0.0, y0_matrix, ...)
    assert values.shape == expected_shape
    assert eqx.tree_equal(expected_partitioning, values)

    kernel = mccube.StratifiedPartitioningKernel(n_parts)
    y0_matrix = y0_matrix.at[:, -1].apply(jnp.abs)
    expected_partitioning = expected_partitioning.at[:, :, -1].apply(jnp.abs)
    values = kernel(0.0, y0_matrix, ..., weighted=True)
    assert values.shape == expected_shape
    assert eqx.tree_equal(expected_partitioning, values)

    kernel = mccube.StratifiedPartitioningKernel(n_parts, norm=None)
    expected_partitioning = y0_matrix.reshape(n_parts, -1, y0_matrix.shape[-1])
    values = kernel(0.0, y0_matrix, ...)
    assert values.shape == expected_shape
    assert eqx.tree_equal(expected_partitioning, values)


# _kernels/_tree.py
@pytest.mark.parametrize("mode", ["KDTree", "BallTree"])
def test_binary_tree_partitioning_kernel(mode):
    n_parts = 3, 4
    y0 = jnp.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    )
    y1 = y0[:-2]

    kernel = mccube.BinaryTreePartitioningKernel(
        {"y0": n_parts[0], "y1": n_parts[1]}, mode
    )
    values = kernel(0.0, {"y0": y0, "y1": y1}, None)
    expected_unique, values_unique = jtu.tree_map(
        lambda x: jnp.unique(x, return_counts=True), ({"y0": y0, "y1": y1}, values)
    )
    assert eqx.tree_equal(expected_unique, values_unique)
    assert values["y0"].shape == (3, 2, 2)
    assert values["y1"].shape == (4, 1, 2)


# # _kernels/_carathedory.py
# def test_carathedory():
#     from mccube._kernels import tlc

#     n, d = 64, 2

#     key = jr.key(42)
#     y0 = jr.multivariate_normal(key, jnp.zeros(d), jnp.eye(d), (n,))
#     weights = jnp.arange(1.0, n + 1.0)

#     new_weights = tlc(y0, weights)

#     expected_com = mccube.center_of_mass(jnp.c_[y0, weights], True)
#     recombined_com = mccube.center_of_mass(jnp.c_[y0, new_weights], True)

#     assert jnp.count_nonzero(weights) > jnp.count_nonzero(new_weights)
#     assert eqx.tree_equal(expected_com, recombined_com, rtol=1e-5, atol=1e-8)
