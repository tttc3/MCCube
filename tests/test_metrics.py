import equinox as eqx
import jax.numpy as jnp
import pytest

import mccube


def test_gaussian_wasserstein_metric():
    d = 3
    means = (jnp.ones(d), jnp.ones(d))
    covs = (jnp.eye(d), jnp.eye(d))
    dist = mccube.gaussian_wasserstein_metric(means, covs)
    assert eqx.tree_equal(dist, jnp.asarray(0.0, jnp.complex128))

    means = (jnp.ones(d), 2 * jnp.ones(d))
    covs = (jnp.eye(d), 2 * jnp.eye(d))
    dist = mccube.gaussian_wasserstein_metric(means, covs)

    with pytest.raises(AssertionError):
        assert eqx.tree_equal(dist, jnp.asarray(0.0, jnp.complex128))


def test_center_of_mass():
    y0 = jnp.array([[1.0, 2.0], [3.0, -4.0], [5.0, 6.0]])
    weights = None

    com = mccube.center_of_mass(y0, weights)
    expected_com = jnp.array([3, 4 / 3])
    assert eqx.tree_equal(com, expected_com)

    y0, weights = mccube.unpack_particles(y0, True)
    com_weighted = mccube.center_of_mass(y0, weights)
    expected_com = jnp.array([5.0])
    assert eqx.tree_equal(com_weighted, expected_com)


def test_pairwise_metric():
    y0 = jnp.array([[0.0, 0.0], [3.0, 4.0], [5.0, 12.0]])
    dist = mccube.pairwise_metric(y0, y0)
    expected_dist = jnp.array(
        [[0.0, 5.0, 13.0], [5.0, 0.0, jnp.sqrt(68)], [13.0, jnp.sqrt(68), 0.0]]
    )
    assert eqx.tree_equal(dist, expected_dist)
