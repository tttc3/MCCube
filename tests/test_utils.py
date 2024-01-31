import equinox as eqx
import jax.numpy as jnp

import mccube


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


def test_pack_pacticles():
    particles = jnp.ones((4, 4))
    weights = jnp.full((4, 1), 1 / 4)
    expected_values = jnp.c_[particles, weights]
    assert eqx.tree_equal(expected_values, mccube.pack_particles(particles, weights))
    assert eqx.tree_equal(expected_values, mccube.pack_particles(particles, weights[:]))
    assert eqx.tree_equal(particles, mccube.pack_particles(particles, None))

    unnormalized_weights = weights * 4
    assert eqx.tree_equal(
        expected_values, mccube.pack_particles(particles, unnormalized_weights)
    )


def test_unpack_particles():
    x = jnp.c_[jnp.ones((3, 4)), jnp.zeros(3)]
    weights = x[:, -1]
    assert eqx.tree_equal((x, None), mccube.unpack_particles(x, False))
    assert eqx.tree_equal((x[:, :-1], weights), mccube.unpack_particles(x, True))
