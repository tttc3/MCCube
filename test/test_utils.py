import mccube
import equinox as eqx
import jax.numpy as jnp


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
