import equinox as eqx
import jax.numpy as jnp
import pytest

import mccube

from .helpers import gaussian_formulae


@pytest.mark.parametrize("formula", gaussian_formulae)
@pytest.mark.parametrize("dimension", [3, 5, 10])
def test_local_linear_cubature_path(formula, dimension):
    f = formula(mccube.GaussianRegion(dimension))
    path = mccube.LocalLinearCubaturePath(f)
    p_t0 = path.evaluate(0.0)
    p_t1 = jnp.asarray(path.evaluate(1.0))
    p_dt = jnp.asarray(path.evaluate(0.0, 1.0))
    p_dt2 = jnp.asarray(path.evaluate(0.0, 0.5))
    weights = jnp.asarray(path.weights)

    assert eqx.tree_equal(jnp.abs(p_t0), jnp.zeros(f.stacked_points.shape))
    assert eqx.tree_equal(p_t1, p_dt, jnp.astype(f.stacked_points, p_t1.dtype))
    assert eqx.tree_equal(p_dt2, jnp.astype(jnp.sqrt(0.5) * p_dt, p_dt2.dtype))
    assert eqx.tree_equal(weights, jnp.astype(f.stacked_weights, weights.dtype))
