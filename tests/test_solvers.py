import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from diffrax import (
    diffeqsolve,
    Euler,
    EulerHeun,
    MultiTerm,
    ODETerm,
    SaveAt,
    VirtualBrownianTree,
    WeaklyDiagonalControlTerm,
)
from jax.scipy.stats import multivariate_normal

import mccube

from .helpers import gaussian_formulae

key = jr.key(42)
init_key, rng_key = jr.split(key)
t0 = 0.0
dt0 = 0.05
epochs = 512
t1 = t0 + dt0 * epochs
k, d = 16, 3
y0 = jr.multivariate_normal(init_key, mean=jnp.ones(d), cov=jnp.eye(d), shape=(k,))

target_mean = jnp.array([1.0, 2.0, 3.0])
target_cov = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])


@eqx.filter_vmap
@eqx.filter_grad
def grad_logdensity(p):
    return multivariate_normal.logpdf(p, mean=target_mean, cov=target_cov)


def test_diffrax_ula():
    """Undajusted Langevin Algorithm in diffrax.

    This will be used as a reference numerical solution, against which the MCC methods
    will be compared.
    """
    ode = ODETerm(lambda t, p, args: grad_logdensity(p))
    cde = WeaklyDiagonalControlTerm(
        lambda t, p, args: jnp.sqrt(2.0),
        VirtualBrownianTree(t0, t1, dt0 / 10, (k, d), key=jr.key(42)),
    )
    terms = MultiTerm(ode, cde)
    diffeqsolve(terms, Euler(), t0, t1, dt0, y0)


def test_MCCSolver_init():
    key = jr.key(42)
    with pytest.raises(
        ValueError, match="n_substeps must be at least one;"
    ), pytest.warns(UserWarning, match="diffrax.Euler solver"):
        mccube.MCCSolver(EulerHeun(), mccube.MonteCarloKernel(10, key=key), 0)


@pytest.mark.parametrize("formula", gaussian_formulae)
def test_MCCSolver_ula(formula):
    cubature = formula(mccube.GaussianRegion(d))
    ode = ODETerm(lambda t, p, args: grad_logdensity(p))
    cde = WeaklyDiagonalControlTerm(
        lambda t, p, args: jnp.sqrt(2.0),
        mccube.LocalLinearCubaturePath(cubature),
    )
    terms = mccube.MCCTerm(ode, cde)

    # Test values of substep.
    with pytest.raises(ValueError, match="n_substeps"):
        solver = mccube.MCCSolver(Euler(), mccube.MonteCarloKernel(k, key=rng_key), 0)

    key1, key2, key_weighted = jr.split(rng_key, 3)

    solver = mccube.MCCSolver(
        Euler(),
        mccube.MonteCarloKernel(k, key=key1),
    )
    sol = diffeqsolve(
        terms, solver, t0, t1, dt0, y0, saveat=SaveAt(dense=True, t1=True)
    )
    assert sol.ys.shape == (1, k, d)  # type: ignore
    assert sol.evaluate(t1).shape == (k, d)

    n_substeps = 2
    solver2 = mccube.MCCSolver(
        Euler(),
        mccube.MonteCarloKernel(k, key=key2),
        n_substeps=n_substeps,
    )
    ts = jnp.arange(t0, t1, dt0)
    sol2 = diffeqsolve(
        terms, solver2, t0, t1, dt0, y0, saveat=SaveAt(ts=ts, dense=True)
    )
    assert sol2.ys.shape == (ts.shape[0], k, d)  # type: ignore
    assert sol2.evaluate(t0 + dt0).shape == (k, d)

    # Test Weighted Particles
    solver_weighted = mccube.MCCSolver(
        Euler(),
        mccube.MonteCarloKernel(k, key=key_weighted),
        n_substeps=2,
        weighted=True,
    )
    y0_weighted = mccube.pack_particles(y0, jnp.ones(k))
    sol_weighted = diffeqsolve(
        terms,
        solver_weighted,
        t0,
        t1,
        dt0,
        y0_weighted,
        saveat=SaveAt(t1=True),
    )
    assert sol_weighted.ys.shape == (1, k, d + 1)  # type: ignore
    particles, weights = mccube.unpack_particles(sol_weighted.ys[0], weighted=True)  # type: ignore
    assert eqx.tree_equal(weights.sum(), jnp.array(1.0), rtol=1e-5, atol=1e-8)  # type: ignore
