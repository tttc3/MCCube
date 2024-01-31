import functools as ft

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.scipy.linalg import sqrtm

import mccube


@pytest.fixture
def reference_problem(n=128, d=5):
    k1, k2, k3 = jr.split(jr.PRNGKey(42), 3)
    mean1, mean2 = jr.uniform(k1, (2, d), minval=-1.0)
    g1, g2 = jnp.sqrt(0.2) * jr.normal(k2, (2, d, d))
    cov1 = g1 @ g1.T
    cov2 = g2 @ g2.T

    return (mean1, mean2), (cov1, cov2)


def test_pairwise_metric():
    y0 = jnp.array([[0.0, 0.0], [3.0, 4.0], [5.0, 12.0]])
    dist = mccube.pairwise_metric(y0, y0)
    expected_dist = jnp.array(
        [[0.0, 5.0, 13.0], [5.0, 0.0, jnp.sqrt(68)], [13.0, jnp.sqrt(68), 0.0]]
    )
    print(expected_dist)
    assert eqx.tree_equal(dist, expected_dist)


# TODO: currently closed-form metrics are only tested for equivalence in their limits.
def test_gaussian_squared_bures_distance(reference_problem):
    means, covs = reference_problem
    A, B = covs
    A_sqrt = sqrtm(A)

    ref_sigma_0 = jnp.abs(jnp.trace(A + B - 2 * sqrtm(A_sqrt @ B @ A_sqrt)))
    result_sigma_0 = mccube.gaussian_squared_bures_distance(A, B, 0.0)
    assert eqx.tree_equal(ref_sigma_0, result_sigma_0, rtol=1e-5, atol=1e-8)

    ref_sigma_inf = jnp.abs(jnp.trace(A + B))
    result_sigma_inf = mccube.gaussian_squared_bures_distance(A, B, jnp.inf)
    assert eqx.tree_equal(ref_sigma_inf, result_sigma_inf)


def test_gaussian_optimal_transport(reference_problem):
    means, covs = reference_problem
    cost = ft.partial(mccube.lpp_metric, p=4)
    mean_cost = cost(means[0], means[1])
    ref_sigma_0 = mean_cost + mccube.gaussian_squared_bures_distance(
        covs[0], covs[1], 0.0
    )
    result_sigma_0 = mccube.gaussian_optimal_transport(means, covs, 0.0, cost)
    assert eqx.tree_equal(ref_sigma_0, result_sigma_0, rtol=1e-5, atol=1e-8)

    ref_sigma_inf = mean_cost + mccube.gaussian_squared_bures_distance(
        covs[0], covs[1], jnp.inf
    )
    result_sigma_inf = mccube.gaussian_optimal_transport(means, covs, jnp.inf, cost)
    assert eqx.tree_equal(ref_sigma_inf, result_sigma_inf, rtol=1e-5, atol=1e-8)


def test_gaussian_wasserstein_metric(reference_problem):
    means, covs = reference_problem
    p = 1.5
    cost = ft.partial(mccube.lpp_metric, p=p)
    ref = mccube.gaussian_optimal_transport(means, covs, 0.0, cost) ** (1 / p)
    result = mccube.gaussian_wasserstein_metric(means, covs, p)
    assert eqx.tree_equal(ref, result)


def test_gaussian_maximum_mean_discrepancy(reference_problem):
    means, covs = reference_problem
    cost = mccube.lp_metric
    ref = cost(means[0], means[1])
    result = mccube.gaussian_maximum_mean_discrepancy(means[0], means[1], cost)
    assert eqx.tree_equal(ref, result, rtol=1e-5, atol=1e-8)


def test_gaussian_sinkhorn_divergence(reference_problem):
    means, covs = reference_problem
    p = 2
    ref_sigma_0 = mccube.gaussian_wasserstein_metric(means, covs, p) ** p
    result_sigma_0 = mccube.gaussian_sinkhorn_divergence(means, covs, 0.0)
    assert eqx.tree_equal(ref_sigma_0, result_sigma_0, rtol=1e-5, atol=1e-8)

    cost = ft.partial(mccube.lp_metric, p=3)
    ref_sigma_inf = mccube.gaussian_maximum_mean_discrepancy(means[0], means[1], cost)
    result_sigma_inf = mccube.gaussian_sinkhorn_divergence(means, covs, jnp.inf, cost)
    assert eqx.tree_equal(ref_sigma_inf, result_sigma_inf, rtol=1e-5, atol=1e-8)
