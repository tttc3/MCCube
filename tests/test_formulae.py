import itertools
from collections.abc import Iterable

import equinox as eqx
import jax
import jax.tree_util as jtu
import numpy as np
import pytest
from equinox.internal import ω
from jax.scipy.special import gamma
from numpy.polynomial.polynomial import polyval

import mccube
from mccube._formulae import (
    _generate_point_permutations,
    builtin_cubature_registry,
)


def _base_formulae_tests(f):
    # Check points, weights, and point_count are consistent.
    assert jtu.tree_structure(f.points) == jtu.tree_structure(f.weights)
    n_points = np.sum(jtu.tree_map(lambda x: x.shape[0], f.points))
    assert f.point_count == n_points
    # Check points and weights stacking.
    assert f.stacked_points.shape[0] == f.stacked_weights.shape[0]
    stacked_points = np.vstack(f.points)
    weight_array = jtu.tree_map(lambda x: np.ones(x.shape[0]), f.points)
    stacked_weights = np.hstack((ω(f.weights) * ω(weight_array)).ω)  # pyright: ignore
    assert eqx.tree_equal(f.stacked_points, stacked_points)
    assert eqx.tree_equal(f.stacked_weights, stacked_weights)


def test_cubature_registry():
    get_registry = mccube.all_subclasses(mccube.AbstractCubature)
    assert eqx.tree_equal(builtin_cubature_registry, get_registry)

    class TestCubature(mccube.AbstractCubature):
        pass

    updated_registry = mccube.all_subclasses(mccube.AbstractCubature)
    builtin_cubature_registry.add(TestCubature)
    assert eqx.tree_equal(updated_registry, builtin_cubature_registry)


@pytest.mark.parametrize(
    "degree, sparse_only, minimal_only, expected_result",
    [
        pytest.param(
            None,
            False,
            False,
            [
                mccube.StroudSecrest63_31,
                mccube.Hadamard,
                mccube.StroudSecrest63_32,
                mccube.StroudSecrest63_52,
            ],
            id="deg=None",
        ),
        pytest.param(
            3,
            False,
            False,
            [mccube.StroudSecrest63_31, mccube.Hadamard, mccube.StroudSecrest63_32],
            id="deg=3",
        ),
        pytest.param(3, False, True, [mccube.StroudSecrest63_31], id="deg=3, minimal"),
        pytest.param(3, True, False, [mccube.StroudSecrest63_31], id="deg=3, sparse"),
        pytest.param(
            5, True, True, [mccube.StroudSecrest63_52], id="deg=5, sparse, minimal"
        ),
    ],
)
def test_search_cubature_registry(degree, sparse_only, minimal_only, expected_result):
    test_registry = {
        mccube.Hadamard,
        mccube.StroudSecrest63_31,
        mccube.StroudSecrest63_32,
        mccube.StroudSecrest63_52,
    }
    region = mccube.GaussianRegion(5)
    result = mccube.search_cubature_registry(
        region, degree, sparse_only, minimal_only, test_registry
    )
    _expected_result = jtu.tree_map(lambda f: f(region), expected_result)
    assert eqx.tree_equal(result, _expected_result)


def test_search_cubature_registry_default():
    mccube.search_cubature_registry(mccube.GaussianRegion(5), 3, False, True)


def test_points_permutations():
    a = np.array([3, 0, 0])
    # Fully symmetric group, GS_d.
    a_FS = _generate_point_permutations(a, "FS")
    a_FS_expected = np.array(
        [
            [-3, 0, 0],
            [0, -3, 0],
            [0, 0, -3],
            [0, 0, 3],
            [0, 3, 0],
            [3, 0, 0],
        ]
    )
    eqx.tree_equal(a_FS, a_FS_expected)
    d = np.shape(a)[0]
    # Symmetry group, S_d.
    a_S = _generate_point_permutations(a, "S")
    a_S_expected = np.atleast_2d(a_FS)[d:, :]
    eqx.tree_equal(a_S, a_S_expected)
    # Reflection group, G_d.
    b = np.array([3, 3, 3])
    b_R = _generate_point_permutations(b, "R")
    b_R_expected = np.array(
        [
            [-3, -3, -3],
            [-3, -3, 3],
            [-3, 3, -3],
            [-3, 3, 3],
            [3, -3, -3],
            [3, -3, 3],
            [3, 3, -3],
            [3, 3, 3],
        ]
    )
    eqx.tree_equal(b_R, b_R_expected)


@pytest.mark.parametrize(
    "formula, degree, test_region_dims",
    [
        pytest.param(mccube.Hadamard, 3, [1, 2, 3, 4], id="Hadamard, d=[1,2,3,4]"),
        pytest.param(
            mccube.StroudSecrest63_31, 3, [1, 2, 3, 4], id="SS63_31, d=[1,2,3,4]"
        ),
        pytest.param(
            mccube.StroudSecrest63_32, 3, [1, 2, 3, 4], id="SS63_32, d=[1,2,3,4]"
        ),
        pytest.param(mccube.StroudSecrest63_52, 5, [2, 3, 4], id="SS63_52, d=[2,3,4]"),
        pytest.param(mccube.StroudSecrest63_53, 5, [3, 4], id="SS63_53, d=[2,3,4]"),
    ],
)
def test_gaussian_cubature(formula, degree, test_region_dims):
    for dim in test_region_dims:
        region = mccube.GaussianRegion(dim)

        f = formula(region)
        _base_formulae_tests(f)

        for monomial in _monomial_generator(f.region.dimension, degree):
            coeffs = np.zeros((degree + 1, dim))
            coeffs[monomial, np.arange(0, dim)] = 1
            integrand = lambda x: np.prod(  # noqa: E731
                polyval(x, coeffs, tensor=False)
            )
            test_integral = _generalized_hermite_monomial_integral(monomial)
            trial_integral = f(integrand)[0]
            assert eqx.tree_equal(test_integral, trial_integral, rtol=1e-5, atol=1e-8)


def _monomial_generator(
    dimension: int, maximum_degree: int
) -> Iterable[tuple[int, ...]]:
    """
    Generate all combinations of multi-variate monomials of degree less than or equal
    to the maximum_degree, for the given dimension.
    """
    monomial_degree_generator = itertools.product(
        range(maximum_degree + 1), repeat=dimension
    )
    exact_monomial_degree_generator = itertools.filterfalse(
        lambda x: sum(x) > maximum_degree, monomial_degree_generator
    )
    return exact_monomial_degree_generator


def _generalized_hermite_monomial_integral(
    monomial: tuple[int, ...], alpha: float = 1 / 2, normalized: bool = True
):
    if any((monomial**ω % 2).ω):
        return jax.numpy.array(0.0)
    normalization_constant = 1.0
    if normalized:
        dim = len(monomial)
        normalization_constant = (alpha ** (-1 / 2) * gamma(1 / 2)) ** dim
    exponent = ((monomial**ω + 1) / 2).ω
    integral = np.prod(jtu.tree_map(lambda m: alpha**-m * gamma(m), exponent))
    return integral / normalization_constant
