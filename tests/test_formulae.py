import itertools
from collections.abc import Iterable

import chex
import jax
import jax.tree_util as jtu
import numpy as np
from absl.testing import absltest, parameterized
from equinox.internal import ω
from jax.scipy.special import gamma
from numpy.polynomial.polynomial import polyval

from mccube.formulae import (
    AbstractCubature,
    Hadamard,
    LyonsVictoir04_512,
    StroudSecrest63_31,
    StroudSecrest63_32,
    StroudSecrest63_52,
    StroudSecrest63_53,
    _CubatureRegistryMeta,
    _generate_point_permutations,
    builtin_cubature_registry,
    search_cubature_registry,
)
from mccube.regions import GaussianRegion, WienerSpace


def _base_formulae_tests(f):
    # Check points, weights, and point_count are consistent.
    chex.assert_trees_all_equal_structs(f.points, f.weights)
    n_points = np.sum(jtu.tree_map(lambda x: x.shape[0], f.points))
    chex.assert_equal(f.point_count, n_points)
    # Check points and weights stacking.
    chex.assert_equal(f.stacked_points.shape[0], f.stacked_weights.shape[0])
    stacked_points = np.vstack(f.points)
    weight_array = jtu.tree_map(lambda x: np.ones(x.shape[0]), f.points)
    stacked_weights = np.hstack((ω(f.weights) * ω(weight_array)).ω)
    chex.assert_trees_all_equal(f.stacked_points, stacked_points)
    chex.assert_trees_all_equal(f.stacked_weights, stacked_weights)


class CubatureFormulaeTests(chex.TestCase):
    def test_cubature_registry(self):
        """
        Check that the :data:`builtin_cubature_registry` alias is initialized as
        expected, and that new cubature classes are added to the registry.
        """
        # Check default registry is correct
        get_registry = _CubatureRegistryMeta.get_registered_formulae()
        chex.assert_trees_all_equal(builtin_cubature_registry, get_registry)

        class TestCubature(AbstractCubature):
            pass

        updated_registry = _CubatureRegistryMeta.get_registered_formulae()
        builtin_cubature_registry.append(TestCubature)
        chex.assert_trees_all_equal(updated_registry, builtin_cubature_registry)

    # fmt: off
    @parameterized.named_parameters(
        ("deg=None"              , None, False, False, [StroudSecrest63_31, Hadamard, StroudSecrest63_32, StroudSecrest63_52]),
        ("deg=3"                 ,    3, False, False,                     [StroudSecrest63_31, Hadamard, StroudSecrest63_32]),
        ("deg=3, minimal"        ,    3, False,  True,                                                   [StroudSecrest63_31]),
        ("deg=3, sparse"         ,    3,  True, False,                                                   [StroudSecrest63_31]),
        ("deg=5, spares, minimal",    5,  True,  True,                                                   [StroudSecrest63_52]),
    )
    # fmt: on
    def test_search_cubature_registry(
        self, degree, sparse_only, minimal_only, expected_result
    ):
        """
        Check that :func:`search_cubature_registry` returns the expected results for
        a set of given filters. Search filters are validated for the ten dimensional
        :class:`GaussianRegion` and :class:`StandardGaussianRegion`.
        """
        test_registry = {
            Hadamard,
            StroudSecrest63_31,
            StroudSecrest63_32,
            StroudSecrest63_52,
        }
        region = GaussianRegion(5)
        result = search_cubature_registry(
            region, degree, sparse_only, minimal_only, test_registry
        )
        _expected_result = jtu.tree_map(lambda f: f(region), expected_result)
        chex.assert_trees_all_equal(result, _expected_result)

    def test_points_permutations(self):
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
        chex.assert_trees_all_equal(a_FS, a_FS_expected)
        d = a.shape[0]
        # Symmetry group, S_d.
        a_S = _generate_point_permutations(a, "S")
        a_S_expected = a_FS[d:, :]
        chex.assert_trees_all_equal(a_S, a_S_expected)
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
        chex.assert_trees_all_equal(b_R, b_R_expected)


class GaussianFormulaeTests(chex.TestCase):
    def setUp(self):
        # Required to obtain reasonable precision comparisons between the closed form
        # of the integrals and the cubature.
        jax.config.update("jax_enable_x64", True)
        return super().setUp()

    @parameterized.named_parameters(
        ("Hadamard, d=[1,2,3,4]", Hadamard, 3, [1, 2, 3, 4]),
        ("SS63_31, d=[1,2,3,4]", StroudSecrest63_31, 3, [1, 2, 3, 4]),
        ("SS63_32, d=[1,2,3,4]", StroudSecrest63_32, 3, [1, 2, 3, 4]),
        ("SS63_52, d=[2,3,4]", StroudSecrest63_52, 5, [2, 3, 4]),
        ("SS63_53, d=[2,3,4]", StroudSecrest63_53, 5, [3, 4]),
    )
    def test_gaussian_cubature(self, formula, degree, test_region_dims):
        """
        Checks:
            - Points and weights have the same PyTree structure.
            - Number of points and point_count match/are consistent.
            - Cubature exactly integrates all monomials of degree \le m.

        Assumptions:
            - _monomial_generator is correct.
            - _generate_hermite_monomial_integral is correct.
        """
        for dim in test_region_dims:
            with self.subTest(dim=dim):
                region = GaussianRegion(dim)

                f = formula(region)
                _base_formulae_tests(f)

                for monomial in _monomial_generator(f.region.dimension, degree):
                    with self.subTest(monomial=monomial):
                        coeffs = np.zeros((degree + 1, dim))
                        coeffs[monomial, np.arange(0, dim)] = 1
                        integrand = lambda x: np.prod(  # noqa: E731
                            polyval(x, coeffs, tensor=False)
                        )
                        test_integral = _generalized_hermite_monomial_integral(monomial)
                        trial_integral = f(integrand)[0]
                        chex.assert_trees_all_close(
                            test_integral, trial_integral, atol=1e-8
                        )


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


class WienerFormulaeTests(chex.TestCase):
    @parameterized.named_parameters(
        ("Hadamard, d=[1,2,3,4]", Hadamard, 3, [1, 2, 3, 4]),
        ("SS63_31, d=[1,2,3,4]", StroudSecrest63_31, 3, [1, 2, 3, 4]),
        ("SS63_32, d=[1,2,3,4]", StroudSecrest63_32, 3, [1, 2, 3, 4]),
        ("SS63_52, d=[2,3,4]", StroudSecrest63_52, 5, [2, 3, 4]),
        ("SS63_53, d=[2,3,4]", StroudSecrest63_53, 5, [3, 4]),
    )
    def test_LyonsVictoir04_cubature(self, formula, degree, test_region_dims):
        for dim in test_region_dims:
            with self.subTest(dim=dim):
                region = WienerSpace(dim)
                f = LyonsVictoir04_512(region, formula)
                _base_formulae_tests(f)

                # Test time-interval definition.
                chex.assert_trees_all_equal(f.t0, f.region.t0, 0.0)
                chex.assert_trees_all_equal(f.t1, f.region.t1, 1.0)

                # Check gaussian cubature property pass-through.
                chex.assert_equal(f.degree, f.gaussian_cubature.degree)
                chex.assert_equal(f.sparse, f.gaussian_cubature.sparse)
                chex.assert_trees_all_equal(
                    f.stacked_points, f.gaussian_cubature.stacked_points
                )
                chex.assert_trees_all_equal(
                    f.stacked_weights, f.gaussian_cubature.stacked_weights
                )
                expected_eval = np.concatenate(
                    [f.stacked_points[:, None, :], f.stacked_weights[:, None, None]], -1
                )
                t0 = 0.5
                t1 = 3.0
                expected_eval2 = np.copy(expected_eval)
                expected_eval2[..., :-1] *= 0.0
                expected_eval3 = np.copy(expected_eval)
                expected_eval3[..., :-1] *= np.sqrt(t1 - t0)
                expected_eval4 = np.copy(expected_eval)
                expected_eval4[..., :-1] *= np.sqrt(f.t1 - t0)
                chex.assert_trees_all_close(f.evaluate(1.0), expected_eval)
                chex.assert_trees_all_close(f.evaluate(0.0), expected_eval2)
                chex.assert_trees_all_close(f.evaluate(t0, t1), expected_eval3)
                chex.assert_trees_all_close(f.evaluate(t0), expected_eval4)


if __name__ == "__main__":
    absltest.main()
