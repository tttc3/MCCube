import itertools

import chex
import sympy
import numpy as np
from absl.testing import absltest, parameterized

from mccube.formulae import Hadamard, StroudSecrest63_31
from mccube.regions import GaussianIntegrationRegion


# Convert Numerical weight functions into sympy weight functions for each region.
_numpy_to_sympy_weights = {
    GaussianIntegrationRegion: lambda syms: sympy.exp(sum([-(x**2) for x in syms]))
}

DELTA_RESOLUTION = np.finfo(np.float32).resolution


class FormulaTest(chex.TestCase):
    """
    Ensure that cubature formulae can be instantiated correctly, have appropriate
    attributes set, and correctly integrate polynomials of the appropriate degree.
    """

    # fmt: off
    @parameterized.named_parameters(
        ("1D (Scalar)", 1, np.array([[ 1],
                                     [-1]]) / np.sqrt(1)),
        ("2D (Even)"  , 2, np.array([[ 1, 1],
                                     [ 1,-1],
                                     [-1,-1],
                                     [-1, 1]]) / np.sqrt(2)),
        ("3D (Odd)"   , 3, np.array([[ 1, 1, 1],
                                     [ 1,-1, 1],
                                     [ 1, 1,-1],
                                     [ 1,-1,-1],
                                     [-1,-1,-1],
                                     [-1, 1,-1],
                                     [-1,-1, 1],
                                     [-1, 1, 1]]) / np.sqrt(3)),
    ) 
    # fmt: on
    def test_hadamard(self, dimension, expected_vectors):
        self.check_cubature(
            Hadamard(dimension),
            expected_region=GaussianIntegrationRegion,
            expected_degree=4,
            expected_vectors=expected_vectors,
        )

    # fmt: off
    @parameterized.named_parameters(
        ("1D (Scalar)", 1, np.array([[ 1],
                                     [-1]]) / np.sqrt(2)),
        ("2D (Even)"  , 2, np.array([[ 1, 0],
                                     [ 0, 1],
                                     [-1, 0],
                                     [ 0,-1]])),
        ("3D (Odd)"   , 3, np.array([[ 1, 0, 0],
                                     [ 0, 1, 0],
                                     [ 0, 0, 1],
                                     [-1, 0, 0],
                                     [ 0,-1, 0],
                                     [ 0, 0,-1]]) * np.sqrt(3/2)),
    ) 
    # fmt:on
    def test_stroudsecrest63_31(self, dimension, expected_vectors):
        self.check_cubature(
            StroudSecrest63_31(dimension),
            expected_region=GaussianIntegrationRegion,
            expected_degree=3,
            expected_vectors=expected_vectors,
        )

    def check_cubature(
        self, formula, expected_region, expected_degree, expected_vectors
    ):
        """Primary cubature formulae tests.

        Implicitly Tests:
            - Integration Region Volume
            - Formula coefficients
            - Formula vector count

        Explicitly Tests:
            - Formula degree
            - Formula vectors
            - Formula region normalization
        """
        self.assertIsInstance(formula.region, expected_region)
        self.assertEqual(formula.degree, expected_degree)
        self.assertEqual(formula.vector_count, formula.vectors.shape[0])
        chex.assert_trees_all_close(formula.vectors, expected_vectors)

        # Check that the cubature formula is indeed of the expected degree.
        monomial_generator = itertools.combinations_with_replacement(
            range(formula.degree + 1), formula.dimension
        )
        exact_monomial_generator = itertools.filterfalse(
            lambda x: sum(x) > formula.degree, monomial_generator
        )

        for monomial_degrees in exact_monomial_generator:
            (
                volume_integral,
                integral,
                unweighted_integrand,
                symbolic_integrand,
            ) = self.symbolic_monomial_integrals(formula, monomial_degrees)
            print(f"Testing cubature for monomial: {symbolic_integrand}")
            self.assertAlmostEqual(volume_integral, formula.region.volume)
            self.assertAlmostEqual(
                integral,
                formula(unweighted_integrand, normalize=False)[0],
                delta=DELTA_RESOLUTION,
            )
            self.assertAlmostEqual(
                integral / volume_integral,
                formula(unweighted_integrand, normalize=True)[0],
                delta=DELTA_RESOLUTION,
            )

        # Check the cubature for affine transformations of the region.
        # Test for standard normal and some random transformations.
        # transformed_formula = formula.transform()

    def symbolic_monomial_integrals(self, formula, monomial_degrees):
        """
        Generates integral and integrand all d-dimension polynomials of degree not
        exceeding $m$.
        """
        # Constuct symbolic integrand.
        syms = sympy.symbols([f"x_{i}" for i, _ in enumerate(monomial_degrees)])
        monomial = sympy.prod([s**d for s, d in zip(syms, monomial_degrees)])
        weight = _numpy_to_sympy_weights[formula.region.__class__](syms)
        integrand = monomial * weight
        # Perform symbolic integration.
        volume = sympy.integrate(weight, *[(x, -sympy.oo, sympy.oo) for x in syms])
        integral = sympy.integrate(integrand, *[(x, -sympy.oo, sympy.oo) for x in syms])
        lambdified_monomial = sympy.lambdify(syms, monomial, modules="jax")
        return volume, integral, lambda x: lambdified_monomial(*x), monomial


if __name__ == "__main__":
    absltest.main()
