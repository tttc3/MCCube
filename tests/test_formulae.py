import chex
import sympy
import numpy as np
from absl.testing import absltest, parameterized

from mccube.formulae import (
    Hadamard,
    StroudSecrest63_31,
    GaussianIntegrationRegion,
    _psd_quadratic_transformation,
)


# Convert Numerical weight functions into sympy weight functions for each region.
_numpy_to_sympy_weights = {
    GaussianIntegrationRegion: lambda syms: sympy.exp(sum([-(x**2) for x in syms]))
}


class FormulaTest(chex.TestCase):
    """
    Ensure that cubature formulae can be instantiated correctly, have appropriate
    attributes set, and correctly integrate polynomials of the appropriate degree.
    """

    # fmt: off
    @parameterized.named_parameters(
        ("1D (Scalar)", 1, np.array([[ 1],
                                     [-1]])),
        ("2D (Even)", 2, np.array([[ 1, 1],
                                   [ 1,-1],
                                   [-1,-1],
                                   [-1, 1]])),
        ("3D (Odd)", 3, np.array([[ 1, 1, 1],
                                  [ 1,-1, 1],
                                  [ 1, 1,-1],
                                  [ 1,-1,-1],
                                  [-1,-1,-1],
                                  [-1, 1,-1],
                                  [-1,-1, 1],
                                  [-1, 1, 1]])),
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
        ("1D (Scalar)", 1, np.array([1/2,-1/2])),
        ("2D (Even)", 2 , np.array([[ 1, 0],
                                    [ 0, 1],
                                    [-1, 0],
                                    [ 0,-1]])),
        ("3D (Odd)", 3, np.array([[ np.sqrt(3/2),            0,            0],
                                  [            0, np.sqrt(3/2),            0],
                                  [            0,            0, np.sqrt(3/2)],
                                  [-np.sqrt(3/2),            0,            0],
                                  [            0,-np.sqrt(3/2),            0],
                                  [            0,            0,-np.sqrt(3/2)]])),
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
        (
            volume_integral,
            max_degree_integral,
            max_degree_integrand,
        ) = self.expected_cubature_integrals(formula)
        self.assertAlmostEqual(volume_integral, formula.region.volume)
        self.assertAlmostEqual(1.0, formula(lambda x: 1.0, normalize=True)[0])
        self.assertAlmostEqual(
            volume_integral, formula(lambda x: 1.0, normalize=False)[0], delta=1e-6
        )
        self.assertAlmostEqual(
            max_degree_integral,
            formula(max_degree_integrand, normalize=False)[0],
            delta=1e-6,
        )

        # Check the cubature for affine transformations of the region.
        # Test for standard normal and some random transformations.
        # transformed_formula = formula.transform()

    def expected_cubature_integrals(self, formula):
        dimension = formula.dimension
        degree = formula.degree
        syms = sympy.symbols(
            [f"x{i}" for i in range(min(dimension, degree))], real=True
        )
        monomial = sympy.prod(syms)
        if degree > dimension:
            monomial = monomial * syms[0] ** (degree - dimension)
        weight = _numpy_to_sympy_weights[formula.region.__class__](syms)
        integrand = monomial * weight
        volume = sympy.integrate(weight, *[(x, -sympy.oo, sympy.oo) for x in syms])
        integral = sympy.integrate(integrand, *[(x, -sympy.oo, sympy.oo) for x in syms])
        lambdified_monomial = sympy.lambdify(syms, monomial, modules="jax")
        return volume, integral, lambda x: lambdified_monomial(*x)


class RegionTest(chex.TestCase):
    """
    Ensure that cubature regions can be instantiated correctly, have appropriate
    attributes set, and correctly define their affine transformations.
    """

    def test_gaussian(self):
        ...

    def test_psd_quadratic_transformation(self):
        """As per example 2 pg 10-12 :cite:p:`stroud1971`, except $T$ is denoted $M$."""
        # fmt: off
        # X as per eq 1.4-10.
        x = np.array([[1, 1, 0],
                      [1,-1, 0],
                      [1, 0, 1],
                      [1, 0,-1]])
        A = np.array([[1,0,0],
                      [0,1,0],
                      [0,0,1]])
        # B is as per eq 1.4-11.
        B = np.array([[ 50,  0,  0],
                      [-20, 20,-16],
                      [-20,-16, 20]]) / 9
        M_inv = np.array([[   1,  0,    0],
                          [-5/3, 4/3,-2/3],
                          [-5/3,-2/3, 4/3]])
        # fmt: on
        # Check linear
        M_linear = _psd_quadratic_transformation(A[1:, 1:], B[1:, 1:], affine=False)
        chex.assert_tree_all_close(B[1:, 1:], M_linear.T @ A[1:, 1:] @ M_linear)
        chex.assert_trees_all_close(M_inv[1:, 1:], M_linear)
        chex.assert_trees_all_close(M_inv[1:, 1:] @ x[:, 1:].T, M_linear @ x[:, 1:].T)

        # Check affine
        M_affine = _psd_quadratic_transformation(A, B)
        chex.assert_trees_all_close(M_inv, M_affine)
        chex.assert_trees_all_close(M_inv @ x.T, M_affine @ x.T)


if __name__ == "__main__":
    absltest.main()
