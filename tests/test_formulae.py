import itertools

import chex
import sympy
from absl.testing import absltest, parameterized
import numpy as np

from mccube.formulae import Hadamard, StroudSecrest63_31, StroudSecrest63_32
from mccube.regions import GaussianIntegrationRegion

# Test Constants
DELTA_RESOLUTION = 1e-4  # based on JAX internal tolerances.

# Convert Numerical weight functions into sympy weight functions for each region.
_numpy_to_sympy_weights = {
    GaussianIntegrationRegion: lambda x_mat: sympy.exp(-x_mat.T * x_mat)
}


# Generate kwargs for creating affine transformed instances of integration regions.
def _gaussian_random_kwargs(rng, dimension):
    mean = rng.uniform(size=dimension)
    random_matrix = rng.uniform(size=(dimension, dimension))
    covariance = random_matrix @ random_matrix.T  # Ensures PSD.
    return dict(mean=mean, covariance=covariance)


_affine_region_kwargs = {GaussianIntegrationRegion: _gaussian_random_kwargs}


class FormulaTest(chex.TestCase):
    """
    Ensure that cubature formulae can be instantiated correctly, have appropriate
    attributes set, and correctly integrate polynomials of the appropriate degree.
    """

    def setUp(self):
        self.n_affine = 5
        self.rng = np.random.default_rng(1234)
        return super().setUp()

    @parameterized.named_parameters(
        ("01D", 1),
        ("02D", 2),
        ("03D", 3),
        ("10D", 10),
        ("25D", 25),
    )
    def test_hadamard(self, dimension):
        self.check_cubature(
            Hadamard(dimension),
            expected_region=GaussianIntegrationRegion,
            expected_degree=3,
        )

    @parameterized.named_parameters(
        ("01D", 1),
        ("02D", 2),
        ("03D", 3),
        ("10D", 10),
        ("25D", 25),
    )
    def test_stroudsecrest63_31(self, dimension):
        self.check_cubature(
            StroudSecrest63_31(dimension),
            expected_region=GaussianIntegrationRegion,
            expected_degree=3,
        )

    @parameterized.named_parameters(
        ("01D", 1),
        ("02D", 2),
        ("03D", 3),
    )
    def test_stroudsecrest63_32(self, dimension):
        self.check_cubature(
            StroudSecrest63_32(dimension),
            expected_region=GaussianIntegrationRegion,
            expected_degree=3,
        )

    def check_cubature(self, formula, expected_region, expected_degree):
        """Primary cubature formulae tests.

        Implicitly Tests:
            - Integration Region Volume.
            - Formula coefficients.
            - Formula vector count.
            - Formula vectors.

        Explicitly Tests:
            - Formula degree.
            - Formula region normalization.
            - Formula region transformation.
            - Formula for affine transformations of the region.
        """
        self.assertIsInstance(formula.region, expected_region)
        self.assertEqual(formula.degree, expected_degree)
        self.assertEqual(formula.vector_count, formula.vectors.shape[0])

        formulae = [formula]
        affine_formulae = [
            formula.transform(
                **_affine_region_kwargs[formula.region.__class__](
                    self.rng, formula.dimension
                ),
            )
        ]
        # Sympy integration is too expensive to perform at dimensions above two.
        if formula.dimension < 3:
            formulae += affine_formulae

        for formula in formulae:
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
                # Ideally would test closeness via ULP rather than this scaling.
                # TODO: investigate ULP closeness tests via chex.
                self.assertAlmostEqual(
                    integral / volume_integral,
                    formula(unweighted_integrand, normalize=False)[0] / volume_integral,
                    delta=DELTA_RESOLUTION,
                )
                self.assertAlmostEqual(
                    integral / volume_integral,
                    formula(unweighted_integrand, normalize=True)[0],
                    delta=DELTA_RESOLUTION,
                )

    def symbolic_monomial_integrals(self, formula, monomial_degrees):
        """
        Generates integral and integrand all d-dimension polynomials of degree not
        exceeding $m$.
        """
        # Construct symbolic integrand.
        syms = sympy.symbols([f"x_{i}" for i, _ in enumerate(monomial_degrees)])
        syms_affine = sympy.Matrix([1, *syms])
        # Convert floats to nearest rational to improve sympy integration performance.
        rational_transform = sympy.nsimplify(
            sympy.Matrix(formula.region.affine_transformation_matrix), rational=True
        ).inv()
        transformed_syms = rational_transform * syms_affine
        transformed_syms = transformed_syms[1:, :]
        monomial = sympy.prod([s**d for s, d in zip(syms, monomial_degrees)])
        weight = _numpy_to_sympy_weights[formula.region.__class__](transformed_syms)[0]
        integrand = monomial * weight
        # Perform symbolic integration.
        # TODO: region needs to change based on formula.region type.W
        volume = sympy.integrate(weight, *[(x, -sympy.oo, sympy.oo) for x in syms])
        integral = sympy.integrate(integrand, *[(x, -sympy.oo, sympy.oo) for x in syms])
        lambdified_monomial = sympy.lambdify(syms, monomial, modules="jax")
        return volume, integral, lambda x: lambdified_monomial(*x), monomial


if __name__ == "__main__":
    absltest.main()
