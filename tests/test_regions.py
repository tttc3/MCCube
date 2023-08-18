import chex
import numpy as np
import sympy as sp
from absl.testing import absltest, parameterized

from mccube.regions import GaussianIntegrationRegion, _psd_quadratic_transformation


class RegionTest(chex.TestCase):
    """
    Ensure that cubature regions can be instantiated correctly, have appropriate
    attributes set, and correctly define their affine transformations.
    """

    def setUp(self):
        # As per example 2 pg 10-12 :cite:p:`stroud1971`, except $T$ is denoted $M$
        # fmt: off
        # x as per eq 1.4-10.
        self.x = sp.Matrix([[1, 1, 0],
                            [1,-1, 0],
                            [1, 0, 1],
                            [1, 0,-1]])
        self.A = sp.Matrix([[1,0,0],
                            [0,1,0],
                            [0,0,1]])
        # B is as per eq 1.4-11.
        self.B = sp.Matrix([[ 50,  0,  0],
                            [-20, 20,-16],
                            [-20,-16, 20]]) / 9
        self.M_inv = sp.Matrix([[    1,                 0,                 0],
                    [sp.Rational(-5,3),  sp.Rational(4,3), sp.Rational(-2,3)],
                    [sp.Rational(-5,3), sp.Rational(-2,3),  sp.Rational(4,3)]])

        self.A_COV = (2*self.A[1:,1:]).inv()
        self.B_COV = (2*self.B[1:,1:]).inv()
        self.A_MEAN = self.A[1:, 0]
        self.B_MEAN = -self.M_inv.inv()[1:, 1:] @ self.M_inv[1:, 0]
        return super().setUp()

    # fmt: off
    @parameterized.named_parameters(
        ("1D (Scalar)"         , 1, 3 * np.ones(1), 4 * np.eye(1)),
        ("2D (None Mean)"      , 2,           None, 2 * np.eye(2)),
        ("3D (None Cov)"       , 3, 2 * np.ones(3),          None),
        ("4D (None Mean + Cov)", 4,           None,          None),
    )
    # fmt: on
    def test_gaussian_init(self, dimension, mean, covariance):
        region = GaussianIntegrationRegion(dimension, mean, covariance)
        self.assertEqual(region.dimension, dimension)
        test_mean = np.zeros(dimension) if mean is None else mean
        test_covariance = np.eye(dimension) / 2 if covariance is None else covariance
        # Test defaults.
        chex.assert_trees_all_equal(region.mean, test_mean)
        chex.assert_trees_all_equal(region.covariance, test_covariance)

    def test_gaussian_affine_transformation(self):
        """
        Check that, given a desired mean and covariance for the multivariate gaussian
        weight, the correct affine_transformation_matrix is generated.
        """
        mean = sp.matrix2numpy(self.B_MEAN, dtype=float)
        cov = sp.matrix2numpy(self.B_COV, dtype=float)
        region = GaussianIntegrationRegion(2, mean=mean, covariance=cov)
        chex.assert_trees_all_close(
            region.affine_transformation_matrix,
            sp.matrix2numpy(self.M_inv.inv(), dtype=float),
        )

    def test_psd_quadratic_transformation(self):
        """
        Check that affine quadratic transformation matrix is computed correctly.
        As per example 2 pg 10-12 :cite:p:`stroud1971`, except $T$ is denoted $M$.
        """
        A = sp.matrix2numpy(self.A, dtype=float)
        B = sp.matrix2numpy(self.B, dtype=float)
        M_inv = sp.matrix2numpy(self.M_inv, dtype=float)
        M = sp.matrix2numpy(self.M_inv.inv(), dtype=float)
        # Check linear.
        M_linear = _psd_quadratic_transformation(A[1:, 1:], B[1:, 1:])
        chex.assert_tree_all_close(B[1:, 1:], M_linear.T @ A[1:, 1:] @ M_linear)
        chex.assert_trees_all_close(M_inv[1:, 1:], M_linear)
        # Check affine.
        M_affine = _psd_quadratic_transformation(A, B, affine=True)
        chex.assert_trees_all_close(M_inv, M_affine)
        # Check Inverted.
        M_linear_inv = _psd_quadratic_transformation(
            A[1:, 1:], B[1:, 1:], inverted=True
        )
        chex.assert_tree_all_close(M[1:, 1:], M_linear_inv)
        M_affine_inv = _psd_quadratic_transformation(A, B, affine=True, inverted=True)
        chex.assert_tree_all_close(M, M_affine_inv)


if __name__ == "__main__":
    absltest.main()
