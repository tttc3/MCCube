import chex

import numpy as np

from absl.testing import absltest, parameterized

from mccube.regions import _psd_quadratic_transformation, GaussianIntegrationRegion


class RegionTest(chex.TestCase):
    """
    Ensure that cubature regions can be instantiated correctly, have appropriate
    attributes set, and correctly define their affine transformations.
    """

    # fmt: off
    @parameterized.named_parameters(
        ("1D (Scalar)"         , 1, 3 * np.ones(1), 4 * np.eye(1)),
        ("2D (None Mean)"      , 2,           None, 2 * np.eye(2)),
        ("3D (None Cov)"       , 3, 2 * np.ones(3),          None),
        ("4D (None Mean + Cov)", 3, 2 * np.ones(3),          None),
    )
    # fmt: on
    def test_gaussian(self, dimension, mean, covariance):
        region = GaussianIntegrationRegion(dimension, mean, covariance)
        self.assertEqual(region.dimension, dimension)
        test_mean = np.ones(dimension) if mean is None else mean
        test_covariance = np.eye(dimension) / 2 if covariance is None else covariance
        chex.assert_trees_all_equal(region.mean, test_mean)
        chex.assert_trees_all_equal(region.covariance, test_covariance)

        test_affine = np.eye(dimension + 1)
        test_affine[1:, 1:] = test_covariance
        test_affine[1:, 0] = test_mean

        canonical_affine = np.eye(dimension + 1)
        canonical_affine[1:, 1:] = np.eye(dimension) / 2
        test_transform = _psd_quadratic_transformation(canonical_affine, test_affine)
        chex.assert_trees_all_equal(region.affine_transformation_matrix, test_transform)

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
