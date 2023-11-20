import chex
import numpy as np
import sympy as sp
from absl.testing import absltest, parameterized

import jax.tree_util as jtu

from mccube.regions import (
    GaussianRegion,
    StandardGaussianRegion,
    WienerSpace,
    psd_quadratic_transformation,
)


class GaussianRegionTests(chex.TestCase):
    """Tests defining the behavior of :class:`GaussianRegion` and :class:`StandardGaussianRegion`."""

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
        ("1D (Scalar)"                    , 1, 3 * np.ones(1), 4 * np.eye(1), False),
        ("2D (None Mean)"                 , 2,           None, 2 * np.eye(2), False),
        ("3D (None Cov)"                  , 3, 2 * np.ones(3),          None, False),
        ("4D (None Mean + Cov)"           , 4,           None,          None, False),
        ("1D (Normalized Scalar)"         , 1, 3 * np.ones(1), 4 * np.eye(1),  True),
        ("2D (Normalized None Mean)"      , 2,           None, 2 * np.eye(2),  True),
        ("3D (Normalized None Cov)"       , 3, 2 * np.ones(3),          None,  True),
        ("4D (Normalized None Mean + Cov)", 4,           None,          None,  True),
    )
    # fmt: on
    def test_gaussian_init(self, dimension, mean, covariance, normalized):
        """
        Check that attributes are initialized properly:
            - dimension,
            - normalized (expected default=False),
            - mean (expected default=np.zeros(dimension)),
            - covariance (expected default=np.eye(dimension) / 2).

        Check that properties give the expected values:
            - volume (expected 1.0 when normalized=True; else expected det(M) pi^(d/2),
            as per :cite:`stroud1971` (pg 222, Section 7.9)).

        Assumptions:
            - :method:`test_gaussian_affine_transformation` passes.
        """
        region = GaussianRegion(dimension, mean, covariance, normalized)

        ## Check attributes.
        self.assertEqual(region.dimension, dimension)
        # Check defaults
        test_normalized = False if normalized is None else normalized
        test_mean = np.zeros(dimension) if mean is None else mean
        test_covariance = np.eye(dimension) / 2 if covariance is None else covariance
        self.assertEqual(region.normalized, test_normalized)
        chex.assert_trees_all_equal(region.mean, test_mean)
        chex.assert_trees_all_equal(region.covariance, test_covariance)

        ## Check properties
        detM = region.affine_transformation[1]
        expected_volume = detM * np.pi ** (region.dimension / 2)
        if region.normalized:
            self.assertAlmostEqual(expected_volume, 1.0)
        self.assertAlmostEqual(region.volume, expected_volume)

    def test_gaussian_affine_transformation(self):
        """
        Check that, given a desired mean and covariance for the multivariate gaussian
        measure, the correct affine_transformation is generated.
        """
        mean = sp.matrix2numpy(self.B_MEAN, dtype=float)
        cov = sp.matrix2numpy(self.B_COV, dtype=float)
        region = GaussianRegion(2, mean=mean, covariance=cov)
        chex.assert_trees_all_close(
            region.affine_transformation[0],
            sp.matrix2numpy(self.M_inv.inv(), dtype=float),
        )
        chex.assert_trees_all_close(
            region.affine_transformation[1], float(self.M_inv.inv().det())
        )

    @parameterized.named_parameters(
        ("1D", 1),
        ("2D", 2),
        ("3D", 3),
        ("4D", 4),
    )
    def test_standard_gaussian_region_init(self, dimension):
        """
        Check that a StandardGaussianRegion is equivalent to a normalized GaussianRegion
        with mean zero and diagonal covariance of one.

        Assumptions:
            - :method:`test_gaussian_init` passes.
        """
        standard_region = StandardGaussianRegion(dimension)
        region = GaussianRegion(
            dimension,
            mean=np.zeros(dimension),
            covariance=np.eye(dimension),
            normalized=True,
        )
        chex.assert_trees_all_equal(
            jtu.tree_leaves(standard_region), jtu.tree_leaves(region)
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
        M_linear, det_linear = psd_quadratic_transformation(A[1:, 1:], B[1:, 1:])
        chex.assert_trees_all_close(B[1:, 1:], M_linear.T @ A[1:, 1:] @ M_linear)
        chex.assert_trees_all_close(M_inv[1:, 1:], M_linear)
        # Check affine.
        M_affine, det_affine = psd_quadratic_transformation(A, B, affine=True)
        chex.assert_trees_all_close(M_inv, M_affine)
        # Check Inverted.
        M_linear_inv, det_linear_inv = psd_quadratic_transformation(
            A[1:, 1:], B[1:, 1:], inverted=True
        )
        chex.assert_trees_all_close(M[1:, 1:], M_linear_inv)
        M_affine_inv, det_affine_inv = psd_quadratic_transformation(
            A, B, affine=True, inverted=True
        )
        chex.assert_trees_all_close(M, M_affine_inv)


class WienerRegionTests(chex.TestCase):
    """Tests defining the behavior of :class:`WienerSpace`."""

    # fmt: off
    @parameterized.named_parameters(
        ("1D"          , 1, None),
        ("2D"          , 2, None),
        ("3D"          , 3, None),
        ("4D"          , 4, None),
        ("1D (dt=001)" , 1, 0.01),
        ("2D (dt=02)"  , 2,  0.2),
        ("3D (dt=3)"   , 3,  3.0),
        ("4D (dt=40)"  , 4, 40.0),
    )
    # fmt: on
    def test_wiener_space_init(self, dimension, dt):
        """
        Check that attributes are initialized properly:
            - dimension,
            - dt (expected default=1.0),
            - normalized (expected always True).

        Check that properties give the expected values:
            - volume (expected always 1.0, by definition of always normalized=True).
            - affine_transformation (expected scale by M = sqrt(dt) and detM = 1).
        """

        args = [dimension]
        test_dt = 1.0
        if dt is not None:
            test_dt = dt
            args.append(dt)
        region = WienerSpace(*args)

        ## Check attributes
        self.assertEqual(region.dimension, dimension)
        self.assertEqual(region.dt, test_dt)
        self.assertEqual(region.normalized, True)

        ## Check properties
        self.assertEqual(region.volume, 1.0)
        test_affine = (np.array([[1, 0], [0, test_dt**0.5]]), 1.0)
        chex.assert_trees_all_equal(region.affine_transformation, test_affine)


if __name__ == "__main__":
    absltest.main()
