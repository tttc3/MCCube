"""Defines abstract and concrete integration regions."""
from __future__ import annotations

import abc

import equinox as eqx
import numpy as np
from jaxtyping import Array, ArrayLike, Float


class AbstractIntegrationRegion(eqx.Module):
    """Abstract base class for all integration regions.

    Attributes:
        dimension: dimension $d$ of the integration region $\Omega$.
    """

    dimension: int

    @abc.abstractmethod
    def weight_function(x: ArrayLike) -> Array:
        """Integration weight function/distribution."""
        ...

    @abc.abstractproperty
    def volume(self) -> float:
        """Volume of the weighted integration region."""
        ...

    @abc.abstractproperty
    def affine_transformation_matrix(self) -> Float[ArrayLike, "d+1 d+1"]:
        """Affine transformation matrix from the canonical region."""
        ...


class GaussianIntegrationRegion(AbstractIntegrationRegion):
    r"""d-dimensional unnormalized gaussian weighted Euclidean integration region.

    Notated as $E_n^{r^2}$ in :cite:p:`stroud1971`, this region represents integration
    over an d-dimensional Euclidean space, weighted by $\exp{-x_1^2 \dots -x_d^2}$.

    Attributes:
        dimension: dimension $d$ of the integration region $\Omega$.
        mean: mean parameter for the Gaussian weight function.
        covariance: covariance parameter for the Gaussian weight function.
        affine_transformation_matrix: a matrix specifying an affine transformation of
            the integration region.
    """
    mean: Float[ArrayLike, " d"]
    covariance: Float[ArrayLike, "d d"]

    def __init__(
        self,
        dimension: int,
        mean: None | Float[ArrayLike, " d"] = None,
        covariance: None | Float[ArrayLike, "d d"] = None,
    ):
        """Initialise Gaussian integration region with specified mean and covariance.

        Args:
            dimension: dimension $d$ of the integration region $\Omega$.
            mean: if specified, mean should be a $d$ vector; implicitly zero if None.
            covariance: if specified, covariance should be a $d \times d$ matrix;
                implicitly diag(1/2) if None.
        """
        self.dimension = dimension
        self.mean = np.ones(dimension) if mean is None else mean
        self.covariance = np.eye(dimension) / 2 if covariance is None else covariance

    def weight_function(self, x: ArrayLike) -> Array:
        return np.exp(-np.sum(x**2, axis=-1))

    @property
    def volume(self) -> float:
        return np.pi ** (self.dimension / 2)

    @property
    def affine_transformation_matrix(self):
        default_cov = np.eye(self.dimension) / 2
        target_cov = self.covariance

        default_mean = np.zeros(self.dimension)
        target_mean = self.mean
        default_transform = np.eye(self.dimension + 1)
        default_transform[1:, 0] = default_mean
        default_transform[1:, 1:] = default_cov

        target_transform = np.copy(default_transform)
        target_transform[1:, 0] = target_mean
        target_transform[1:, 1:] = target_cov

        return _psd_quadratic_transformation(default_transform, target_transform)


def _psd_quadratic_transformation(A, B, affine=True):
    """Compute transformation matrix from A to B.

    Args:
        A: the current transformation $A$. The linear component of this matrix must be
            a scalar multiple of the Identity matrix, and the affine component zero.
        B: the target transformation $B = M^T A M$. The linear component must be
            positive definite.
        affine: idicates if to treat A and B as affine transformation matrices.

    Returns:
        The (affine) transformation matrix $M$.
    """

    if affine:
        A_linear = A[1:, 0]
        A = A[1:, 1:]
        B_linear = B[1:, 0]
        B = B[1:, 1:]
    D_B, P = np.linalg.eigh(B)
    D = np.sqrt(D_B / np.diag(A))
    M_quadratic = P @ np.diag(D) @ P.T
    if not affine:
        return M_quadratic
    M_quadratic_inv = P @ np.diag(1 / D) @ P.T
    M_linear = (M_quadratic_inv @ B_linear - A_linear) / 2
    M_affine = np.eye(A.shape[0] + 1)
    M_affine[1:, 1:] = M_quadratic
    M_affine[1:, 0] = M_linear
    return M_affine
