"""
Defines the integration regions (measure spaces) against which :mod:`mccube.formulae` 
can be defined.
"""
from __future__ import annotations

import abc
from functools import cached_property
from typing import ClassVar

import equinox as eqx
import numpy as np
from jaxtyping import Array, ArrayLike, Float


class AbstractRegion(eqx.Module):
    r"""Abstract base class for all (weighted) integration regions.

    Integration regions are measure spaces $(\Omega, \mathcal{F}, \mu)$, where
    $\mathcal{F}$ is the Borel $\sigma$-algebra on the region $\Omega$, and $\mu$ is
    some suitable Borel (probability) measure that 'weights' the region. It is assumed
    that the measure $\mu(x) = \mu^\prime(Mx)$, an affine transformation of the
    canonical measure for the region $\mu^\prime$.


    Attributes:
        dimension: dimension $d$ of the integration region $\Omega$.
        normalized: if the measure $\mu$ is a probability measure (normalized to have
            a volume of one).
    """
    dimension: int
    normalized: bool

    @cached_property
    def volume(self) -> float:
        r"""Measure $\mu$ of $\Omega$, given by $V=V^\prime\det(M)$."""
        return self._volume * self.affine_transformation[1]

    @abc.abstractproperty
    def _volume(self) -> float:
        r"""Measure $\mu^\prime$ of $\Omega$, denoted $V^\prime$."""
        ...

    @cached_property
    def affine_transformation(self) -> (Float[Array, "d+1 d+1"], float):
        r"""A potentially normalized affine transformation M on $\Omega$ and its determinant $\det(M)$.

        Constructs the potentially normalized affine transformation $M$ from the
        canonical measure for the region $\mu^\prime$, to the potentially normalized
        (probability) measure $\mu$, given the unnormalized affine transformation
        $M^\prime$. If :attr:`~AbsractRegion.normalized` is set, then the linear
        component of $M = cM^\prime$ where $c = (V^\prime \det(M^\prime))^{-1/d}$,
        otherwise $M = M^\prime$.
        """
        matrix, det = self._affine_transformation
        if self.normalized:
            # Rescale such that the linear sub-matrix of the affine transformation has a
            # determinant of 1/self._volume.
            target_det = 1 / self._volume
            normalization_factor = (target_det / det) ** (1 / self.dimension)
            matrix[1:, 1:] *= normalization_factor
            det = target_det
        return matrix, det

    @abc.abstractproperty
    def _affine_transformation(self) -> (Float[Array, "d+1 d+1"], float):
        r"""An affine transformation $M^\prime$ on $\Omega$ and its determinant $\det(M^\prime)$.

        Constructs the affine transformation $M^\prime$ from the canonical measure for
        the region $\mu^\prime$ to the unnormalized measure $\mu(x) = \mu^\prime(M^\prime x)$.
        """
        ...


class GaussianRegion(AbstractRegion):
    r"""Euclidean space :math:`\mathbb{R}^d` with (unnormalized) Gaussian measure.

    Denoted as $E_n^{r^2}$ in :cite:p:`stroud1971`, this integration region represents
    the measure space $(\mathbb{R}^d, \mathcal{F}, \mu)$, where $\mu$ is a potentially
    normalized affine transformation of the unnomalized Gaussian measure, with mean zero
    and diagonal covariance of one half, given by $\mu^\prime(x_1, \dots, x_d) = \exp(-x_1^2 -\dots-x_d^2)$.

    Note: the canonical measure $\mu^\prime$ is the weight for which the "physicist's"
    $d$-dimensional Hermite polynomials are orthogonal.

    Attributes:
        dimension: dimension $d$ of the Euclidean space $\mathbb{R}^d$.
        mean: mean (location) parameter for the $d$-dimensional Gaussian measure $\mu$;
            the canonical measure $\mu^\prime$ has mean given by the $d$-dimensional
            zero vector, $\boldsymbol{0}$.
        covariance: covariance (scale) parameter for the $d$-dimensional Gaussian
            measure $\mu$; the canonical measure $\mu^\prime$ has covariance given by
            the $d$-dimensional identity matrix times one half, $0.5*I_d$.
        normalized: if to normalize the affine transformed measure $\mu^\prime$, such
            that $\mu$ is a $d$-dimensional Gaussian probability measure.
    """
    dimension: int
    mean: Float[ArrayLike, " d"]
    covariance: Float[ArrayLike, "d d"]
    normalized: bool
    _default_mean: Float[ArrayLike, " d"]
    _default_cov: Float[ArrayLike, "d d"]

    def __init__(
        self,
        dimension: int,
        mean: None | Float[ArrayLike, " d"] = None,
        covariance: None | Float[ArrayLike, "d d"] = None,
        normalized: bool = False,
    ):
        r"""Construct a $d$-dimensional Gaussian region.

        Parametrizes the :attr:`~GaussianRegion.affine_transformation` such that
        $\mu^\prime$ (the affine transformed measure) is the $d$-dimensional Gaussian
        measure with the specified mean and covariance.

        Args:
            dimension: dimension $d$ of the Euclidean space $\mathbb{R}^d$.
            mean: if specified, mean should be a $d$-dimensional vector; implicitly zero
                if None.
            covariance: if specified, covariance should be a $d \times d$ matrix;
                implicitly diag(1/2) if None.
            normalized: if to normalize the Gaussian measure.
        """
        self._default_mean = np.zeros(dimension)
        self._default_cov = np.eye(dimension) / 2

        self.dimension = dimension
        self.mean = self._default_mean if mean is None else mean
        self.covariance = self._default_cov if covariance is None else covariance
        self.normalized = normalized

    @property
    def _volume(self) -> float:
        r"""Measure $\mu^\prime$ of $\Omega$, given by $V^\prime=\pi^{d/2}$."""
        return np.pi ** (self.dimension / 2)

    @property
    def _affine_transformation(self) -> (Float[Array, "d d"], float):
        r"""An affine transformation $M$ on $\Omega$ and its determinant $\det(M)$.

        Computes the appropriate affine transformation to yield an implied measure
        $\mu^\prime$ with the attribute specified mean and covariance, given the
        mean and covariance for the canonical measure $\mu$.
        """
        transform_cov, det = psd_quadratic_transformation(
            self._default_cov, self.covariance
        )
        transform = np.eye(self.dimension + 1)
        transform[1:, 1:] = transform_cov
        transform[1:, 0] = np.squeeze(self.mean)
        return (transform, det)


class StandardGaussianRegion(GaussianRegion):
    r"""Euclidean space :math:`\mathbb{R}^d` with normalized Standard Gaussian measure.

    While the :class:`GuassianRegion` defines a measure for which the "physicist's"
    $d$-dimensional Hermite polynomials are orthogonal, this region defines a measure
    for which the "probabilist's" $d$-dimensional Hermite polynomials are orthogonal.

    Note: this class is just a specific parameterization of the :class:`GaussianRegion`
    where the :attr:`~GaussianRegion.mean` is zero and :attr:`~GaussianRegion.covariance`
    is the $d \times d$ identity matrix, $I_d$.
    """

    def __init__(self, dimension: int) -> None:
        r"""Construct a $d$-dimensional Standard Gaussian region.

        Parametrizes the :attr:`~GaussianRegion.affine_transformation` such that
        $\mu^\prime$ (the affine transformed measure) is the normalized $d$-dimensional
        Gaussian with :attr:`~GaussianRegion.mean` zero and :attr:`~GaussianRegion.covariance`
        given by the $d \times d$ identity matrix, $I_d$.

        Args:
            dimension: dimension $d$ of the Euclidean space $\mathbb{R}^d$.
        """
        return super().__init__(
            dimension,
            mean=np.zeros(dimension),
            covariance=np.eye(dimension),
            normalized=True,
        )


class WienerSpace(AbstractRegion):
    r"""Classical Wiener space with paths :math:`\omega \in C_0^0([0,T], \mathbb{R}^d)`.

    An infinite-dimensional space of continuous paths (functions) defined over the
    interval $[0,T]$, starting at zero, and taking values in $\mathbb{R}^d$. Defines a
    measure space $(C_0^0([0,T], \mathbb{R}^d), \mathcal{F}, \mu)$, where $\mathcal{F}$
    is the Borel $\sigma$-field and $\mu$ is the standard Wiener measure.

    The canonical measure $\mu^\prime$ is defined for the Wiener space over the unit
    interval and, for any desired $T > 0$, the measure $\mu = \sqrt{T}\mu^\prime$,
    thanks to the scaling properties of Wiener space :cite:p:`lyons2004`.

    Attributes:
        dimension: dimension $d$ of the Euclidean space in which the paths take values.
        dt: the width of the time-domain $[0, T]$ on which the Wiener space is defined.
            As the time-domain is defined to start at zero, $dt = T$.
    """
    dimension: int
    dt: float = 1.0
    normalized: ClassVar[bool] = True

    @property
    def _volume(self) -> float:
        return 1.0

    @property
    def _affine_transformation(self) -> (float, float):
        dt = self.dt
        t_affine = np.array([[1, 0], [0, dt**0.5]])
        return t_affine, 1.0


def psd_quadratic_transformation(
    A: Float[Array, "d d"], B: Float[Array, "d d"], affine=False, inverted=False
) -> (Float[Array, "d d"], float):
    r"""Compute transformation matrix from A to B.

    Args:
        A: the current transformation $A$. The linear component of this matrix must be
            a scalar multiple of the Identity matrix, and the affine component zero.
        B: the target transformation $B = M^T A M$. The linear component must be
            positive definite.
        affine: indicates if to treat A and B as affine transformation matrices.
        inverted: indicates if to return $M^{-1}$. Avoids repeating expensive inversions.

    Returns:
        The (affine) transformation matrix $M$ and the determinant $\det(M)$.
    """

    if affine:
        A_translate = A[1:, 0]
        A = A[1:, 1:]
        B_translate = B[1:, 0]
        B = B[1:, 1:]
    D_B, P = np.linalg.eigh(B)
    D = np.sqrt(D_B / np.diag(A))
    det = np.prod(D)  # Matrix cookbook pg6 equation (18)
    M_quadratic = P @ np.diag(D) @ P.T
    M_quadratic_inv = P @ np.diag(1 / D) @ P.T
    if not affine:
        if inverted:
            return M_quadratic_inv, 1 / det
        return M_quadratic, det
    M_translate = (M_quadratic_inv @ B_translate - A_translate) / 2
    M_affine = np.eye(A.shape[0] + 1)
    if inverted:
        det = 1 / det
        M_quadratic = M_quadratic_inv
        M_translate = -M_quadratic_inv @ M_translate
    M_affine[1:, 1:] = M_quadratic
    M_affine[1:, 0] = M_translate
    return M_affine, det
