"""Defines the integration regions (measure spaces) against which [`AbstractCubatures`][mccube.AbstractCubature] 
can be defined."""

import abc

import equinox as eqx


class AbstractRegion(eqx.Module):
    r"""Abstract base class for all (weighted) integration regions.

    Integration regions are measure spaces $(\Omega, \mathcal{F}, \mu)$, where
    $\mathcal{F}$ is the Borel $\sigma$-algebra on the region $\Omega$, and $\mu$ is
    some suitable positive Borel (probability) measure that 'weights' the region.

    Attributes:
        dimension: dimension $d$ of the integration region $\Omega$.
    """

    dimension: int

    @abc.abstractproperty
    def volume(self) -> float:
        r"""Measure $\mu$ of the entirety of $\Omega$, denoted by $V$."""
        ...


class GaussianRegion(AbstractRegion):
    r"""Euclidean space $\mathbb{R}^d$ with Gaussian probability measure.

    The probability space $(\mathbb{R}^d, \mathcal{F}, \mu)$, where $\mu$ is the standard
    d-dimensional Gaussian measure, with mean zero and identity covariance.
    I.E. $\mu(x_1, \dots, x_d) = (2\pi)^{-d/2}\exp(-\frac{x_1^2}{2} - \dots - \frac{x_d^2}{2})$.
    This is the measure against which the *"probabilist's"* Hermite polynomials are
    orthogonal.

    **Note:** if the covariance is scaled by one half, then $\mu$ is the measure for
    which the *"physicist's"* Hermite polynomials are orthogonal. I.E.
    $\mu(x_1, \dots, x_d) = \pi^{-d/2}\exp(-x_1^2 - \dots - x_d^2)$.

    Attributes:
        dimension: dimension $d$ of the Euclidean space $\mathbb{R}^d$.
    """

    dimension: int

    @property
    def volume(self) -> float:
        r"""Measure $\mu$ of the entirety of $\Omega$, given by $V=1$."""
        return 1.0
