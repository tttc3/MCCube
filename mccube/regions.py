"""
Defines the integration regions (measure spaces) against which :mod:`mccube.formulae` 
can be defined.
"""
from __future__ import annotations

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
    r"""Euclidean space :math:`\mathbb{R}^d` with normalized Gaussian measure.

    The probability space $(\mathbb{R}^d, \mathcal{F}, \mu)$, where $\mu$ is the standard
    d-dimensional Gaussian measure, with mean zero and identity covariance.
    I.E. $\mu(x_1, \dots, x_d) = \frac{1}{\sqrt{2\pi}^d}\exp(-\frac{x_1^2}{2} - \dots - \frac{x_d^2}{2})$.
    This is the measure against which the *"probabilist's"* Hermite polynomials are
    orthogonal.

    **Note:** if the covariance is scaled by one half, then $\mu$ is measure for which the
    *"physicist's"* Hermite polynomials are orthogonal. I.E.
    $\mu(x_1, \dots, x_d) = \frac{1}{\sqrt{\pi}^d}\exp(-x_1^2 - \dots - x_d^2)$.

    Attributes:
        dimension: dimension $d$ of the Euclidean space $\mathbb{R}^d$.
    """
    dimension: int

    @property
    def volume(self) -> float:
        r"""Measure $\mu of the entirety of $\Omega$, given by $V=1."""
        return 1.0


class WienerSpace(AbstractRegion):
    r"""Classical Wiener space with paths :math:`\omega \in C_0^0([0,1], \mathbb{R}^d)`.

    An infinite-dimensional space of continuous paths (functions) defined over the
    interval $[0,1]$, starting at zero, and taking values in $\mathbb{R}^d$. Defines a
    measure space $(C_0^0([0,1], \mathbb{R}^d), \mathcal{F}, \mu)$, where $\mathcal{F}$
    is the Borel $\sigma$-field and $\mu$ is the standard Wiener measure.

    Note: one can rescale the Wiener measure for any desired time-interval $[0, T]$,
    such that the rescaled measure $\mu^\prime = \sqrt{T}\mu$ :cite:p:`lyons2004`.

    Attributes:
        dimension: dimension $d$ of the Euclidean space in which the paths take values.
    """
    dimension: int

    @property
    def volume(self) -> float:
        r"""Measure $\mu of the entirety of $\Omega$, given by $V=1."""
        return 1.0

    @property
    def t0(self):
        """Left endpoint of the domain for the Wiener paths $[t0=0.0, t1=1.0]$."""
        return 0.0

    @property
    def t1(self):
        """Right endpoint of the domain for the Wiener paths $[t0=0.0, t1=1.0]$."""
        return 1.0
