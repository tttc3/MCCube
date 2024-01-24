"""Defines custom paths for performing MCC in diffrax.

See [`diffrax.AbstractPath`][] for futher information on the path API.
"""
import equinox as eqx
import jax.numpy as jnp
from diffrax import AbstractPath

from ._custom_types import CubaturePoints, RealScalarLike, CubatureWeights
from ._formulae import AbstractGaussianCubature


class AbstractCubaturePath(AbstractPath):
    """Abstract base class for paths that define a cubature on Wiener space [`@lyons2004`].

    ??? cite "Reference: [`@lyons2004`]"

        ```bibtex
        @article{lyons2004,
          title     = {Cubature on Wiener Space},
          author    = {Lyons, Terry and Victoir, Nicolas},
          year      = {2004},
          journal   = {Proceedings: Mathematical, Physical and Engineering Sciences},
          publisher = {The Royal Society},
          number    = {2041},
          volume    = {460},
          pages     = {169--198},
          issn      = {13645021},
          url       = {https://www.jstor.org/stable/4143098}
        }
        ```

    Attributes:
        weights: a vector of weights associated with the collection of cubature control
            paths.
    """

    weights: eqx.AbstractVar[CubatureWeights]


class LocalLinearCubaturePath(AbstractCubaturePath):
    r"""Piecewise linear cubature paths.

    The paths $f(t_0, t_1) = \sqrt{(t_1 - t_0)} M$, where $M$ is the matrix of cubature
    points for a given [`mccube.AbstractGaussianCubature`][].
    """

    gaussian_cubature: AbstractGaussianCubature
    t0 = 0.0
    t1 = 1.0

    def __init__(self, gaussian_cubature: AbstractGaussianCubature):
        """
        Args:
            gaussian_cubature: an instance of an [`mccube.AbstractGaussianCubature`][],
                whose points given the matrix $M$.
        """
        self.gaussian_cubature = gaussian_cubature

    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> CubaturePoints:
        del left
        if t1 is None:
            t1 = t0
            t0 = self.t0
        dt = t1 - t0
        # Avoid infinite gradient at zero.
        coeff = jnp.where(t0 == t1, 0.0, jnp.sqrt(dt))
        points = coeff * self.gaussian_cubature.stacked_points
        return points

    @property
    def weights(self) -> CubatureWeights:  # type: ignore
        """Vector of cubature weights associated with the cubature control paths."""
        return self.gaussian_cubature.stacked_weights
