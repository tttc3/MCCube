"""Defines cubature formulae for integrating functions over the regions (measure spaces) 
defined in :mod:`mccube.regions`.

A very limited number of formulae have been implemented for the regions in 
:mod:`mccube._regions`. Thus, any contributions of additional formulae are very much 
welcomed."""
import abc
import itertools
from collections.abc import Callable, Collection
from functools import cached_property
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from diffrax import AbstractBrownianPath
from equinox.internal import ω
from jaxtyping import ArrayLike, Shaped
from scipy.linalg import hadamard

from ._custom_types import (
    CubaturePoints,
    CubaturePointsTree,
    CubatureWeights,
    CubatureWeightsTree,
    IntScalarLike,
    RealScalarLike,
)
from ._regions import AbstractRegion, GaussianRegion, WienerSpace
from ._utils import all_subclasses


class AbstractCubature[_Region: AbstractRegion](eqx.Module, strict=True):
    r"""Abstract base class for cubature formulae.

    A concrete implementation of this class allows one to construct the :attr:`~AbstractCubature.points`
    and :attr:`~AbstractCubature.weights` of a pre-defined cubature formula over some
    specified integration :attr:`~AbstractCubature.region` (measure space).

    Attributes:
        region: an integration region (measure space) of a specified dimension.
        degree: the degree $m$ of polynomials, defined over the :attr:`~AbstractCubature.region`,
            for which the cubature formulae is an exact integrator (with respect to the
            measure $\mu$).
        sparse: indicates if the cubature points have a sparsity structure.
    """

    region: _Region
    degree: int = eqx.field(init=False)
    sparse: bool = eqx.field(init=False, default=False)

    @cached_property
    @abc.abstractmethod
    def weights(self) -> CubatureWeightsTree:
        r"""A PyTree of Cubature weights $\lambda_j \in \mathbb{R}_+$ for the measure $\mu$."""
        ...

    @cached_property
    @abc.abstractmethod
    def points(self) -> CubaturePointsTree:
        r"""A PyTree of Cubature points $x_j \in \Omega$ for the measure $\mu$."""
        ...

    @cached_property
    @abc.abstractmethod
    def point_count(self) -> IntScalarLike:
        r"""Cubature point count $n$."""
        ...

    @cached_property
    def stacked_weights(self) -> CubatureWeights:
        """:meth:`~AbstractCubature.weights` stacked into a single vector."""
        weight_array = jtu.tree_map(lambda x: np.ones(x.shape[0]), self.points)
        return np.hstack((ω(self.weights) * ω(weight_array)).ω)  # pyright: ignore

    @cached_property
    def stacked_points(self) -> CubaturePoints:
        """:meth:`~AbstractCubature.points` stacked into a single matrix."""
        return np.vstack(self.points)

    def __call__(
        self,
        integrand: Callable[[CubaturePoints], RealScalarLike],
    ) -> tuple[RealScalarLike, CubatureWeights]:
        r"""Approximately integrate some function $f$ over the integration region.

        Computes the cubature formula $Q[f] = \sum_{j=1}^{n} \lambda_j f(x_j)$.

        Args:
            integrand: the jax transformable function to integrate.

        Returns:
            Approximated integral and stakced weighted evaluations of $f$ at each
            vector $x_j$.
        """
        return evaluate_cubature(self.weights, self.points, integrand)


def evaluate_cubature(
    weights: CubatureWeightsTree,
    points: CubaturePointsTree,
    integrand: Callable[[CubaturePoints], RealScalarLike],
) -> tuple[RealScalarLike, CubatureWeights]:
    r"""Evaluate a cubature formula for a given integrand $f$.

    Args:
        weights: cubature formula weights $\lambda_j$.
        points: cubature formula points $x_j$.
        integrand: function $f$ to integrate.

    Returns:
        Computed integral and the weighted evaluation points $\lambda_j f(x_j)$.
    """
    eval_points = jtu.tree_map(
        jax.vmap(lambda c, x: c * integrand(x), [None, 0]), weights, points
    )
    eval_points = jnp.hstack(eval_points)
    return jnp.sum(eval_points), eval_points


class AbstractGaussianCubature(AbstractCubature[GaussianRegion]):
    """Abstract base class for cubature formula that are valid for the
    :class:`~mccube.regions.GaussianRegion`.

    The Gaussian region is assumed to have the "probabilist's" Hermite measure.

    Note: we assume the Gaussian measure is the normalized "probabilist's" Hermite
    weight (to be consistent with :cite:t:`victoir2004`). However, other authors, such
    as :cite:t:`stroud1971`, assume the measure is the "physicist's" Hermite measure.
    Cubature formulae defined with respect to the later measure must be appropriately
    rescaled to be compatible with the measure assumed here.
    """


class Hadamard(AbstractGaussianCubature):
    """Degree 3 Gaussian cubature from :cite:p:`victoir2004`."""

    degree: int = 3

    @cached_property
    def weights(self) -> RealScalarLike:
        return self.region.volume / self.point_count

    @cached_property
    def points(self) -> CubaturePoints:
        hadamard_matrix = hadamard(self.point_count // 2)
        new_matrix = hadamard_matrix[:, : self.region.dimension]
        return np.vstack((new_matrix, -new_matrix))

    @cached_property
    def point_count(self) -> IntScalarLike:
        r"""Cubature point count $n = 2^{\lceil{\log_2 d}\rceil + 1}$."""
        max_power = np.ceil(np.log2(self.region.dimension)) + 1
        return int(2**max_power)


class StroudSecrest63_31(AbstractGaussianCubature):
    r"""Degree 3 Gaussian cubature from :cite:p:`stroudSecrest1963`, listing :math:`E_n^{r^2}`
    3-1 (pg315) in :cite:p:`stroud1971`."""

    degree: int = 3
    sparse: bool = True

    @cached_property
    def weights(self) -> RealScalarLike:
        return self.region.volume / self.point_count

    @cached_property
    def points(self) -> CubaturePoints:
        radius = np.sqrt(self.region.dimension)
        points_symmetric = radius * np.diag(np.ones(self.region.dimension))
        points_fully_symmetric = np.vstack([points_symmetric, -points_symmetric])
        return points_fully_symmetric

    @cached_property
    def point_count(self) -> IntScalarLike:
        """Cubature point count $n = 2d$."""
        return int(2 * self.region.dimension)


class StroudSecrest63_32(AbstractGaussianCubature):
    r"""Degree 3 Gaussian cubature from :cite:p:`stroudSecrest1963`, listing :math:`E_n^{r^2}`
    3-2 (pg316) in :cite:p:`stroud1971`.

    This formula is identical to the :class:`Hadamard` formula for dimensions less than
    four. For all other dimensions, :class:`Hadamard` is strictly more efficient.
    """

    degree: int = 3

    @cached_property
    def weights(self) -> float:
        return self.region.volume / 2**self.region.dimension

    @cached_property
    def points(self) -> CubaturePoints:
        d = self.region.dimension
        radius = 1.0
        points = _generate_point_permutations(radius * np.ones(d), "R")
        return points

    @cached_property
    def point_count(self) -> IntScalarLike:
        """Cubature point count $n = 2^d$."""
        return int(2**self.region.dimension)


class StroudSecrest63_52(AbstractGaussianCubature):
    r"""Degree 5 Gaussian cubature from :cite:p:`stroudSecrest1963`, listing :math:`E_n^{r^2}`
    5-2 (pg317) in :cite:p:`stroud1971`."""

    degree: int = 5
    sparse: bool = True

    @cached_property
    def weights(self) -> CubatureWeightsTree:
        d = self.region.dimension
        V = self.region.volume
        A = 2 / (d + 2) * V
        B = (4 - d) / (2 * (d + 2) ** 2) * V
        C = 1 / (d + 2) ** 2 * V
        return (A, B, C)

    @cached_property
    def points(self) -> CubaturePointsTree:
        d = self.region.dimension
        # Rescaled from physicist's to probablist's Hermite measure.
        r2 = d + 2
        s2 = (d + 2) / 2
        A_points = np.zeros((1, d))
        B_points = np.copy(A_points)
        B_points[0, 0] = r2**0.5
        B_points = _generate_point_permutations(B_points, "FS")
        C_points = np.copy(A_points)
        C_points[0, :2] = s2**0.5
        C_points = _generate_point_permutations(C_points, "FS")
        return (A_points, B_points, C_points)

    @cached_property
    def point_count(self) -> IntScalarLike:
        """Cubature point count $n = 2d^2 + 1$."""
        return int(2 * self.region.dimension**2 + 1)


class StroudSecrest63_53(AbstractGaussianCubature):
    r"""Degree 5 Gaussian cubature from :cite:p:`stroudSecrest1963`, listing :math:`E_n^{r^2}`
    5-3 (pg317) in :cite:p:`stroud1971`. Valid for regions with dimension d > 2."""

    degree: int = 5

    def __check_init__(self):
        d = self.region.dimension
        if d < 2:
            raise ValueError(
                f"StroudSecrest63_53 is only valid for regions with d > 2; got d={d}"
            )

    @cached_property
    def weights(self) -> CubatureWeightsTree:
        d = self.region.dimension
        A = 8 * d / (d + 2) ** 2 / self.points[0].shape[0]
        B = ((d - 2) / (d + 2)) ** 2 / self.points[1].shape[0]
        return (A, B)

    @cached_property
    def points(self) -> CubaturePointsTree:
        d = self.region.dimension
        r2 = (d + 2) / 2
        s2 = (d + 2) / (d - 2)
        A_points = np.zeros((1, d))
        A_points[0, 0] = r2**0.5
        A_points = _generate_point_permutations(A_points, "FS")
        B_points = s2**0.5 * np.ones(d)
        B_points = _generate_point_permutations(B_points, "R")
        return (A_points, B_points)

    @cached_property
    def point_count(self) -> IntScalarLike:
        """Cubature point count $n = 2^d + 2d$."""
        return 2 * self.region.dimension + 2**self.region.dimension


_modes = {"R", "S", "FS"}


def _generate_point_permutations(
    point: Shaped[ArrayLike, "... d"], mode: Literal["R", "S", "FS"] = "FS"
) -> CubaturePoints:
    r"""Generate a matrix of permutations of a given $d$-dimensional point/vector.

    Args:
        point: a $d$-dimensional vector from which to generate permutations.
        mode: the type of permutations to generate, based on the notation in
            :cite:p:`Stroud1971` (see :cite:p:`cools1992` for group theoretic
            definitions):
            * "R"  - Reflected permutations $(\pm r, \pm r, dots, \pm r)$.
            * "S"  - Symmetric permutations $(r, r, \dots, 0)_S$.
            * "FS" - Fully Symmetric permutations $(r, r, \dots, 0)_{FS}$.

    Returns:
        matrix of generated permutations of the given $d$-dimensional point/vector.

    Raises:
        ValueError: invalid mode specified.
    """
    if mode not in _modes:
        raise ValueError(f"Mode must be one of {_modes}, got {mode}.")

    _point = np.atleast_1d(point)
    point_dim = np.shape(_point)[-1]
    non_zero = _point[_point != 0]
    non_zero_dim = non_zero.shape[0]

    # Handle symmetric mode where reflections aren't generated.
    reflected = np.atleast_2d(non_zero)
    if mode in {"R", "FS"}:
        reflections = itertools.product((-1, 1), repeat=non_zero_dim)
        iterable_dtype = np.dtype((_point.dtype, non_zero.shape))
        reflected = np.fromiter((r * non_zero for r in reflections), iterable_dtype)
        if mode == "R":
            return reflected

    permutations = list(itertools.permutations(np.arange(0, point_dim), non_zero_dim))
    permutation_count = len(permutations)
    permutation_index = np.arange(0, permutation_count)[:, None]
    buffer = np.zeros((permutation_count, reflected.shape[0], point_dim))
    buffer[permutation_index, :, permutations] = reflected.T
    generated_points = buffer.reshape(-1, point_dim)
    unique_generated_points = np.unique(generated_points, axis=0)
    return unique_generated_points


class AbstractWienerCubature(AbstractCubature[WienerSpace], AbstractBrownianPath):
    # Provides Diffrax path compatibility.
    @property
    def t0(self) -> RealScalarLike:
        return self.region.t0

    @property
    def t1(self) -> RealScalarLike:
        return self.region.t1

    @abc.abstractmethod
    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> CubaturePoints:
        r"""Evaluate the cubature paths $\omega_j \in C_{0,bv}^0([0,1],\R^d)$.

        The evaluated paths ($d$-dimensional vectors) are concatenated with the
        path weights to produce an augmented array with trailing dimension of $d+1$.
        This allows an AbstractWienerCubature to be used as an AbstractPath in Diffrax
        providing minor alterations are made to the SDE vector fields, to handle the
        weights.

        The remainder of this docstring is copied from :ref:`Diffrax <https://docs.kidger.site/diffrax/api/path/#diffrax.AbstractPath>`.

        Args:
            t0: Any point in $[t_0, t_1]$ at which to evaluate the paths.
            t1: If passed, then the increment from t1 to t0 is evaluated instead.
            left: Across jump points: whether to treat the path as left-continuous or
                right-continuous.

        Returns:
            If `t1` is None, returns the augmented array of the paths evaluated at `t0`
            with the cubature weights. If `t1` is passed, returns the same as before,
            but this time for the increment of the paths between `t0` and `t1`.
        """
        ...


class LyonsVictoir04_512(AbstractWienerCubature):
    """
    Defines a Cubature on d-dimensional Wiener space via linear paths starting at zero,
    and whose end points are the points of a d-dimensional `AbstractGaussianCuabture`.
    """

    gaussian_cubature: AbstractGaussianCubature

    def __init__(
        self,
        region: WienerSpace,
        gaussian_cubature: type[AbstractGaussianCubature] = Hadamard,
    ):
        self.gaussian_cubature = gaussian_cubature(GaussianRegion(region.dimension))
        self.region = region
        self.degree = gaussian_cubature.degree
        self.sparse = gaussian_cubature.sparse

    @cached_property
    def weights(self) -> CubatureWeightsTree:
        return self.gaussian_cubature.weights

    @cached_property
    def points(self) -> CubaturePointsTree:
        return self.gaussian_cubature.points

    @cached_property
    def point_count(self) -> IntScalarLike:
        return self.gaussian_cubature.point_count

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
        particles = coeff * self.stacked_points
        weights = jnp.expand_dims(self.stacked_weights, -1)
        return jnp.concatenate([particles, weights], -1)


# Must be at the bottom such that the above subclasses are included.
builtin_cubature_registry: set[type[AbstractCubature]] = all_subclasses(
    AbstractCubature
)
"""A searchable registry of all cubature formulae that are subclasses of 
:class:`AbstractCubature`."""


def search_cubature_registry(
    region: AbstractRegion,
    degree: int | None = None,
    sparse_only: bool = False,
    minimal_only=False,
    searchable_formulae: Collection[type[AbstractCubature]] = builtin_cubature_registry,
) -> list[AbstractCubature]:
    """Returns a list of cubature formulae of a specified degree for a given region.

    Args:
        region: the region for which the cubature formulae must be of degree $m$.
        degree: the required degree $m$ of cubature formulae. Ignored if none.
        sparse_only: if to only select cubature formulae with sparse cubature points.
        minimal_only: if to only return formulae with minimal point count.
        searchable_formulae: collection from which to search for suitable cubature
            formulae. Defaults to the :data:`builtin_cubature_registry`.
    """
    selected_formulae = list()
    minimum_point_count = float("inf")
    for f in searchable_formulae:
        degree_condition = (f.degree != degree) and (degree is not None)
        sparse_condition = sparse_only and not f.sparse
        if degree_condition or sparse_condition:
            continue
        # Try to instantiate the formula in order to establish a point count.
        try:
            formula = f(region)
        except ValueError:
            continue
        current_point_count = formula.point_count
        if minimal_only:
            if current_point_count > minimum_point_count:
                continue
            elif current_point_count < minimum_point_count:
                selected_formulae = [formula]
                minimum_point_count = current_point_count
            else:
                selected_formulae.append(formula)
        else:
            selected_formulae.append(formula)
    return sorted(selected_formulae, key=lambda f: f.point_count)  # pyright: ignore
