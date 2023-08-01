from __future__ import annotations

import abc
from typing import Callable, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int, PyTree
from scipy.linalg import hadamard as scipy_hadamard

# TODO: include the error estimates for each cubature formula where available.
# TODO: include an integration function.
# TODO: consider moving into separate package.


class _CubatureFormulaRegistryMeta(eqx._module._ModuleMeta):
    _registry = set()

    def __init__(cls, name, bases, attrs):
        super(_CubatureFormulaRegistryMeta, cls).__init__(name, bases, attrs)
        if name not in ["AbstractCubatureFormula", "_InitableModule"]:
            _CubatureFormulaRegistryMeta._registry.add(cls)

    @classmethod
    def get_registered_formulae(cls):
        return cls._registry


formula_registry = _CubatureFormulaRegistryMeta


class AbstractIntegrationRegion(eqx.Module):
    dimension: int

    @abc.abstractmethod
    def weight_function(x: ArrayLike) -> Array:
        ...

    @abc.abstractproperty
    def volume(self) -> float:
        ...


class EuclideanIntegrationRegion(AbstractIntegrationRegion):
    """N-dimensional Euclidean integration region, $\mathbb{R}^n$."""

    def weight_function(self, x: ArrayLike) -> Array:
        return 1.0

    @property
    def volume(self) -> float:
        return np.inf


class GaussianIntegrationRegion(AbstractIntegrationRegion):
    """N-dimensional unnormalized gaussian weighted Euclidean integration region.

    Notated as $E_n^{r^2}$ in :cite:p:`stroud1971`, this region represents integration
    over an N-dimensional Euclidean space, weighted by $\exp{-x_1^2 - x_2^2 \dots -x_d^2}$.
    """  # noqa: E501

    def weight_function(self, x: ArrayLike) -> Array:
        return np.exp(-np.sum(x**2, axis=-1))

    @property
    def volume(self) -> float:
        return np.pi ** (self.dimension / 2)


class AbstractCubatureFormula(eqx.Module, metaclass=_CubatureFormulaRegistryMeta):
    r"""Abstract base class for cubature formulae.

    Attributes:
        dimension:

        degree: polynomial degree $D$ for which the cubature formula is an exact
            integrator.
        sparse: if cubature formula produces a sparse matrix of cubature vectors.
        valid_dimensions: a range or specific set of vector-filed dimensions $d$, for
            which the cubature is an exact integrator of polynomials of degree $D$.
    """
    dimension: int
    degree: int = 0
    sparse: bool = False
    region: AbstractIntegrationRegion = EuclideanIntegrationRegion
    valid_dimensions: Tuple[int, int] | Sequence[Int] = (0, np.inf)

    def __init__(self, dimension):
        self.dimension = self._validate_dimension(dimension) and dimension
        self.region = self.region(self.dimension)

    @classmethod
    def _validate_dimension(cls, dimension: int) -> bool:
        """If cubature formula is valid for the given dimension.

        Args:
            dimension: dimension of the vector-field over which the cubature formula
                should be an exact integrator.

        Raises:
            ValueError: dimension not in range/set specified by `self.valid_dimensions`.
        """
        valid_dims = cls.valid_dimensions
        min_dim, max_dim = min(valid_dims), max(valid_dims)
        in_range = min_dim <= dimension <= max_dim
        # If valid_dims is length 2, then treat as range (lower, upper), else treat as
        # list/sequence of allowed dimensions.
        in_list = dimension in cls.valid_dimensions if len(valid_dims) < 2 else True
        if in_range and in_list:
            return True
        in_range_err_msg = f"{min_dim} <= dimension <= {max_dim}"
        in_list_err_msg = f"dimension in {valid_dims}"
        err_msg = in_range_err_msg if len(valid_dims) < 2 else in_list_err_msg
        raise ValueError(
            "Cubature formula cannot exactly integrate a vector-field of",
            f"the give dimension; expected {err_msg}, got {dimension}",
        )

    @abc.abstractproperty
    def coefficients(self) -> PyTree[float]:
        ...

    @abc.abstractproperty
    def vector_count(self) -> int:
        ...

    @abc.abstractproperty
    def vectors(self) -> PyTree[Float[Array, "k d"]]:
        ...

    def __call__(self, func: Callable[[Float[ArrayLike, " d"]], Float[Array, ""]]):
        """Approximately integrate some function $f$ over the integration region.

        Args:
            f: the jax transformable function to integrate.

        Returns:
            Approximated integral and weighted evaluations of $f$ at each vector $e_i$.
        """
        func_vmap = jax.vmap(func, 0)
        eval_points = jnp.vstack(
            jtu.tree_map(lambda c, v: c * func_vmap(v), self.coefficients, self.vectors)
        )
        return sum(eval_points), eval_points


class AbstractGaussianCubatureFormula(AbstractCubatureFormula):
    """Cubature formulae for a :class:`GaussianIntegrationRegion`."""

    region: AbstractIntegrationRegion = GaussianIntegrationRegion


# TODO: add reference here.
class Hadamard(AbstractGaussianCubatureFormula):
    degree: int = 4

    @property
    def coefficients(self) -> float:
        return self.region.volume / self.vector_count

    @property
    def vector_count(self) -> int:
        max_power = np.ceil(np.log2(self.dimension))
        max_dim = np.power(2, max_power)
        return max_dim * 2

    @property
    def vectors(self) -> Int[Array, "k d"]:
        hadamard_matrix = scipy_hadamard(self.vector_count // 2)
        new_matrix = hadamard_matrix[:, : self.dimension]
        return np.vstack((new_matrix, -new_matrix))


class StroudSecrest63_31(AbstractGaussianCubatureFormula):
    r"""Degree 3 cubature formula from :cite:p:`stroudSecrest1963`, listing $E^{r^2}_n$
    3-1. (pg315) in :cite:p:`stroud1971`."""

    degree: int = 3
    sparse: bool = True

    @property
    def coefficients(self) -> float:
        return self.region.volume / self.vector_count

    @property
    def vector_count(self) -> int:
        return 2 * self.dimension

    @property
    def vectors(self) -> Float[Array, "k d"]:
        radius = np.sqrt(self.dimension / 2)
        points_symmetric = radius * np.diag(np.ones(self.dimension))
        points_fully_symmetric = np.vstack([points_symmetric, -points_symmetric])
        return points_fully_symmetric * np.sqrt(2)


def minimal_cubature_formula(
    dimension: int,
    degree: int = 3,
    region=GaussianIntegrationRegion,
    sparse_only: bool = False,
    return_all: bool = False,
) -> AbstractCubatureFormula | Sequence[AbstractCubatureFormula]:
    """Cubature formula with the fewest cubature vectors for a given degree.

    Args:
        dimension: dimension of the space over which the cubature matrix must integrate.
        degree: polynomial degree for which the cubature matrix is an exact integrator.
        sparse_only: only consider cubature formulas that produce sparse vectors.
        return_all: if multiple
    """
    formulae = formula_registry.get_registered_formulae()
    if sparse_only:
        formulae = {f for f in formulae if f.sparse}
    # Need to convert set to array for indexing only the lowest count formulae.
    valid_formulae = np.fromiter(
        (
            f(dimension)
            for f in formulae
            if f.degree >= degree and issubclass(f.region, region)
        ),
        dtype=np.object_,
    )
    formulae_points = np.fromiter((f.vector_count for f in valid_formulae), dtype=int)
    minimal_formulae = np.where(formulae_points == formulae_points.min())
    minimal_formulae = valid_formulae[minimal_formulae]

    if return_all:
        return minimal_formulae
    return minimal_formulae[0]
