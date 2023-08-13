from __future__ import annotations

import abc
from typing import Callable, Sequence, Tuple


import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int
from scipy.linalg import hadamard as scipy_hadamard

from mccube.regions import AbstractIntegrationRegion, GaussianIntegrationRegion

# TODO: include the error estimates for each cubature formula where available.
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


class AbstractCubatureFormula(eqx.Module, metaclass=_CubatureFormulaRegistryMeta):
    r"""Abstract base class for cubature formulae.

    Attributes:
        dimension: dimension $d$ of the integration region $\Omega$.
        region: generalized dimension integration region over which the cubature
            formula is valid.
        degree: polynomial degree $m$ for which the cubature formula is an exact
            integrator.
        sparse: if cubature formula produces a sparse matrix of cubature vectors.
        valid_dimensions: a range or specific set of dimensions $d$, for which the
            cubature is an exact integrator of $d\text{-dimensional}$ polynomials of
            degree $m$.
    """
    dimension: int
    region: AbstractIntegrationRegion
    degree: int = 0
    sparse: bool = False
    valid_dimensions: Tuple[int, int] | Sequence[Int] = (0, np.inf)

    def __init__(self, dimension: int, *args, **kwargs):
        r"""Instantiate $d$ dimensional cubature formula.

        Args:
            dimension: dimension $d$ of the integration region $\Omega$.
            args: additional non-keyword arguments to pass to the integration region
                initializer.
            kwargs: additional keyword arguments to pass to the integration region
                initializer.
        """
        self.dimension = self._validate_dimension(dimension)
        self.region = self.region(self.dimension, *args, **kwargs)

    @classmethod
    def _validate_dimension(cls, dimension: int) -> int:
        """If cubature formula is valid for the given dimension.

        Args:
            dimension: dimension $d$ of the integration region $\Omega$, for which
                the cubature formula is valid.

        Returns:
            dimension

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
            return dimension
        in_range_err_msg = f"{min_dim} <= dimension <= {max_dim}"
        in_list_err_msg = f"dimension in {valid_dims}"
        err_msg = in_range_err_msg if len(valid_dims) < 2 else in_list_err_msg
        raise ValueError(
            "Cubature formula cannot exactly integrate a vector-field of",
            f"the give dimension; expected {err_msg}, got {dimension}",
        )

    @abc.abstractproperty
    def coefficients(self) -> Float[Array, " k"]:
        r"""Formula coefficients $B$."""
        ...

    @abc.abstractproperty
    def vector_count(self) -> int:
        r"""Formula vector count $k$."""
        ...

    @abc.abstractproperty
    def vectors(self) -> Float[Array, "k d"]:
        r"""Formula vectors $v$."""
        ...

    def __call__(
        self,
        integrand: Callable[[Float[ArrayLike, " d"]], Float[Array, "..."]],
        normalize=True,
    ):
        r"""Approximately integrate some function $f$ over the weighted region $\Omega$.

        Computes the cubature formula $Q[f] = 1/z_c\sum_{i=1}^{k} B_i f(v_i)$.

        Args:
            integrand: the jax transformable function to integrate.
            normalize: if True, normalizes the volume of the integration region to one.
                Useful when the region's weight is an unnormalized probability density,
                and one wishes to transform it to a 'proper' normalized density (E.G
                :class:`GaussianIntegrationRegion`).

        Returns:
            Approximated integral and weighted evaluations of $f$ at each vector $v_i$.
        """
        coefficients_, vectors_ = transform_cubature_formula(
            self.coefficients, self.vectors, self.region.affine_transformation_matrix
        )
        return evaluate_cubature_formula(coefficients_, vectors_, integrand, normalize)


class AbstractGaussianCubatureFormula(AbstractCubatureFormula):
    """Cubature formulae for a :class:`GaussianIntegrationRegion`."""

    region = GaussianIntegrationRegion


class Hadamard(AbstractGaussianCubatureFormula):
    degree: int = 4

    @property
    def coefficients(self) -> Float[Array, " k"]:
        return self.region.volume / self.vector_count * np.ones(self.vector_count)

    @property
    def vector_count(self) -> int:
        max_power = np.ceil(np.log2(self.dimension))
        max_dim = np.power(2, max_power)
        return int(max_dim * 2)

    @property
    def vectors(self) -> Int[Array, "k d"]:
        hadamard_matrix = scipy_hadamard(self.vector_count // 2)

        new_matrix = hadamard_matrix[:, : self.dimension]
        return np.vstack((new_matrix, -new_matrix)) / np.sqrt(self.dimension)


class StroudSecrest63_31(AbstractGaussianCubatureFormula):
    r"""Degree 3 cubature formula from :cite:p:`stroudSecrest1963`, listing $E^{r^2}_n$
    3-1. (pg315) in :cite:p:`stroud1971`."""

    degree: int = 3
    sparse: bool = True

    @property
    def coefficients(self) -> Float[Array, " k"]:
        return self.region.volume / self.vector_count * np.ones(self.vector_count)

    @property
    def vector_count(self) -> int:
        return int(2 * self.dimension)

    @property
    def vectors(self) -> Float[Array, "k d"]:
        radius = np.sqrt(self.dimension / 2)
        points_symmetric = radius * np.diag(np.ones(self.dimension))
        points_fully_symmetric = np.vstack([points_symmetric, -points_symmetric])
        return points_fully_symmetric


def evaluate_cubature_formula(
    coefficients: Float[ArrayLike, " k"],
    vectors: Float[ArrayLike, "k d"],
    integrand: Callable[[Float[ArrayLike, " d"]], Float[Array, "..."]] = lambda x: 1.0,
    normalize: bool = True,
) -> Tuple[float, Float[Array, "k d"]]:
    """Evaluate a cubature formula for a given integrand $f$.

    Args:
        coefficients: cubature formula coefficients $B_i$.
        vectors: cubature formula vectors $v_i$.
        integrand: function to integrate $f$ defaults to $f(x) = 1.0$.
        normalize: if True, normalizes the volume of the integration region to one.
            Useful when the region's weight is an unnormalized probability density,
            and one wishes to transform it to a 'proper' normalized density (E.G
            :class:`GaussianIntegrationRegion`).

    Returns:
        Computed integral and the weighted evaluation points $B_i f(v_i)$.
    """
    eval_vmap = jax.vmap(lambda b, v: b * integrand(v), [0, 0])
    _coefficients = coefficients / jnp.where(normalize, jnp.sum(coefficients), 1.0)
    eval_points = eval_vmap(_coefficients, vectors)
    return sum(eval_points), eval_points


def transform_cubature_formula(
    coefficients: Float[ArrayLike, "k d"],
    vectors: Float[ArrayLike, "k d"],
    affine_transformation_matrix: None | Float[ArrayLike, "d+1 d+1"] = None,
) -> Tuple[Float[ArrayLike, "k d"], Float[ArrayLike, "k d"]]:
    """Affine transformation of cubature formula coefficients and vectors.

    Args:
        coefficients: cubature formula coefficients $B_i$.
        vectors: cubature formula vectors $v_i$.
        affine_transformation_matrix: a matrix specifying an affine transformation of
            the integration region.

    Returns:
        Affine transformed coefficients and vectors.
    """

    if affine_transformation_matrix is None:
        return coefficients, vectors

    coefficients_ = np.linalg.det(affine_transformation_matrix) * coefficients
    vectors_ = np.matmul(
        np.hstack([np.ones((vectors.shape[0], 1)), vectors]),
        affine_transformation_matrix,
    )
    return coefficients_, vectors_[:, 1:]


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
