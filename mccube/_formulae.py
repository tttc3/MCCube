"""Defines cubature formulae for integrating functions over the regions (measure spaces) 
defined as [`AbstractRegions`][mccube.AbstractRegion].
"""
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
from ._regions import AbstractRegion, GaussianRegion
from ._utils import all_subclasses


class AbstractCubature[_Region: AbstractRegion](eqx.Module):
    r"""Abstract base class for cubature formulae.

    A concrete implementation of this class allows one to construct the [`points`][mccube.AbstractCubature.points]
    and [`weights`][mccube.AbstractCubature.weights] of a pre-defined cubature formula
    over some specified integration `region` (measure space).

    Example:
        ```python

        class Formula(AbstractCubature):
            degree = 3
            sparse = False

            @cached_property
            def weights(self):
                ...

            @cached_property
            def points(self):
                ...

            @cached_property
            def point_count(self):
                ...

        region = mccube.GaussianRegion(5)
        formula = Formula(region)
        # Formula(region=GaussianRegion(dimension=5))
        ```

    Attributes:
        region: an [`AbstractRegion`][mccube.AbstractRegion] (measure space) of a
            specified dimension.
        degree: the degree $m$ of polynomials, defined over the `region`, for which the
            cubature formulae is an exact integrator (with respect to the measure $\mu$).
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
        r"""Cubature point count $k$."""
        ...

    @cached_property
    def stacked_weights(self) -> CubatureWeights:
        """[`weights`][mccube.AbstractCubature.weights] stacked into a single vector."""
        weight_array = jtu.tree_map(lambda x: np.ones(x.shape[0]), self.points)
        return np.hstack((ω(self.weights) * ω(weight_array)).ω)  # pyright: ignore

    @cached_property
    def stacked_points(self) -> CubaturePoints:
        """[`points`][mccube.AbstractCubature.points] stacked into a single matrix."""
        return np.vstack(self.points)

    def __call__(
        self,
        integrand: Callable[[CubaturePoints], RealScalarLike],
    ) -> tuple[RealScalarLike, CubatureWeights]:
        r"""Approximately integrate some function $f$ over the integration region.

        Computes the cubature formula $Q[f] = \sum_{j=1}^{k} \lambda_j f(x_j)$.

        Args:
            integrand: the jax transformable function to integrate.

        Returns:
            Approximated integral and stacked weighted evaluations of $f$ at each
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
    [`GaussianRegion`][mccube.GaussianRegion].

    The Gaussian region is assumed to have the "probabilist's" Hermite measure.

    !!! warning

        Here it is assumed that the Gaussian measure is the normalized "probabilist's"
        Hermite weight (to be consistent with [`@victoir2004`]). However, other authors,
        such as [`@stroud1971`], assume the measure is the "physicist's" Hermite
        measure. Cubature formulae defined with respect to the later measure must be
        appropriately rescaled to be compatible with the measure assumed here.

    ??? cite "Reference: [`@victoir2004`]"

        ```bibtex
        @article{victoir2004,
          title   = {Asymmetric Cubature Formulae with Few Points in High Dimension for
                     Symmetric Measures},
          author  = {Victoir, Nicolas},
          year    = {2004},
          journal = {SIAM Journal on Numerical Analysis},
          volume  = {42},
          number  = {1},
          pages   = {209-227},
          doi     = {10.1137/S0036142902407952},
        }
        ```

    ??? cite "Reference: [`@stroud1971`]"

        ```bibtex
        @book{stroud1971,
        title     = {Approximate Calculation of Multiple Integrals},
        author    = {Stroud, A. H.},
        year      = {1971},
        publisher = {Prentice-Hall},
        pages     = {431},
        isbn      = {9780130438935},
        url       = {https://archive.org/details/approximatecalcu0000stro_b8j7}
        }
        ```
    """


class Hadamard(AbstractGaussianCubature):
    """Degree 3 Gaussian cubature from [`@victoir2004`].

    ??? cite "Reference: [`@victoir2004`]"

        ```bibtex
        @article{victoir2004,
          title   = {Asymmetric Cubature Formulae with Few Points in High Dimension for
                     Symmetric Measures},
          author  = {Victoir, Nicolas},
          year    = {2004},
          journal = {SIAM Journal on Numerical Analysis},
          volume  = {42},
          number  = {1},
          pages   = {209-227},
          doi     = {10.1137/S0036142902407952},
        }
        ```
    """

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
        r"""Cubature point count $k = 2^{\lceil{\log_2 d}\rceil + 1}$."""
        max_power = np.ceil(np.log2(self.region.dimension)) + 1
        return int(2**max_power)


class StroudSecrest63_31(AbstractGaussianCubature):
    r"""Degree 3 Gaussian cubature from [`@stroudSecrest1963`], listing $E_n^{r^2}$ 3-1
    (pg315) in [`@stroud1971`].


    ??? cite "Reference: [`@stroudSecrest1963`]"

        ```bibtex
        @article{stroudSecrest1963,
          title     = {Approximate Integration Formulas for Certain Spherically
                       Symmetric Regions},
          author    = {Stroud, A. H. and Secrest, Don},
          year      = {1963},
          journal   = {Mathematics of Computation},
          number    = {82},
          pages     = {105--135},
          publisher = {American Mathematical Society},
          volume    = {17},
          issn      = {00255718, 10886842},
          url       = {http://www.jstor.org/stable/2003633}
        }
        ```

    ??? cite "Reference: [`@stroud1971`]"

        ```bibtex
        @book{stroud1971,
          title     = {Approximate Calculation of Multiple Integrals},
          author    = {Stroud, A. H.},
          year      = {1971},
          publisher = {Prentice-Hall},
          pages     = {431},
          isbn      = {9780130438935},
          url       = {https://archive.org/details/approximatecalcu0000stro_b8j7}
        }
        ```
    """

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
        """Cubature point count $k = 2d$."""
        return int(2 * self.region.dimension)


class StroudSecrest63_32(AbstractGaussianCubature):
    r"""Degree 3 Gaussian cubature from [`@stroudSecrest1963`], listing $E_n^{r^2}$ 3-2
    (pg316) in [`@stroud1971`].

    This formula is identical to the [`mccube.Hadamard`][] formula for dimensions less
    than four. For all other dimensions, [`mccube.Hadamard`][] is strictly more
    efficient.

    ??? cite "Reference: [`@stroudSecrest1963`]"

        ```bibtex
        @article{stroudSecrest1963,
          title     = {Approximate Integration Formulas for Certain Spherically
                       Symmetric Regions},
          author    = {Stroud, A. H. and Secrest, Don},
          year      = {1963},
          journal   = {Mathematics of Computation},
          number    = {82},
          pages     = {105--135},
          publisher = {American Mathematical Society},
          volume    = {17},
          issn      = {00255718, 10886842},
          url       = {http://www.jstor.org/stable/2003633}
        }
        ```

    ??? cite "Reference: [`@stroud1971`]"

        ```bibtex
        @book{stroud1971,
          title     = {Approximate Calculation of Multiple Integrals},
          author    = {Stroud, A. H.},
          year      = {1971},
          publisher = {Prentice-Hall},
          pages     = {431},
          isbn      = {9780130438935},
          url       = {https://archive.org/details/approximatecalcu0000stro_b8j7}
        }
        ```
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
        """Cubature point count $k = 2^d$."""
        return int(2**self.region.dimension)


class StroudSecrest63_52(AbstractGaussianCubature):
    r"""Degree 5 Gaussian cubature from [`@stroudSecrest1963`], listing $E_n^{r^2}$ 5-2
    (pg317) in [`@stroud1971`].

    ??? cite "Reference: [`@stroudSecrest1963`]"

        ```bibtex
        @article{stroudSecrest1963,
          title     = {Approximate Integration Formulas for Certain Spherically
                       Symmetric Regions},
          author    = {Stroud, A. H. and Secrest, Don},
          year      = {1963},
          journal   = {Mathematics of Computation},
          number    = {82},
          pages     = {105--135},
          publisher = {American Mathematical Society},
          volume    = {17},
          issn      = {00255718, 10886842},
          url       = {http://www.jstor.org/stable/2003633}
        }
        ```

    ??? cite "Reference: [`@stroud1971`]"

        ```bibtex
        @book{stroud1971,
          title     = {Approximate Calculation of Multiple Integrals},
          author    = {Stroud, A. H.},
          year      = {1971},
          publisher = {Prentice-Hall},
          pages     = {431},
          isbn      = {9780130438935},
          url       = {https://archive.org/details/approximatecalcu0000stro_b8j7}
        }
        ```
    """

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
        """Cubature point count $k = 2d^2 + 1$."""
        return int(2 * self.region.dimension**2 + 1)


class StroudSecrest63_53(AbstractGaussianCubature):
    r"""Degree 5 Gaussian cubature from [`@stroudSecrest1963`], listing $E_n^{r^2}$ 5-3
    (pg317) in [`@stroud1971`]. Valid for regions with dimension d > 2.


    ??? cite "Reference: [`@stroudSecrest1963`]"

        ```bibtex
        @article{stroudSecrest1963,
          title     = {Approximate Integration Formulas for Certain Spherically
                       Symmetric Regions},
          author    = {Stroud, A. H. and Secrest, Don},
          year      = {1963},
          journal   = {Mathematics of Computation},
          number    = {82},
          pages     = {105--135},
          publisher = {American Mathematical Society},
          volume    = {17},
          issn      = {00255718, 10886842},
          url       = {http://www.jstor.org/stable/2003633}
        }
        ```

    ??? cite "Reference: [`@stroud1971`]"

        ```bibtex
        @book{stroud1971,
          title     = {Approximate Calculation of Multiple Integrals},
          author    = {Stroud, A. H.},
          year      = {1971},
          publisher = {Prentice-Hall},
          pages     = {431},
          isbn      = {9780130438935},
          url       = {https://archive.org/details/approximatecalcu0000stro_b8j7}
        }
        ```
    """

    degree: int = 5

    def __check_init__(self):
        d = self.region.dimension
        if d <= 2:
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
        """Cubature point count $k = 2^d + 2d$."""
        return 2 * self.region.dimension + 2**self.region.dimension


_modes = {"R", "S", "FS"}


def _generate_point_permutations(
    point: Shaped[ArrayLike, "... d"], mode: Literal["R", "S", "FS"] = "FS"
) -> CubaturePoints:
    r"""Generate a matrix of permutations of a given $d$-dimensional point/vector.

    ??? cite "Reference: [`@stroud1971`]"

        ```bibtex
        @book{stroud1971,
          title     = {Approximate Calculation of Multiple Integrals},
          author    = {Stroud, A. H.},
          year      = {1971},
          publisher = {Prentice-Hall},
          pages     = {431},
          isbn      = {9780130438935},
          url       = {https://archive.org/details/approximatecalcu0000stro_b8j7}
        }
        ```

    ??? cite "Reference: [`@cools1997`]"

        ```bibtex
        @article{cools1997,
          title     = {Constructing cubature formulae: the science behind the art},
          author    = {Cools, Ronald},
          year      = {1997},
          journal   = {Acta Numerica},
          publisher = {Cambridge University Press},
          pages     = {1-54},
          volume    = {6},
          doi       = {10.1017/S0962492900002701}
        }
        ```

    Args:
        point: a $d$-dimensional vector from which to generate permutations.
        mode: the type of permutations to generate, based on the notation in
            [`@stroud1971`] (see [`@cools1997`] for group theoretic definitions):
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


# Must be at the bottom such that the above subclasses are included.
builtin_cubature_registry: set[type[AbstractCubature]] = all_subclasses(
    AbstractCubature
)
"""A searchable registry of all cubature formulae, in the current scope, that are 
subclasses of [`mccube.AbstractCubature`][]."""


def search_cubature_registry(
    region: AbstractRegion,
    degree: int | None = None,
    sparse_only: bool = False,
    minimal_only: bool = False,
    searchable_formulae: Collection[type[AbstractCubature]] = builtin_cubature_registry,
) -> list[AbstractCubature]:
    """Returns a list of cubature formulae of a specified degree for a given region.

    Example:
        ```python
        result = search_cubature_registry(mccube.GaussianRegion(2), degree=3)
        # [StroudSecrest63_31(...), StroudSecrest63_32(...), Hadamard(...)]

        result = search_cubature_registry(mccube.GaussianRegion(10), minimal_only=True)
        # [StroudSecrest63_31(...)]
        ```

    Args:
        region: the region for which the cubature formulae must be of degree $m$.
        degree: the required degree $m$ of cubature formulae. Ignored if none.
        sparse_only: if to only select cubature formulae with sparse cubature points.
        minimal_only: if to only return formulae with minimal point count.
        searchable_formulae: collection from which to search for suitable cubature
            formulae.
    """
    selected_formulae = list()
    minimum_point_count = float("inf")
    for f in searchable_formulae:
        # Try to instantiate the formula in order to establish a point count.
        try:
            formula = f(region)
        except ValueError:
            continue
        degree_condition = (f.degree != degree) and (degree is not None)
        sparse_condition = sparse_only and not f.sparse
        if degree_condition or sparse_condition:
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
