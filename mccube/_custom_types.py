"""Defines custom types that are used throughout the package. The following symbols 
are used in the definitions of the custom types:

-   **d**: the dimensionality of the particles.
-   **n**: the number of particles.
-   **n_hat**: the number of recombined particles.
-   **k**: the number of versions of a particles (resulting from the same number of 
cubature paths/points).
-   **m**: the number of partitions of the particles.
"""

from typing import Any, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from jaxtyping import Array, ArrayLike, Bool, Float, Int, PyTree, Shaped

# These are identical to the definitions in diffrax.
if TYPE_CHECKING:
    BoolScalarLike = bool | Array | npt.NDArray[np.bool_]
    FloatScalarLike = float | Array | npt.NDArray[np.float_]
    IntScalarLike = int | Array | npt.NDArray[np.int_]
else:
    BoolScalarLike = Bool[ArrayLike, ""]
    FloatScalarLike = Float[ArrayLike, ""]
    IntScalarLike = Int[ArrayLike, ""]
    """A value which can be considered as an integer scalar value."""

RealScalarLike = FloatScalarLike | IntScalarLike
"""A value which can be considered as a real scalar value."""

Particles = PyTree[Shaped[Array, "?n d"], "P"]
"""A PyTree where each leaf is an array of `n` particles of dimension `d`."""

PartitionedParticles = PyTree[Shaped[Array, "?m ?n_div_m d"], "P"]
"""A [`Particles`][mccube._custom_types.Particles] PyTree where each leaf has been 
reorganised into `m` equally sized partitions of `n/m` particles of dimension `d`."""

RecombinedParticles = PyTree[Shaped[Array, "?n_hat d"], "P"]
"""A [`Particles`][mccube._custom_types.Particles] PyTree where each leaf has been 
recombined/compressed into `n_hat < n` particles of dimension `d`."""

UnpackedParticles = PyTree[Shaped[Array, "?n d-1"], "P"]
"""A [`Particles`][mccube._custom_types.Particles] PyTree of `n` particles of dimension 
`d-1`, which have been unpacked from a PyTree of `n` particles of dimension `d`, where 
the `d`-th dimension represents the particle [`Weights`][mccube._custom_types.Weights]."""

PackedParticles = PyTree[Shaped[Array, "?n d+1"], "P"]
"""A [`Particles`][mccube._custom_types.Particles] PyTree of `n` particles of dimension
`d+1`, where the `d+1`-th dimension represents the particle [`Weights`][mccube._custom_types.Weights],
which have been packed from a PyTree of `n` particles of dimension `d`, and a PyTree of
`n` weights."""

Weights = PyTree[Shaped[Array, "?*n"] | None, "P"]
"""A PyTree where each leaf is an array of `n` [`Weights`][mccube._custom_types.Weights] or [`None`][]."""

Args = PyTree[Any]
"""A PyTree of auxillary arguments."""

CubaturePoints = Shaped[ArrayLike, "k d"]
"""An array of `k` points/vectors of dimension `d` which form the nodes of a cubature
formula."""

CubaturePointsTree = PyTree[Shaped[ArrayLike, "?k_i d"], "CPT"]
"""A PyTree of [`CubaturePoints`][mccube._custom_types.CubaturePoints], where each leaf
is associated with a different leaf in a [`CubatureWeightsTree`][mccube._custom_types.CubatureWeightsTree]."""

CubatureWeights = Shaped[ArrayLike, " k"]
"""An array of `k` weights which together with the appropriate [`CubaturePoints`][mccube._custom_types.CubaturePoints],
defines a cubature formula."""

CubatureWeightsTree = PyTree[RealScalarLike, "CPT"]
"""A PyTree of [`CubatureWeights`][mccube._custom_types.CubatureWeights], where each leaf
is associated with a different leaf in a [`CubaturePointsTree`][mccube._custom_types.CubaturePointsTree]."""


DenseInfo = dict[str, PyTree[Array]]


del Array, ArrayLike, Bool, Float, Int, PyTree, Shaped
