"""Defines custom types that are used throughout the package. The following roughly 
describes the semantics behind each axis name used in the custom types.

-   **k**: the unique particle axis.
-   **n**: the particles variant axis. I.E. the unique particle in `k` updated with 
    respect to `n` cubature points.
-   **d**: the dimensionality of the particles.
"""

from __future__ import annotations

import numpy as np
from typing import Any, TYPE_CHECKING
from jaxtyping import Array, ArrayLike, PyTree, Shaped, Int, Float, Bool

# These are identical to a future version of diffrax.
if TYPE_CHECKING:
    BoolScalarLike = bool | Array | np.ndarray
    FloatScalarLike = float | Array | np.ndarray
    IntScalarLike = int | Array | np.ndarray
else:
    FloatScalarLike = Float[ArrayLike, ""]
    IntScalarLike = Int[ArrayLike, ""]
    BoolScalarLike = Bool[ArrayLike, ""]

RealScalarLike = FloatScalarLike | IntScalarLike

P = PyTree[Shaped[ArrayLike, "?k #n d"], "P"]
"""Particle PyTree, alias of :code:`PyTree[Shaped[ArrayLike, "?k #n d"], "P"]`."""

XP = PyTree[Shaped[ArrayLike, "?k #n d"], "XP"]
"""Expanded particle PyTree, alias of :code:`PyTree[Shaped[ArrayLike, "?k #n d"], "XP"]`."""

Args = PyTree[Any]
"""Auxillary argument PyTree, alias of :code:`PyTree[Any]`."""

CubaturePoints = Shaped[ArrayLike, "n d"]
"""Cubature points, alias of :code:`Shaped[ArrayLike, "n d"]`."""

CubaturePointsTree = PyTree[Shaped[ArrayLike, "?n_i d"], "CPT"]
"""Tree of cubature points, alias of PyTree[Shaped[ArrayLike, "?n_i d"], "CPT"]"""

CubatureWeights = Shaped[ArrayLike, " n"]
"""Cubature weights, alias of :code:`Shaped[ArrayLike, " n"]`"""

CubatureWeightsTree = PyTree[RealScalarLike, "CPT"]
"""Tree of cubature weights, PyTree[RealScalarLike, "CPT"]"""
