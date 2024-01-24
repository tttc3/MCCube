"""Defines custom terms for performing MCC in diffrax.

See [`diffrax.AbstractTerm`][] for further information on the terms API.
"""
from diffrax import AbstractTerm, ODETerm, WeaklyDiagonalControlTerm
from equinox.internal import ω
import jax.tree_util as jtu
import equinox as eqx
from jaxtyping import ArrayLike, PyTree

from ._custom_types import (
    Args,
    Particles,
    PartitionedParticles,
    RealScalarLike,
)


def _tree_flatten(tree: PyTree[ArrayLike]):
    return jtu.tree_map(
        lambda x: x.reshape(-1, x.shape[-1]), tree, is_leaf=eqx.is_array
    )


class MCCTerm(AbstractTerm):
    """Provides a convenience interface for CDEs whose control is a Cubature Path.

    Performs shape coercion to ensure the `ode` and `cde` terms can be broadcast
    together. One can achive the same result with a [`diffrax.MultiTerm`][] providing
    suitable modifcations are made to the input and vector field shapes.

    Example:
        ```python
        cubature = mccube.Hadamard(mccube.GaussianRegion(10))
        ode = ODETerm(lambda t, y, args: -y)
        cde = WeaklyDiagonalControlTerm(
            lambda t, y, args: 2.0,
            mccube.LocalLinearCubaturePath(cubature)
        )
        term = MCCTerm(ode, cde)
        sol = diffrax.diffeqsolve(term, ...)
        ```
    """

    ode: ODETerm
    cde: WeaklyDiagonalControlTerm

    def __init__(self, ode: ODETerm, cde: WeaklyDiagonalControlTerm):
        """
        Args:
            ode: is a [`diffrax.ODETerm`][].
            cde: is a [`diffrax.WeaklyDiagonalControlTerm`][], driven by a
                [`mccube.AbstractCubaturePath`][].
        """
        self.ode = ode
        self.cde = cde

    def vf(
        self, t: RealScalarLike, y: Particles | PartitionedParticles, args: Args
    ) -> tuple[PyTree[ArrayLike], ...]:
        ode_vf = self.ode.vf(t, _tree_flatten(y), args)
        cde_vf = self.cde.vf(t, _tree_flatten(y), args)
        return ode_vf, cde_vf

    def contr(
        self, t0: RealScalarLike, t1: RealScalarLike
    ) -> tuple[PyTree[ArrayLike], ...]:
        return self.ode.contr(t0, t1), self.cde.contr(t0, t1)

    def prod(
        self,
        vf: tuple[PyTree[ArrayLike], ...],
        control: tuple[PyTree[ArrayLike], ...],
    ) -> PyTree[ArrayLike, "Particles"]:
        ode_prod = self.ode.prod(vf[0], control[0])
        cde_prod = self.cde.prod(vf[1], control[1])
        return (ω(ode_prod)[:, None, ...] + ω(cde_prod)[None, ...]).ω  # type: ignore
