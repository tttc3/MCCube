"""Base classes for defining and using Recombinator components."""
from __future__ import annotations

import functools
from typing import Callable, Sequence, overload

import chex
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, Float, PyTree

from mccube.components.base import AbstractComponent, P, T


class AbstractRecombinator(AbstractComponent):
    r"""Abstract base class for all Recombinators.

    Extends the concept of Components to transforms that depend on a recombination
    factor $r_f$. Also implements a concrete validate method to enforce the following
    properties of the transformation:

    1. The transformation **must return** the recombinant particles $p^{\prime}(t)$ as a **rank-two tensor** (matrix).
    2. The transformation **must not change the particle dimension** $d$.
    3. The transformation **must not increase the particle count** ($m \ge n$).
    """  # noqa: E501

    @overload
    def transform(
        self,
        recombination_factor: int | float,
        time: float,
        particles: PyTree[Float[ArrayLike, "n d"]],
        args: PyTree,
    ) -> PyTree[Float[Array, "m d"]]:
        r"""Transform the particles.

        Represents a function $f(r_f, t, p(t), args)$.

        Args:
            recombination_factor: factor by which to reduce the number of particles.
            time: particle existence time.
            particles: particles to transform.
            args: additional static arguments passed to the transform.

        Returns:
            Transformed/recombinant particles.
        """
        ...

    @staticmethod
    def validate(transform: Callable[P, T]) -> Callable[P, T]:
        r"""Validate the transform.

        Ensures that the transform obeys the properties:

        1. The transformation **must return** the recombinant particles $p^{\prime}(t)$ as a **rank-two tensor** (matrix).
        2. The transformation **must not change the particle dimension** $d$.
        3. The transformation **must not increase the particle count** ($m \ge n$).

        Args:
            transform: transform to validate, $f(r_f, t, p(t), args)$.

        Returns:
            Valid transform $(v \circ f)(r_f, t, p(t), args)$.
        """  # noqa: E501

        @functools.wraps(transform)
        def valid_transform(
            recombination_factor: int,
            time: float,
            particles: PyTree[Float[ArrayLike, "n d"]],
            args: PyTree,
        ) -> PyTree[Float[ArrayLike, "m d"]]:
            recombinant_particles = transform(
                recombination_factor, time, particles, args
            )
            chex.assert_rank(
                recombinant_particles,
                2,
                custom_message=(
                    "Recombinator transform must produce a particle array of rank two."
                ),
            )
            chex.assert_tree_shape_suffix(
                recombinant_particles,
                (particles.shape[-1],),
                custom_message=(
                    "Recombinator transform must not alter particle dimensionality."
                ),
            ),
            chex.assert_axis_dimension_lteq(
                jnp.vstack(recombinant_particles),
                0,
                particles.shape[0],
                custom_message=(
                    "Recombinator transform must not increase the number of particles."
                ),
            )

            return recombinant_particles

        return valid_transform

    def __call__(
        self,
        recombination_factor: Callable[[float, PyTree, PyTree], PyTree],
        time: float,
        particles: PyTree[Float[ArrayLike, "n d"]],
        args: PyTree,
    ) -> PyTree[Float[ArrayLike, "m d"]]:
        r"""Evaluate valid transform, $(v \circ f)(r_f, t, p(t), args)$.

        Args:
            recombination_factor: factor by which to reduce the number of particles.
            time: particle existence time.
            particles: particles to transform with PyTree structure $P$.
            args: additional static arguments passed to the transform.

        Returns:
            A PyTree of transformed particles with structure $P$.

        Raises:
            AssertionError: if any of the transform properties are not valid/obeyed.
        """

        def validated_transform(
            particles: PyTree[Float[ArrayLike, "n d"]]
        ) -> PyTree[Float[Array, "m d"]]:
            """Allows for tree_map to be applied only over the particles."""
            _validated_transform = self.validate(self.transform)
            return _validated_transform(recombination_factor, time, particles, args)

        return jtu.tree_map(validated_transform, particles, is_leaf=eqx.is_array_like)


class WrappedRecombinator(AbstractRecombinator):
    """Wraps a sequence of Callables as a Recombinator.

    Callables are composed in the sequence [a, b, c] -> c(b(a(...))), and are all
    assumed to have the same call signature as an `AbstractRecombinator`.
    Useful in cases where one has a suitable Callable, or composable sequence of
    Callables, which they wish to be validated and distinguished as an instance of
    :class:`AbstractRecombinator`.

    Attributes:
        recombinators: callables representing the Recombinator transforms.
    """

    # fmt: off
    recombinators: Sequence[Callable[P,T]] = lambda c, t, p, a: p  # noqa: E731
    # fmt: on
    def transform(
        self,
        recombination_factor: int | float,
        time: float,
        particles: PyTree[Float[ArrayLike, "n d"]],
        args: PyTree,
    ) -> PyTree[Float[ArrayLike, "m d"]]:
        return _compose_recombinators(self.recombinators)(
            recombination_factor, time, particles, args
        )


def _compose_recombinators(*transforms):
    def compose(f, g):
        return lambda ld, t, p, args: g(ld, t, f(t, p, args), args)

    if isinstance(transforms, Sequence) and len(transforms) > 1:
        return functools.reduce(compose, transforms)
    return transforms[0]
