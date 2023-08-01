"""Base classes for defining and using Propagator components."""
from __future__ import annotations

import functools
from typing import Callable, Sequence, overload

import chex
import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, Float, PyTree

from mccube.components.base import AbstractComponent, P, T


class AbstractPropagator(AbstractComponent):
    r"""Abstract base class for all Propagators.

    Extends the concept of Components to transforms that depend on the evaluation
    of a function $g(t, p(t), args)$. Also implements a concrete validate method to
    enforce the following properties of the transformation:

    1. The transformation **must return** the propagated particles $p^{\prime}(t)$ as a **rank-two tensor** (matrix).
    2. The transformation **must not change the particle dimension** $d$.
    3. The transformation **must not reduce the particle count** ($m \ge n$).
    """  # noqa: E501

    @overload
    def transform(
        self,
        logdensity: Callable[[float, PyTree, PyTree], PyTree],
        time: float,
        particles: PyTree[Float[ArrayLike, "n d"]],
        args: PyTree,
    ) -> PyTree[Float[Array, "m d"]]:
        r"""Transform the particles.

        Represents a function $f(g, t, p(t), args)$.

        Args:
            logdensity: particle interaction log-density function $g(t, p(t), args)$.
            time: particle existance time.
            particles: particles to transform.
            args: additional static arguments passed to the transform.

        Returns:
            Transformed/propagated particles.
        """
        ...

    @staticmethod
    def validate(transform: Callable[P, T]) -> Callable[P, T]:
        r"""Validate the transform.

        Ensures that the transform obeys the properties:

        1. The transformation **must return** the propagated particles $p^{\prime}(t)$
        as a **rank-two tensor** (matrix).
        2. The transformation **must not change the particle dimension** $d$.
        3. The transformation **must not reduce the particle count** ($m \ge n$).

        Args:
            transform: transform to validate, $f(g, t, p(t), args)$.

        Returns:
            Valid transform $(v \circ f)(g, t, p(t), args)$.
        """

        @functools.wraps(transform)
        def valid_transform(
            logdensity: Callable[[float, PyTree, PyTree], PyTree],
            time: float,
            particles: PyTree[Float[ArrayLike, "n d"]],
            args: PyTree[Float[Array, "m d"]],
        ) -> PyTree:
            propagated_particles = transform(logdensity, time, particles, args)
            chex.assert_rank(
                propagated_particles,
                2,
                custom_message=(
                    "Propagator transform should return particle array of rank two."
                ),
            )
            chex.assert_equal_shape_suffix(
                [propagated_particles, particles],
                1,
                custom_message=(
                    "Propagator transform must not alter particle dimensionality."
                ),
            )
            chex.assert_axis_dimension_gteq(
                propagated_particles,
                0,
                particles.shape[0],
                custom_message=(
                    "Propagator transform must not reduce the number of particles."
                ),
            )
            return propagated_particles

        return valid_transform

    def __call__(
        self,
        logdensity: Callable[[float, PyTree, PyTree], PyTree],
        time: float,
        particles: PyTree[Float[ArrayLike, "n d"]],
        args: PyTree,
    ) -> PyTree[Float[Array, "m d"]]:
        r"""Evaluate valid transform, $(v \circ f)(g, t, p(t), args)$.

        Args:
            logdensity: particle interaction log-density function $g(t, p(t), args)$.
            time: particle existance time.
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
            return _validated_transform(logdensity, time, particles, args)

        return jtu.tree_map(validated_transform, particles, is_leaf=eqx.is_array_like)


class WrappedPropagator(AbstractPropagator):
    """Wraps a sequence of Callables as a Propagator.

    Callables are composed in the sequence [a, b, c] -> c(b(a(...))), and are all
    assumed to have the same call signature as an :class:`AbstractPropagator`. Useful in
    cases where one has a suitable Callable, or composable sequence of Callables, which
    they wish to be validated and distinguished as an instance of
    :class:`AbstractPropagator`.

    Attributes:
        propagators: callables representing the Propagator transforms.
    """

    # fmt: off
    propagators: Sequence[Callable[[float, PyTree, PyTree], PyTree]] = lambda ld, t, p, a: p  # noqa: E731, E501
    # fmt: on
    def transform(
        self,
        logdensity: Callable[[float, PyTree, PyTree], PyTree],
        time: float,
        particles: PyTree[Float[ArrayLike, "n d"]],
        args: PyTree,
    ) -> PyTree[Float[ArrayLike, "m d"]]:
        return _compose_propagators(self.propagators)(logdensity, time, particles, args)


def _compose_propagators(*transforms):
    def compose(f, g):
        return lambda ld, t, p, args: g(ld, t, f(t, p, args), args)

    if isinstance(transforms, Sequence) and len(transforms) > 1:
        return functools.reduce(compose, transforms)
    return transforms[0]
