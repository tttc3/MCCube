"""Base classes for creating and defining components."""

from __future__ import annotations

import abc
from typing import Callable, TypeVar
from typing_extensions import ParamSpec

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, Float, PyTree

from mccube.utils import identity_transform, identity_validator

P = ParamSpec("P")
T = TypeVar("T")


class AbstractComponent(eqx.Module):
    r"""Abstract base class for all Components.

    Given some particles $p(t)$, a Component defines the composition of a
    transformation $f(t, p(t), args)$, with a validation rule $v(f)$, resulting in
    the validated transform $h(t, p(t), args) = (v \circ f)(t, p(t), args)$.
    """

    @abc.abstractmethod
    def transform(
        self, time: float, particles: PyTree[Float[ArrayLike, "..."]], args: PyTree
    ) -> PyTree[Float[Array, "..."]]:
        r"""Transform the particles.

        Represents a function $f(t, p(t), args)$.

        Args:
            time: particle existance time.
            particles: particles to transform.
            args: additional static arguments passed to the transform.

        Returns:
            Transformed particles.
        """
        ...

    @abc.abstractmethod
    def validate(self, transform: Callable[P, T]) -> Callable[P, T]:
        r"""Validate the transform.

        Ensures that the transform obeys certain properties.

        Args:
            transform: transform to validate, $f(t, p(t), args)$.

        Returns:
            Validated transform $(v \circ f)(t, p(t), args)$.
        """
        ...

    def __call__(
        self, time: float, particles: PyTree[Float[ArrayLike, "..."]], args: PyTree
    ) -> PyTree[Float[Array, "..."]]:
        r"""Evaluate the validated transform.

        Args:
            time: particle existance time.
            particles: particles to transform with PyTree strucutre $P$.
            args: additional static arguments passed to the transform.

        Returns:
            A PyTree of transformed particles with strucutre $P$.

        Raises:
            AssertionError: if any of the transform properties are not valid/obeyed.
        """

        def validated_transform(
            particles: PyTree[Float[ArrayLike, "..."]]
        ) -> PyTree[Float[Array, "..."]]:
            """Allows for tree_map to be applied only over the particles."""
            _validated_transform = self.validate(self.transform)
            return _validated_transform(time, particles, args)

        return jtu.tree_map(validated_transform, particles, is_leaf=eqx.is_array_like)


class Component(AbstractComponent):
    """Defines a component without the need for subclassing `AbstractComponent`.

    Attributes:
        transformer: callable representing the component transform.
        validator: callable representing the component transform validator.
    """

    transformer: Callable[[float, PyTree, PyTree], PyTree] = identity_transform
    validator: Callable[[Callable[P, T]], Callable[P, T]] = identity_validator

    def transform(
        self, time: float, particles: PyTree[Float[ArrayLike, "..."]], args: PyTree
    ) -> PyTree[Float[Array, "..."]]:
        return self.transformer(time, particles, args)

    def validate(self, transform: Callable[P, T]) -> Callable[P, T]:
        return self.validator(transform)
