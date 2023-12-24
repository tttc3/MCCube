from __future__ import annotations

import abc
import functools
from collections.abc import Callable

import chex
import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree

from .._custom_types import Args, P, RealScalarLike, XP


class AbstractKernel(eqx.Module):
    r"""Abstract base class for all Kernels.

    Composes a transform $f\colon(t_0, t_1, p(t_0), \text{args}) \to p(t_1)$, with a validation
    rule $v \colon f \to h$, to yield $h \colon (t_0, t_1, p(t_0), \text{args}) \to p(t_1)$,
    where $h := (v \circ f)(t_0, t_1, p(t_0), \text{args})$ is the validated transform.

    The primary utility of this class is to ensure that the transforms defined in
    concrete implementations obey expected properties (as defined by the abstract class'
    implementation of the validate method).
    """

    @abc.abstractmethod
    def transform(
        self, t0: RealScalarLike, t1: RealScalarLike, particles: P, args: Args
    ) -> P:
        r"""Transform the particles.

        Represents a function $f \colon (t_0, t_1, p(t_0), \text{args}) \to p(t_1)$.

        Args:
            t0: current time; particle state observation time.
            t1: transformation time; transformed particle state observation time.
            particles: particles to transform, $p(t_0)$.
            args: additional static arguments passed to the transform.

        Returns:
            Transformed particles, $p(t_1)$.
        """
        ...

    def validate[T](self, transform: Callable[..., T]) -> Callable[..., T]:
        r"""Validate the transform.

        Ensures that the transform obeys certain properties. Note: the default
        implementation does nothing, simply returning the non-validated transform.

        Args:
            transform: transform to validate, $f(t_0, t_1, p(t_0), \text{args})$.

        Returns:
            Validated transform $(v \circ f)(t_0, t_1, p(t_0), \text{args})$.
        """
        return transform

    def __call__(
        self, t0: RealScalarLike, t1: RealScalarLike, particles: PyTree, args: Args
    ) -> PyTree:
        r"""Evaluate the validated transform.

        Args:
            t0: current time; particle state observation time.
            t1: future time; future particle state observation time.
            particles: particles to transform, $p(t_0)$.
            args: additional static arguments passed to the transform.

        Returns:
            A PyTree of transformed particles, $p(t_1)$, with structure $P$.

        Raises:
            AssertionError: if any of the transform properties are not valid/obeyed.
        """
        _validated_transform = self.validate(self.transform)
        return _validated_transform(t0, t1, particles, args)


class AbstractRecombinationKernel(AbstractKernel):
    r"""Abstract base class for all Recombination Kernels.

    An :class:`AbstractKernel` is a :class:`AbstractRecombinationKernel` if it has a
    :attr:`recombined_shape` attribute, and its :meth:`transform` obeys the following
    validation property:

        -

    Attributes:
        recombined_shape: the particle shape :meth:`transform` should yield.
    """

    recombined_shape: PyTree[tuple[int, ...], "XP"]

    def transform(
        self, t0: RealScalarLike, t1: RealScalarLike, particles: XP, args: Args
    ) -> P:
        ...

    def validate[T](self, transform: Callable[..., T]) -> Callable[..., T]:
        r"""Validate the transform.

        Ensures that the transform obeys the following property:


        Args:
            transform: transform to validate, $f(t_0, t_1, p(t_0), \text{args})$.

        Returns:
            Validated transform $(v \circ f)(t_0, t_1, p(t_0), \text{args})$.
        """

        @functools.wraps(self.transform)
        def valid_transform(
            t0: RealScalarLike,
            t1: RealScalarLike,
            particles: XP,
            args: Args,
        ) -> T:
            recombined_particles = self.transform(t0, t1, particles, args)

            def _check(_shape, _recombined):
                chex.assert_shape(_recombined, _shape)
                return _recombined

            return jtu.tree_map(
                _check,
                self.recombined_shape,
                recombined_particles,
                is_leaf=lambda x: isinstance(x, tuple),
            )

        return valid_transform


AbstractRecombinationKernel.__init__.__doc__ = """Args:
    recombined_count: the particle count that :method:`transform` should yield. 
"""
