"""Helpful utilities used throughout MCCube."""

import dataclasses
from typing import Any

import equinox as eqx


def nop(*args, **kwargs):
    """Callable which accepts any arguments and does nothing."""
    ...


def all_subclasses(cls: type) -> set[type]:
    """Recursively identifies all subclasses of a class.

    Args:
        cls: the class whose subclasses are to be enumerated.

    Returns:
        All subclasses of `cls` in the current scope.
    """
    subclasses = set()

    for subclass in cls.__subclasses__():
        subclasses.add(subclass)
        subclasses.update(all_subclasses(subclass))
    return subclasses


@dataclasses.dataclass(frozen=True)
class if_valid_array:
    """Similar to :func:`eqx.if_array` but returns a callable that additionally checks
    if the array is of the required dimension to support indexing along :attr:`axis`.

    Attributes:
        axis: the axis one wishes to index the array along.
    """

    axis: int

    def __call__(self, x: Any) -> int | None:
        return self.axis if is_valid_array(self.axis)(x) else None


@dataclasses.dataclass(frozen=True)
class is_valid_array:
    """Similar to :func:`eqx.is_array` but returns a callable that additionally checks
    that the array is of the required dimension to support indexing along :attr:`axis`.

    Attributes:
        axis: the axis one wishes to index the array along.
    """

    axis: int

    def __call__(self, x: Any) -> bool:
        return eqx.is_array(x) and x.ndim - abs(self.axis) > 0
