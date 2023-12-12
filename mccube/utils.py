"""Helpful utilities used throughout MCCube."""
from typing import Callable, Optional, Type, TypeVar, Any

import dataclasses
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

T = TypeVar("T")


def no_operation(*args, **kwds):
    ...


def identity_transform(t: float, particles: PyTree, args: PyTree) -> PyTree:
    return particles


def identity_validator(transform: Type[T]) -> T:
    return transform


@dataclasses.dataclass(frozen=True)
class if_valid_array:
    axis: int

    def __call__(self, x: Any) -> Optional[int]:
        return self.axis if is_valid_array(self.axis)(x) else None


@dataclasses.dataclass(frozen=True)
class is_valid_array:
    axis: int

    def __call__(self, x: Any) -> bool:
        return eqx.is_array(x) and x.ndim - abs(self.axis) > 0


# Below code is taken from Diffrax https://github.com/patrick-kidger/diffrax/blob/f101e75976e9dea86eb57f028f1f3bed646af1db/diffrax/misc.py#L155
def split_by_tree(key, tree, is_leaf: Optional[Callable[[PyTree], bool]] = None):
    """From Diffrax (https://github.com/patrick-kidger/diffrax/). Like jax.random.split
    but accepts tree as a second argument and produces a tree of keys with the same
    structure.
    """
    treedef = jtu.tree_structure(tree, is_leaf=is_leaf)
    return jtu.tree_unflatten(treedef, jax.random.split(key, treedef.num_leaves))


_itemsize_kind_type = {
    (1, "i"): jnp.int8,
    (2, "i"): jnp.int16,
    (4, "i"): jnp.int32,
    (8, "i"): jnp.int64,
    (2, "f"): jnp.float16,
    (4, "f"): jnp.float32,
    (8, "f"): jnp.float64,
}


def force_bitcast_convert_type(val, new_type):
    val = jnp.asarray(val)
    intermediate_type = _itemsize_kind_type[new_type.dtype.itemsize, val.dtype.kind]
    val = val.astype(intermediate_type)
    return jax.lax.bitcast_convert_type(val, new_type)
