"""Helpful utilities used throughout MCCube."""
from typing import Callable, Optional

import jax
import jax.tree_util as jtu
from jaxtyping import PyTree


def no_operation(*args, **kwds):
    ...


def identity_transform(t: float, particles: PyTree, args: PyTree) -> PyTree:
    return particles


def identity_validator(transform: Callable) -> Callable:
    return transform


# Below code is taken from Diffrax https://github.com/patrick-kidger/diffrax/blob/f101e75976e9dea86eb57f028f1f3bed646af1db/diffrax/misc.py#L155
def split_by_tree(key, tree, is_leaf: Optional[Callable[[PyTree], bool]] = None):
    """From Diffrax (https://github.com/patrick-kidger/diffrax/). Like jax.random.split
    but accepts tree as a second argument and produces a tree of keys with the same
    structure.
    """
    treedef = jtu.tree_structure(tree, is_leaf=is_leaf)
    return jtu.tree_unflatten(treedef, jax.random.split(key, treedef.num_leaves))
