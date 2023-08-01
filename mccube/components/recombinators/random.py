"""Recombinators that depend on random subsampling."""
from __future__ import annotations

from typing import Callable

import jax
import jax.tree_util as jtu
from jax._src.random import KeyArray
from jaxtyping import PyTree

from mccube.components.recombinators.base import AbstractRecombinator
from mccube.utils import no_operation, split_by_tree


class MonteCarloRecombinator(AbstractRecombinator):
    """Monte-carlo particle sub-sampling/recombination.

    Attributes:
        key: random key to initialise monte-carlo selection.
        weighting_function: selection weight/probability for each particle $x \in p(t)$.
        with_replacement: if to sample particles $x \in p(t)$ with/without replacement.
            Without replacement is usually significantly slower.
    """

    key: KeyArray
    weighting_function: Callable = no_operation
    with_replacement: bool = True

    def transform(
        self,
        key: KeyArray,
        recombination_factor: int | float,
        time: float,
        particles: PyTree,
        args: PyTree,
    ) -> PyTree:
        recombined_point_count = particles.shape[0] // recombination_factor
        weights = self.weighting_function(particles)
        return jax.random.choice(
            key,
            particles,
            (recombined_point_count,),
            self.with_replacement,
            weights,
        )

    def __call__(
        self,
        recombination_factor: Callable[[float, PyTree, PyTree], PyTree],
        time: float,
        particles: PyTree,
        args: PyTree,
    ) -> PyTree:
        key = jax.random.fold_in(self.key, time)
        keys = split_by_tree(key, particles)
        return jtu.tree_map(
            lambda key, particles: self.transform(
                key, recombination_factor, time, particles, args
            ),
            keys,
            particles,
        )
