"""Recombinators that depend on random subsampling."""
from __future__ import annotations

from typing import Callable

import jax
import jax.tree_util as jtu
from jaxtyping import PyTree, PRNGKeyArray

from mccube.components.recombinators.base import AbstractRecombinator
from mccube.utils import no_operation, split_by_tree, force_bitcast_convert_type


class MonteCarloRecombinator(AbstractRecombinator):
    """Monte-carlo particle sub-sampling/recombination.

    Attributes:
        key: random key to initialise monte-carlo selection.
        weighting_function: selection weight/probability for each particle $x \in p(t)$.
        with_replacement: if to sample particles $x \in p(t)$ with/without replacement.
            Without replacement is usually significantly slower.
    """

    key: PRNGKeyArray
    weighting_function: Callable = no_operation
    with_replacement: bool = True

    def transform(
        self,
        key: PRNGKeyArray,
        recombination_factor: int | float,
        time: float,
        particles: PyTree,
        args: PyTree,
    ) -> PyTree:
        recombined_point_count = (
            particles.shape[0] * particles.shape[1] // recombination_factor
        )
        reshaped_particles = particles.reshape(-1, *particles.shape[2:])
        weights = self.weighting_function(reshaped_particles)
        selected = jax.random.choice(
            key,
            reshaped_particles,
            (recombined_point_count,),
            self.with_replacement,
            weights,
        )
        return selected.reshape(
            recombined_point_count // particles.shape[1], *particles.shape[1:]
        )

    def __call__(
        self,
        recombination_factor: Callable[[float, PyTree, PyTree], PyTree],
        time: float,
        particles: PyTree,
        args: PyTree,
    ) -> PyTree:
        time_ = force_bitcast_convert_type(time, jax.numpy.int32)
        key = jax.random.fold_in(self.key, time_)
        keys = split_by_tree(key, particles)
        return jtu.tree_map(
            lambda key, particles: self.transform(
                key, recombination_factor, time, particles, args
            ),
            keys,
            particles,
        )
