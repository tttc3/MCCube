from typing import Callable, Optional
import equinox as eqx
import jax.tree_util as jtu
import jax.numpy as jnp
from diffrax import AbstractWrappedSolver
from diffrax.custom_types import Bool, DenseInfo, PyTree, Scalar
from diffrax.solution import RESULTS
from diffrax.solver.base import _SolverState
from diffrax.term import AbstractTerm

import jax
from mccube.utils import if_valid_array, is_valid_array


class RecombinationSolver(AbstractWrappedSolver):
    """Provides compatibility with multi-path control terms.

    When using multi-path control terms, the previous system state is updated according
    to `n` paths, yielding $n$ updated system states. This violates a necessary
    condition for diffrax; the PyTree structure and shape of the system state `y` must
    be preserved over solution steps.

    This solver provides a means to handle such expanding state multi-path controls by
    wrapping some [`diffrax.AbstractSolver`] and composing its solution step with a
    `recombination_kernel`. The job of the recombination kernel is to recombine the $n$
    updated system states, obtained from the solution step of the wrapped solver, into
    a single (somehow maximally representative) state, such that the RecombinationSolver
    # satisfies the necessary condition on PyTree structure and shape preservation of
    the system state `y`.

    !!! example

        Markov chain cubature requires one to solve a multi-path CDE. Without the use
        of recombination, the system state grows with $n^m$, where $n$ is the number of
        paths, and $m$ the number of solution steps. This has two major problems:

            1. Diffrax's necessary condition on preserving the shape and structure of
                the system state `y` cannot be satisfied.
            2. Exponential scaling in the system state makes computation intractable
                for all but very small $m$.

        Recombination solves both these problems by preserving the system state, and
        allowing the $n$ system states to only exist ephemerally as an intermediate
        component of the solution step.

    """

    recombination_kernel: Callable[[PyTree, Scalar, PyTree, PyTree], PyTree]
    n_substeps: int = 2

    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def interpolation_cls(self):
        return self.solver.interpolation_cls

    @property
    def nonlinear_solver(self):
        return self.solver.nonlinear_solver

    # Strictly these order terms will be incorrect, due to the recombination.
    # The weak order is governed by the recombination rule, in conjunction with the solver.
    def order(self, terms: PyTree[AbstractTerm]) -> Optional[int]:
        return self.solver.order(terms)

    def strong_order(self, terms: PyTree[AbstractTerm]) -> Scalar | None:
        return self.solver.strong_order(terms)

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
    ) -> _SolverState:
        # Required for FSAL RK Methods
        return eqx.filter_vmap(lambda y: self.solver.init(terms, t0, t1, y, args))(y0)

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        args: PyTree,
        solver_state: _SolverState,
        made_jump: Bool,
    ) -> tuple[PyTree, Optional[PyTree], DenseInfo, _SolverState, RESULTS]:
        n_particles, n_paths, *_ = y0.shape
        expansion_ratio = n_paths ** (self.n_substeps - 1)
        n_expanded_particles = n_particles * expansion_ratio
        n_sub_expanded_particles = n_expanded_particles // n_paths

        dt0_substep = t1 - t0 / (self.n_substeps)

        # Input Shape:  [n_particles, n_paths, ...]
        # Output Shape: [n_particles, n_paths, ...]
        @jtu.Partial(eqx.filter_vmap, in_axes=if_valid_array(0))
        def _vmapped_solver_step(
            _t0: Scalar,
            _t1: Scalar,
            _y: PyTree,
            _solver_state: PyTree | None,
        ):
            """Wrapped solver vectorized over the leading axis of `_y` and conditioned
            on the sub-step invariant arguments (terms, args, made_jump)."""
            return self.solver.step(terms, _t0, _t1, _y, args, _solver_state, made_jump)

        # Input Shape:  [n_particles, n_paths, ...]
        # Output Shape: [n_expanded_particles, n_paths, ...]
        def _init_fn(
            _y: PyTree,
        ) -> tuple[PyTree, Optional[PyTree], DenseInfo, _SolverState, RESULTS]:
            """Tiles the input `_y` to provide shape preservation over the substep loop
            and executes the inital step from `t0` to `t0` + `dt0_substep`."""
            _y_tiled = jnp.tile(_y, (expansion_ratio,) + (1,) * (_y.ndim - 1))
            _sol = _vmapped_solver_step(t0, t0 + dt0_substep, _y_tiled, solver_state)
            return *_sol[:-1], jnp.max(_sol[-1])

        # Input Shape:  [n_expanded_particles, n_paths, ...]
        # Output Shape: [n_expanded_particles, n_paths, ...]
        def _body_fn(
            substep_count, state
        ) -> tuple[PyTree, Optional[PyTree], DenseInfo, _SolverState, RESULTS]:
            """Uses a step size of `dt0_substep`."""
            flat_substep_size = n_particles * n_paths**substep_count
            flat_subset_indices = jnp.expand_dims(
                jnp.arange(0, n_expanded_particles), tuple(range(y0.ndim - 1, 0, -1))
            )

            # Input Shape:  [n_sub_expanded_particles, n_paths, ...]
            # Output Shape: [n_sub_expanded_particles, n_paths, ...]
            def _reshape(arr):
                # Shape: [n_expanded_particles, 1, ...]
                flat_arr = arr[:n_sub_expanded_particles].reshape(-1, 1, *arr.shape[2:])
                # Probably unnecessary but helps with debugging.
                masked_flat_arr = jnp.where(
                    flat_subset_indices < flat_substep_size, flat_arr, jnp.inf
                )
                tiled_masked_flat_arr = jnp.tile(
                    masked_flat_arr, (1, n_paths) + (1,) * (y0.ndim - 2)
                )
                return tiled_masked_flat_arr

            # Split the state into element that need reshaping and those that don't.
            tree1, tree2 = eqx.partition(state, is_valid_array(y0.ndim - 1))
            tree1_reshaped = jtu.tree_map(_reshape, tree1)
            _state = eqx.combine(tree1_reshaped, tree2)

            # Step information
            _t0 = t0 + dt0_substep * substep_count
            _t1 = _t0 + dt0_substep
            _y = _state[0]
            _solver_state = _state[3]
            _sol = _vmapped_solver_step(_t0, _t1, _y, _solver_state)
            return *_sol[:-1], jnp.max(_sol[-1])

        init_state = _init_fn(y0)
        if self.n_substeps > 1:
            out_state = jax.lax.fori_loop(1, 1, _body_fn, init_state)
        elif self.n_substeps == 1:
            out_state = init_state
        else:
            raise ValueError(f"`n_substeps` must be >= 1; got {self.n_substeps}.")

        # Input Shape:  [n_expanded_particles, n_paths, ...]
        # Output Shape: [n_particles, n_paths, ...]
        def _recombination(arr):
            recombination_factor = arr.shape[0] // y0.shape[0]
            return self.recombination_kernel(recombination_factor, t0, arr, args)

        tree1, tree2 = eqx.partition(out_state, is_valid_array(y0.ndim - 1))
        recombined_state = jtu.tree_map(_recombination, tree1)
        # jax.debug.breakpoint()
        return eqx.combine(recombined_state, tree2)

    def func(self, terms: PyTree[AbstractTerm], t0: Scalar, y0: PyTree, args: PyTree):
        return self.recombination_kernel(self.solver.func(terms, t0, y0, args))


RecombinationSolver.__init__.__doc__ = """**Arguments:**

- `solver`: The solver to wrap.
- `recombination_kernel`: The recombination rule.
"""
