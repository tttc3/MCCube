"""Markov chain cubature inference via diffrax."""

from collections.abc import Callable
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from diffrax import AbstractSolver, AbstractWrappedSolver
from diffrax.custom_types import DenseInfo  # To replace in v0.5.1
from diffrax.solution import RESULTS
from diffrax.term import AbstractTerm
from jaxtyping import PyTree

from ._custom_types import (
    XP,
    Args,
    BoolScalarLike,
    IntScalarLike,
    P,
    RealScalarLike,
)
from ._kernels import AbstractRecombinationKernel
from ._utils import if_valid_array, is_valid_array

_SolverState = TypeVar("_SolverState")


class MCCubatureSolver(AbstractWrappedSolver):
    r"""Markov chain cubature solver for diffrax.

    Composes a diffrax :class:`AbstractSolver` with a :class:`RecombinationKernel`,
    such that a single step of the solver, is equivalent to the evaluation of an
    approximate cubature kernel from $t_0$ to $t_1$.

    If the recombination kernel is not present, the wrapped solver step is equivalent
    to an exact cubature kernel from $t_0$ to $t_1$. Such a kernel will fail to preseve
    the shape of $y(t)$ across steps and, thus, is incompatible with diffrax.

    However, one can subdivide the time interval into :attr:`n_substeps` and compute
    the exact kernel for each sub-step before finally composing the shape preserving
    recombination kernel. This provides a useful dial for tuning the tradeoff between
    memory usage and recombination information loss.

    Attributes:
        solver: a standard diffrax solver which, in conjuction with :data:`terms`,
            defines an exact cubature kernel.
        recombination_kernel: a callable which takes the interval end time, $t_1$, the
            potentially expanded state $y(t_1)$, and any additional $\text{args}$, and
            yields a recombined (shape preserved) state $\hat{y}(t_1)$.
        n_substeps: the number of steps to subdivide the interval $[t_0, t_1]$ into.
            Equivalently can be considered as the number of exact kernel evluations.
            **Note: memory scales with $\mathcal{O}(n^\text{n_substeps})$, where $n$ is
            the number of cubature vectors/paths.**
    """

    solver: AbstractSolver
    recombination_kernel: AbstractRecombinationKernel | Callable[
        [RealScalarLike, RealScalarLike, XP, Args], P
    ]
    n_substeps: IntScalarLike = 1

    def __check_init__(self):
        assert self.n_substeps >= 1

    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def interpolation_cls(self):  # pyright: ignore
        return self.solver.interpolation_cls

    @property
    def nonlinear_solver(self):
        return self.solver.nonlinear_solver

    @property
    def scan_kind(self):
        return self.solver.scan_kind

    def order(self, terms: PyTree[AbstractTerm]) -> int | None:
        """Order of the solver for solving ODEs.

        Defaults to the wrapped :attr:`solver` definition.

        Args:
            terms: The PyTree of terms representing the vector fields and controls.

        Returns:
            The error order.
        """
        return self.solver.order(terms)

    def strong_order(self, terms: PyTree[AbstractTerm]) -> RealScalarLike | None:
        """Strong order of the solver for solving SDEs.

        Defaults to the wrapped :attr:`solver` definition.

        Args:
            terms: The PyTree of terms representing the vector fields and controls.

        Returns:
            The strong error order.
        """
        return self.solver.strong_order(terms)

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: P,
        args: Args,
    ) -> _SolverState:
        """Initialise any hidden state for the solver.

        Args:
            terms: The PyTree of terms representing the vector fields and controls.
            t0: The start of the region of integration.
            t1: The end of the region of integration.
            y0: The initial particles/state.
            args: Any additional arguments to pass to the :data:`terms`.

        Returns:
            The initial solver state; used the first time :meth:`step` is called.
        """
        _y0 = self._tile_particles(jnp.shape(y0), y0)
        return eqx.filter_vmap(lambda y: self.solver.init(terms, t0, t1, y, args))(_y0)

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: P,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[P, P | None, DenseInfo, _SolverState, RESULTS]:
        r"""Make a single step of the solver.

        Each step is made over the specified interval $[t_0, t_1]$.

        Args:
            terms: The PyTree of terms representing the vector fields and controls.
            t0: The start of the interval over which the step is made.
            t1: The end of the interval over which the step is made.
            y0: The current value of the particles/solution at $t_0$.
            args: Any additional arguments to pass to the :data:`terms`.
            solver_state: Any evolving state for the solver itself, at $t_0$.
            made_jump: Whether there was a discontinuity in the vector field at $t_0$.
                Some solvers (notably FSAL Runge-Kutta solvers) may assume that there
                are no jumps and re-use information between steps; this indicates that
                a jump has just occured and this assumption is not true.

        Returns:
            A tuple of several objects:

            - The value of the recombined particles/solution at $t_1$.
            - A recombined local error estimate made during the step. (Used by adaptive
              step size controlers to change the step size). May be :code:`None` if no
              estimate was made.
            - Some dictonary of information that is passed to the :attr:`solver`
              interpolation routine to calculate dense output (Used with
              :code:`SaveAt(ts=...)` or :code:`SaveAt(dense=...)`).
            - The value of the sovler state at $t_1$.
            - An integer (corresponding to :data:`diffrax.RESULTS`) indicating whether
              the step happend successfully, or if it failed for some reason.
        """
        # Identify and calculate input and output shape information.
        y0_shape = jnp.shape(y0)
        y0_ndim = jnp.ndim(y0)
        n_particles, n_paths, *_ = y0_shape
        expansion_ratio = n_paths ** (self.n_substeps - 1)
        n_expanded_particles = n_particles * expansion_ratio
        n_sub_expanded_particles = n_expanded_particles // n_paths
        dt0_substep = (t1 - t0) / self.n_substeps

        ## Map the solution step over the leading axes of the input.
        # Input Shape:  [n_particles, n_paths, ...]
        # Output Shape: [n_particles, n_paths, ...]
        cubature_kernel = eqx.filter_vmap(
            jtu.Partial(self.solver.step, terms), in_axes=if_valid_array(0)
        )
        recombination_kernel = jtu.Partial(self.recombination_kernel, t0, t1, args=args)

        ## For computing the initial solution sub-step.
        # Input Shape:  [n_particles, n_paths, ...]
        # Output Shape: [n_expanded_particles, n_paths, ...]
        def _init_fn(
            _y: P,
        ) -> tuple[XP, XP | None, DenseInfo, PyTree | None, RESULTS]:
            """Tiles the input `_y` to provide shape preservation over the substep loop
            and executes the inital step from `t0` to `t0` + `dt0_substep`."""
            _y_tiled = self._tile_particles(y0_shape, _y)
            _sol = cubature_kernel(
                t0, t0 + dt0_substep, _y_tiled, args, solver_state, made_jump
            )
            _sol = _moveaxis(_sol, y0_ndim)
            return _sol[:-1] + (jnp.max(_sol[-1]),)

        ## For computing subsequent solution sub-steps.
        # Input Shape:  [..., n_expanded_particles, n_paths, ...]
        # Output Shape: [..., n_expanded_particles, n_paths, ...]
        def _body_fn(
            substep_count: IntScalarLike,
            state: tuple[XP, XP | None, DenseInfo, PyTree | None, RESULTS],
        ) -> tuple[XP, XP | None, DenseInfo, PyTree | None, RESULTS]:
            """Uses a step size of `dt0_substep`."""

            ## Flatten multi-path results into new particles for the next sub-step.
            # Input Shape:  [..., n_sub_expanded_particles, n_paths, ...]
            # Output Shape: [..., n_sub_expanded_particles, n_paths, ...]
            def _flatten_update(_arr: XP) -> XP:
                # Handle arrays of dimension y0.ndim + 1 that arise in the solution
                # pytree for RK solvers.
                _arr = jnp.moveaxis(_arr, 0, y0_ndim - jnp.ndim(_arr))
                # Flatten update particles
                target_shape = (-1, 1, *_arr.shape[2:])
                _flat_arr = _arr[:n_sub_expanded_particles].reshape(target_shape)
                _tiled_flat_arr = _tile_paths(y0_shape, _flat_arr)
                # Revert array handling from the beginning and return.
                return jnp.moveaxis(_tiled_flat_arr, y0_ndim - _arr.ndim, 0)

            # Split the state into elements that need reshaping and those that don't.
            _state = _valid_tree_map(_flatten_update, state, y0_ndim)

            # Step information
            _t0 = t0 + dt0_substep * substep_count
            _t1 = _t0 + dt0_substep
            _y = _state[0]
            _solver_state = _state[3]
            _sol = cubature_kernel(_t0, _t1, _y, args, _solver_state, made_jump)
            _sol = _moveaxis(_sol, y0_ndim)
            return _sol[:-1] + (jnp.max(_sol[-1]),)

        ## Execute the actual stepping computations here.
        out_state = init_state = _init_fn(y0)
        if self.n_substeps > 1:
            out_state = jax.lax.fori_loop(1, self.n_substeps, _body_fn, init_state)

        # Perform the recombination.
        recombination_state = out_state[:2]
        passthrough_state = out_state[2:]
        recombined_state = _valid_tree_map(
            recombination_kernel,
            recombination_state,
            y0_ndim - 1,
        )

        # Update the dense info with the recombined points.
        # Note that the interpolation coefficients will be incorrect. Although, this
        # should not be an issue as only the terminal state is of interest in MCC.
        passthrough_state[0]["y0"] = self._tile_particles(y0_shape, y0)
        passthrough_state[0]["y1"] = self._tile_particles(y0_shape, recombined_state[0])
        return recombined_state + passthrough_state

    def func(
        self, terms: PyTree[AbstractTerm], t0: RealScalarLike, y0: P, args: Args
    ) -> P:
        """Evaluate the vector field at a point.

        Unlike :meth:`step`, evluates the vector field at a specific point, rather than
        over an interval. This is needed for things like selecting an initial step size.

        Args:
            terms: The PyTree of terms representing the vector fields and controls.
            t0: The start of the region of integration.
            t1: The end of the region of integration.
            y0: The initial particles/state.
            args: Any additional arguments to pass to the :data:`terms`.

        Returns:
            The evaluate of the vector field at $(t_0, y(t_0)$.
        """
        f = self.recombination_kernel(
            t0, t0, self.solver.func(terms, t0, y0, args), args
        )
        return f

    def _tile_particles(self, target_shape: tuple[int, ...], _arr: PyTree) -> PyTree:
        _, n_paths, *_ = target_shape
        expansion_ratio = n_paths ** (self.n_substeps - 1)
        return jnp.tile(_arr, (expansion_ratio,) + (1,) * (len(target_shape) - 1))


def _tile_paths(target_shape: tuple[int, ...], _arr: PyTree) -> PyTree:
    _, n_paths, *_ = target_shape
    return jnp.tile(_arr, (1, n_paths) + (1,) * (len(target_shape) - 2))


# Need to reorder the axes for arrays of dimension greater than `y0` so all
# arrays in the solution pytree have the batched axes at `-y0.ndim`.
def _moveaxis(_tree: PyTree, _ndim: int):
    return _valid_tree_map(
        lambda x: jnp.moveaxis(x, 0, jnp.ndim(x) - _ndim), _tree, _ndim
    )


def _valid_tree_map(map, tree, dim):
    tree1, tree2 = eqx.partition(tree, is_valid_array(dim))
    tree1_updated = jtu.tree_map(map, tree1)
    return eqx.combine(tree1_updated, tree2)
