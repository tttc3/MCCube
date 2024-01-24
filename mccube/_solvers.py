"""Defines custom solvers for performing MCC in diffrax.

See [`diffrax.AbstractSolver`][] for further information on the solvers API.
"""
import warnings
from typing import Callable, ClassVar, TypeAlias

from diffrax import (
    AbstractSolver,
    AbstractTerm,
    AbstractWrappedSolver,
    Euler,
    LocalLinearInterpolation,
    RESULTS,
)
from jaxtyping import PyTree

from ._custom_types import (
    Args,
    BoolScalarLike,
    DenseInfo,
    Particles,
    RealScalarLike,
)
from ._kernels import AbstractRecombinationKernel
from ._term import MCCTerm
from ._utils import pack_particles, unpack_particles

_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None


class MCCSolver(AbstractWrappedSolver[_SolverState]):
    r"""Markov chain cubature solver for [`diffrax.diffeqsolve`][].

    Composes a [`diffrax.AbstractSolver`][] with a [`mccube.AbstractRecombinationKernel`][],
    such that a single step of the solver is equivalent to the evaluation of an
    approximate cubature kernel from $t_0$ to $t_1$.

    If the recombination kernel is not present, the wrapped solver step is equivalent
    to an exact cubature kernel from $t_0$ to $t_1$. Such a kernel will fail to preserve
    the shape of $y(t)$ across steps and, thus, is incompatible with diffrax.

    However, one can subdivide the time interval into `n_substeps` and compute the exact
    kernel for each sub-step before finally composing the shape preserving recombination
    kernel. This provides a useful dial for tuning the tradeoff between memory usage and
    recombination information loss.

    Example:
        ```python
        import jax.numpy as jnp
        import jax.random as jr
        from diffrax import diffeqsolve, Euler

        key, rng_key = jr.split(jr.PRNGKey(42))
        t0, t1 = 0.0, 1.0
        dt0 = 0.001
        particles = jnp.ones((32,8))
        weights = jr.uniform(rng_key, (32,))
        y0 = mccube.pack_particles(particles, weights)
        n, d = y0.shape

        gaussian_cubature = mccube.Hadamard(mccube.GaussianRegion(d))
        cubature_control = mccube.LocalLinearCubaturePath(gaussian_cubature)
        ode = ODETerm(lambda t,y,args: -y)
        cde = WeaklyDiagonalControlTerm(lambda t,y,args: jnp.sqrt(2), cubature_control)
        terms = mccube.MCCTerm(ode, cde)

        kernel = mccube.MonteCarloKernel(n, key=key)
        solver = mccube.MCCSolver(Euler(), kernel, n_substeps=2, weighted=True)
        sol = diffeqsolve(solver, terms, t0, t1, y0)
        ```

    Attributes:
        solver: a [`diffrax.AbstractSolver`][] which, in conjuction with the `terms`,
            defines an exact cubature kernel. Note: support is only provided for the
            [`diffrax.Euler`][] solver at present.
        recombination_kernel: a callable which takes the interval end time, $t_1$, the
            potentially expanded state $y(t_1)$, and any additional $\text{args}$, and
            yields a recombined (shape preserved) state $\hat{y}(t_1)$.
        n_substeps: the number of steps, $n_s$, to subdivide the interval $[t_0, t_1]$
            into. Equivalently can be considered as the number of exact kernel
            evluations. **Note: memory scales with $\mathcal{O}(k^{n_s})$, where $k$ is
            the number of cubature vectors/paths.**
    """

    solver: AbstractSolver[_SolverState]  # type: ignore
    recombination_kernel: AbstractRecombinationKernel
    n_substeps: int = 1
    weighted: bool = False
    term_structure: ClassVar = MCCTerm
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def __check_init__(self):
        if not isinstance(self.solver, Euler):
            warnings.warn(
                f"""Support is only provided for the diffrax.Euler solver at present;
                got {self.solver}. Expect undefined behaviour!"""
            )
        if self.n_substeps < 1:
            raise ValueError(f"n_substeps must be at least one; got {self.n_substeps}")

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Particles,
        args: Args,
    ):
        return self.solver.init(terms, t0, t1, y0, args)

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Particles,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Particles, Particles | None, DenseInfo, _SolverState, RESULTS]:
        _y0, weights = unpack_particles(y0, self.weighted)
        n, d = _y0.shape
        dt_substep = (t1 - t0) / self.n_substeps
        _y0 = _y0[:, None, :]
        _t0 = t0
        _t1 = t0 + dt_substep

        for i in range(0, self.n_substeps):
            _t0 = _t1
            _t1 = _t0 + dt_substep
            _sol = self.solver.step(terms, _t0, _t1, _y0, args, solver_state, made_jump)
            _sol = (_sol[0].reshape(-1, d), *_sol[1:])
            _y0 = _sol[0][:, None, :]
            if weights is not None:
                cde_cubature_weights = terms.term.cde.control.weights[..., None]
                weights = weights[None, ...] * cde_cubature_weights
                weights = weights.reshape(-1)
        y1 = _sol[0]  # type: ignore
        y1_packed = pack_particles(y1, weights)
        y1_hat = self.recombination_kernel(t0, y1_packed, args, self.weighted)
        # Used to renormalize the weights post recombination.
        y1_res = pack_particles(*unpack_particles(y1_hat, weighted=self.weighted))
        dense_info = dict(y0=y0, y1=y1_hat)
        return (y1_res, _sol[1], dense_info, *_sol[3:])  # type: ignore

    def func(
        self, terms: PyTree[AbstractTerm], t0: RealScalarLike, y0: Particles, args: Args
    ) -> Particles:
        return self.recombination_kernel(
            t0, self.solver.func(terms, t0, y0, args), args, self.weighted
        )


MCCSolver.__init__.__doc__ = r"""Args:
    solver: a standard diffrax solver which, in conjuction with :data:`terms`,
        defines an exact cubature kernel. Note: only tested for the Euler solve.
    recombination_kernel: a callable which takes the interval end time, $t_1$, the
        potentially expanded state $y(t_1)$, and any additional $\text{args}$, and
        yields a recombined (shape preserved) state $\hat{y}(t_1)$.
    n_substeps: the number of steps to subdivide the interval $[t_0, t_1]$ into.
        Equivalently can be considered as the number of exact kernel evluations.
        **Note: memory scales with $\mathcal{O}(n^\text{n_substeps})$, where $n$ is
        the number of cubature vectors/paths.**
"""
