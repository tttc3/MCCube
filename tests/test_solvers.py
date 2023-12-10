import chex
import inspect
import jax
from absl.testing import absltest, parameterized

import diffrax
from diffrax import (
    AbstractSolver,
    AbstractItoSolver,
    AbstractStratonovichSolver,
    ODETerm,
    WeaklyDiagonalControlTerm,
    MultiTerm,
    diffeqsolve,
    VirtualBrownianTree,
    UnsafeBrownianPath,
    SaveAt,
    RecursiveCheckpointAdjoint,
    DirectAdjoint,
)
import equinox as eqx
from jax.scipy.stats import multivariate_normal
import jax.numpy as jnp
import jax.tree_util as jtu
import time

# from mccube.components import MonteCarloRecombinator
from mccube.diffrax.solvers import RecombinationSolver
from mccube.formulae import WienerSpace, LyonsVictoir04_512
from mccube.metrics import cubature_target_error
from mccube.components.recombinators import MonteCarloRecombinator
import numpy as np

jax.config.update("jax_enable_x64", True)


def solver_predicate(object: object) -> bool:
    is_class = inspect.isclass(object)
    if is_class:
        is_solver = issubclass(object, AbstractSolver)
        is_not_abstract = "Abstract" not in object.__name__
    return is_class and is_solver and is_not_abstract


SOLVERS = {(i, v) for i, v in inspect.getmembers(diffrax.solver, solver_predicate)}
SDE_SOLVERS = {
    (i, v)
    for i, v in SOLVERS
    if issubclass(v, (AbstractItoSolver, AbstractStratonovichSolver))
}


def solver_controllers(
    solvers=SOLVERS, sde_only=True, include_adaptive=True
) -> set[tuple[str, diffrax.AbstractSolver, diffrax.AbstractStepSizeController]]:
    """Generate name, solver, controller tuples from a set of diffrax.AbstractSolvers."""
    adaptive = {x for x in solvers if issubclass(x[1], diffrax.AbstractAdaptiveSolver)}
    non_adaptive = solvers - adaptive
    non_adaptive_solver_controler = {
        (i, v(), diffrax.ConstantStepSize()) for i, v in non_adaptive
    }
    # Adaptive solvers and special case of HalfSolver.
    adaptive -= {x for x in solvers if issubclass(x[1], diffrax.HalfSolver)}
    PID_controller = diffrax.PIDController(
        rtol=1e-1, atol=1e-2, pcoeff=0.1, icoeff=0.3, dcoeff=0
    )
    adaptive_solver_controler = {(i, v(), PID_controller) for i, v in adaptive}
    for i, v in non_adaptive:
        if issubclass(v, diffrax.Euler) and sde_only:
            continue
        adaptive_solver_controler.add(
            (f"HalfSolver_{i}", diffrax.HalfSolver(v()), PID_controller)
        )
    if include_adaptive:
        return non_adaptive_solver_controler.union(adaptive_solver_controler)
    return non_adaptive_solver_controler


class InferenceTests(chex.TestCase):
    def setUp(self):
        # Problem settings
        self.target_dimension = 3
        self.target_mean = 2 * np.ones(self.target_dimension)
        self.target_cov = 3 * np.eye(self.target_dimension)
        # Solver settings
        self.key = jax.random.PRNGKey(42)
        self.particles = 2
        self.epochs = 512
        self.dt0 = 0.00001
        self.t0 = 0.0
        self.t1 = self.epochs * self.dt0
        self.y0 = jax.random.multivariate_normal(
            self.key,
            self.target_mean - 2,
            self.target_cov / 3,
            shape=(self.particles,),
        )
        # Open a file for writing the results
        self.results = []

    def tearDown(self) -> None:
        print(self.results, file=open("out.txt", "+w"))

    def ula_cde(self, unsafe=False, cubature=None, as_ode=False):
        @eqx.filter_vmap
        @eqx.filter_grad
        def target_grad_logdensity(p):
            return multivariate_normal.logpdf(p, self.target_mean, self.target_cov)

        def ode_vf(t, p, args):
            return target_grad_logdensity(p)

        ode = ODETerm(ode_vf)
        # Make adjustment to the vector field to handle cubature weights.
        if cubature:

            def ode_vf_cubature(t, p, args):
                points = p[0, ..., :-1]
                weights = jnp.expand_dims(p[0, ..., -1], -1)
                updated_points = ode_vf(t, points, args)
                new_p = jnp.concatenate([updated_points, weights], -1)
                return jnp.tile(new_p, (cubature.point_count, 1))

            ode = ODETerm(ode_vf_cubature)

        def cde_vf(t, p, args):
            return np.sqrt(2.0)

        y0 = self.y0
        adjoint = RecursiveCheckpointAdjoint()

        key, _ = jax.random.split(self.key)
        cde_path = VirtualBrownianTree(
            self.t0,
            self.t1,
            self.dt0 / 10,
            shape=(self.particles, self.target_dimension),
            key=key,
        )
        if cubature:
            cde_path = cubature
            weights = np.zeros_like(y0[..., 0])
            y0 = np.concatenate([y0, weights[..., None]], -1)[:, None, :]
            y0 = np.tile(y0, (1, cubature.point_count, 1))

        elif unsafe:
            cde_path = UnsafeBrownianPath(
                shape=(self.particles, self.target_dimension), key=key
            )
            adjoint = DirectAdjoint()
        cde = WeaklyDiagonalControlTerm(cde_vf, cde_path)
        # if as_ode:
        #     cde = cde.to_ode()
        controlled_differential_equation = MultiTerm(ode, cde)

        return jtu.Partial(
            diffeqsolve,
            terms=controlled_differential_equation,
            t0=self.t0,
            t1=self.t1,
            dt0=self.dt0,
            y0=y0,
            saveat=SaveAt(t0=True, t1=True),
            adjoint=adjoint,
            # max_steps=None
        )

    def compilation_analysis(self, func, *args, **kwargs):
        tic = time.time()
        aot = eqx.filter_jit(func).lower(*args, **kwargs).compile()
        jax.block_until_ready(aot)
        toc = time.time()
        _aot = aot.compiled
        aot_cost = _aot.cost_analysis()[0]
        aot_cost["gen_code_size"] = _aot.memory_analysis().generated_code_size_in_bytes
        # Not all compilations have an optimal_seconds
        try:
            optimal = aot_cost["optimal_seconds"]
        except KeyError:
            optimal = None

        print(
            f"""Compilation Costs:
            Time: {toc-tic}
              - flops: {aot_cost['flops']}
              - bytes_accessed: {aot_cost['bytes accessed']}
              - transcendentals: {aot_cost['transcendentals']}
              - optimal_seconds: {optimal}
            """
        )
        return aot, aot_cost

    def runtime_analysis(self, func, *args, **kwargs):
        tic = time.time()
        solution = func(*args, **kwargs)
        jax.block_until_ready(solution)
        toc = time.time()
        dt = toc - tic
        y_sol = solution.ys[-1]
        if y_sol.ndim > 2:
            y_sol = y_sol.reshape(-1, y_sol.shape[-1])[..., :-1]

        # jax.debug.breakpoint()
        cost = (
            dt,
            cubature_target_error(y_sol, self.target_mean, self.target_cov),
        )
        print(cost)
        return solution, cost

    def solver_analysis(
        self, solver, controller, unsafe=False, cubature=None, as_ode=False
    ):
        func = self.ula_cde(unsafe, cubature, as_ode)
        aot, aot_cost = self.compilation_analysis(
            func, solver=solver, stepsize_controller=controller
        )
        sol, sol_cost = self.runtime_analysis(
            aot, solver=solver, stepsize_controller=controller
        )
        self.results.append((self._testMethodName, aot_cost, sol_cost))


class DiffraxTests(InferenceTests):
    @parameterized.named_parameters(solver_controllers(SDE_SOLVERS))
    def test_baseline_virtual(self, solver, controller):
        self.solver_analysis(solver, controller)

    # Won't work for adaptive solvers.
    @parameterized.named_parameters(
        solver_controllers(SDE_SOLVERS, include_adaptive=False)
    )
    def test_baseline_unsafe(self, solver, controller):
        self.solver_analysis(solver, controller, unsafe=True)

    @parameterized.named_parameters(solver_controllers(SDE_SOLVERS))
    def test_cubature(self, solver, controller):
        cow = LyonsVictoir04_512(WienerSpace(self.target_dimension))
        solver = RecombinationSolver(solver, MonteCarloRecombinator(self.key))
        self.solver_analysis(solver, controller, cubature=cow)

    # TODO: implement suitable path derivative.
    # @parameterized.named_parameters(solver_controllers(SOLVERS))
    # def test_cubature_ode(self, solver, controller):
    #     cow = LyonsVictoir04_512(WienerSpace(self.target_dimension))
    #     solver = RecombinationSolver(solver, MonteCarloRecombinator(self.key))
    #     self.solver_analysis(solver, controller, cubature=cow, as_ode=True)

    def test_recombination_solver_correctness(self, solver):
        # Test substeps.
        # Test recombination.
        ...

    def _blackjax_inference_loop(self, key, kernel, initial_state, num_samples):
        def one_step(state, key):
            state, _ = kernel(key, state)

        keys = jax.random.split(key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)
        return states


if __name__ == "__main__":
    absltest.main()
