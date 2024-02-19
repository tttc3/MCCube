---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Quickstart
In this notebook we will explain, in detail, the quick example presented in the [README](/#example). By the end, you should be equipped with all the information needed to use MCCube for (weakly) solving any suitable SDE.

## Overdamped Langevin Equation
The [README](/#example) example presents an Markov Chain Cubature (MCC) variant of the Unadjusted Langevin Algorithm (ULA), where one attempts to weakly solve the overdamped Langevin equation

$$
dy_t = -\nabla f(y_t)\ dt + \sqrt{2}\ dW_t,
$$

given some sufficiently nice (interaction) potential $f \colon \mathbb{R}^d \to \mathbb{R}$.

The Langevin SDE is of practical interest thanks to its unique ergodic stationary distribution $\pi(x) \propto \exp(-f(x))$. Thus, if one wishes to draw samples from some negative logdensity function $f$, it is sufficent to solve the SDE (pathwise) for some arbitrary initial condition(s), up until the steady-state is reached.
One may then obtain $n$ samples from a single path (by observing the path state at $n$ different times), or alternatively obtain a single sample from $n$ independant paths (by observing the state of the $n$ paths at the end time).

For now, we will ignore the very real issue of practically identify if/when the steady-state has been reached (in general one must resort to using empircal *"diagnostic"* quantities).

### Solving the Langevin SDE via the ULA
The defacto standard approach to solving the above SDE, is to simulate each path via a Markov Chain Monte Carlo (MCMC) method. In this case, we specifically consider the somewhat flawed, Unadjusted Langvin Algorithim, obtained by discretising the SDE via the Euler-Maruyama method:

$$
Y_{i} = Y_{i} -\nabla f(Y_{i})\ h + \sqrt{2h} \Delta W_{i},
$$

where the step size $h=t_{i+1}-t_{i}$ is constant, and each $\Delta W_{i}$ is an idependant sample from a (potentially multi-variate) Gaussian variable with mean zero and diagonal covariance $h$.

### ULA in Diffrax
It is very easy to implement the ULA in Diffrax, as demonstrated in the below example, which generates 512 independant Markov chains by simulating the SDE via the Euler-Maruyama method, performing the standard unadjusted Langevin algorithm for a single initial condition ($Y_0$ is a d-dimensional vector with all elements equal to one). 

It is important to note that while the computation of the 512 chains is performed in parallel, this is not a "Parallel MCMC" method as each path is independant (unlike in MCC).

```python
import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import multivariate_normal

from mccube import gaussian_wasserstein_metric, unpack_particles

jax.config.update("jax_enable_x64", True)

key, rng_key = jr.split(jr.key(42))
n, d = 512, 10
t0 = 0.0
n_epochs = 1024
n_state_samples = 128
dt0 = 0.05
t1 = t0 + dt0 * n_epochs

# The initial state (repeated 512 times)
y0 = jr.multivariate_normal(key, jnp.zeros(d), jnp.eye(d), (n,))

target_mean = 2*jnp.ones(d)
target_cov = 3*jnp.eye(d)

# The "model" log-density, $-log f(Y_t)$.
def logdensity(p):
    return multivariate_normal.logpdf(p, mean=target_mean, cov=target_cov)

# Construct the overdamped Langevin equation.
ode = diffrax.ODETerm(lambda t, p, args: jax.vmap(jax.grad(logdensity))(p))
cde = diffrax.WeaklyDiagonalControlTerm(
    lambda t, p, args: jnp.sqrt(2.0),
    diffrax.UnsafeBrownianPath(shape=(n,d), key=key), # 512 d-dimensional standard Gaussian RVs.
)
terms = diffrax.MultiTerm(ode, cde)
solver = diffrax.Euler()

# Solve the SDE via Euler-Maruyama.
sol = diffrax.diffeqsolve(
    terms,
    solver,
    t0, 
    t1,
    dt0,
    y0,
    adjoint=diffrax.DirectAdjoint(),
    saveat=diffrax.SaveAt(ts=jnp.arange(t1 - dt0*n_state_samples, t1+dt0, dt0))
)

# Compare distribution of the chains at the final time step against the "model" distribution.i
def evaluate_method(particles, method_name, weighted=False):
    particles, weights = unpack_particles(particles, weighted)
    res_mean = jnp.average(particles, axis=0, weights=weights)
    res_cov = jnp.cov(particles, ddof=0, rowvar=False, aweights=weights)
    metric = gaussian_wasserstein_metric((
        target_mean, res_mean), (target_cov, res_cov)
    )
    print(f"[{method_name} | weighted={weighted}]\n2-Wasserstein distance: {metric}")

particles = sol.ys.reshape(-1, d)
evaluate_method(particles, "Diffrax ULA")

# [Diffrax ULA | weighted=False]
# 2-Wasserstein distance: (0.07028661938395156+0j)
```

### Adjusted Langevin Algorithm in Blackjax
ULA does not strictly obey the [detailed balance](https://en.wikipedia.org/wiki/Detailed_balance) properties required for a unique ergodic stationary distribution to exist, and as such, is unlikely to be used in practice.

A more realistic scenario would be the use of the [Blackjax](https://github.com/blackjax-devs/blackjax) package and one of its more advanced samplers. For example, the [Metropolis-Adjusted Langevin Algorithm (MALA)](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm), see demonstration below, which adjusts the ULA to ensure the detailed balance properties are enforced. 

```python
import blackjax

# Inference loop from:
# https://blackjax-devs.github.io/blackjax/examples/howto_sample_multiple_chains.html
def inference_loop(kernel, initial_state, n_epochs, num_chains, *, key):

    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, _ = kernel(keys, states)
        return states, states

    keys = jax.random.split(rng_key, n_epochs)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

key, sampler_key = jr.split(jr.key(42))
sampler = blackjax.mala(logdensity, dt0)
init_state = jax.vmap(sampler.init)(y0)
state = inference_loop(
    jax.vmap(sampler.step), init_state, n_epochs, n, key=sampler_key
)

particles = state.position[-n_state_samples:].reshape(-1, d)
evaluate_method(particles, "Blackjax MALA")

# [Blackjax MALA | weighted=False]
# 2-Wasserstein distance: (0.060678729884975705+0j)
```

## Markov Chain Cubature
Markov chain cubature allows us to take a fundamentally different approach to the problem of solving the Langevin SDE (equivalently obtaining samples from $f$). 
Rather than atempting to obtain (potenially $n$) independant pathwise solutions to the SDE, with MCC, one attempts to find a set of $n$ time-evolving dependant particles which at any point in time attempt to weakly solve the SDE (that is solve the SDE in law/distribution).

The crucial difference here is that paths traced by these particles need not coincide with any pathwise solutions of the SDE. The only requirement is that the distribution of these particles be identical to the distribution of all the infinitely many pathwise solutions.

### How MCC works
To better understand this difference, consider the MCC discretisation of the SDE:

$$
Y_{i+1} = Y_{i} - \nabla f(Y_{i})\ h + \sqrt{2h}\ Z_i
$$

where $Z_i$ is a matrix of $k$ gaussian cubature vectors (those familiar with Cubature on Wiener Space should note that $\sqrt{h}Z_k$ ammounts to the evaluation of piecewise linear cubature paths at a right hand node). The key thing to notice is that after every step of the above equation, the number of particles will expand by a factor of $k$; the particles will evolve along $k$ deterministic trajectories. The idea is that while these trajectories may not coincide with any sample paths, they do satisfy certain weak quantities of the SDE (certain degree itterated integrals in the Stratonovich Taylor expansion of the SDE).

The major problem with the above approach is that the particle count scales according to $\mathcal{O}(k^N)$, where $N$ is the number of steps, very quickly leading to an explosion in particle count, and computational intractibility. The solution is to realise that when simulating a cloud of $n$ paticles there may be many redundant/uninformative trajectories (particularly in higher dimensions). That is to say, the $k \times n$ particles may be recombined into $n$ (potentially new) particles which are under some (pseudo-)metric are as close as possible/as informative as the fully expanded particles. This recombination step makes the particles an interacting/dependant set.

In the parlance of MCC, one may consider the raw step as defining a transition kernel, called the cubature kernel, which when composed with a recombination kernel/ defines an approximate cubature kernel. Unless otherwise stated, it is always assumed that an MCC has an approximate cubature kernel.

### ULA in MCCube
Now returning to the example in the [README](/#example) reproduced below:

```python
from mccube import (
    GaussianRegion,
    Hadamard, 
    LocalLinearCubaturePath,
    MCCSolver, 
    MCCTerm,
    MonteCarloKernel,
    PartitioningRecombinationKernel,
    BinaryTreePartitioningKernel,
)

key = jr.key(42)
gaussian_cubature = Hadamard(GaussianRegion(d))
mcc_cde = diffrax.WeaklyDiagonalControlTerm(
    lambda t, p, args: jnp.sqrt(2.0),
    LocalLinearCubaturePath(gaussian_cubature)
)
terms = MCCTerm(ode, mcc_cde)
kernel = MonteCarloKernel(n, key=key)

solver = MCCSolver(diffrax.Euler(), kernel)

sol = diffrax.diffeqsolve(
    terms,
    solver,
    t0, 
    t1,
    dt0,
    y0,
    saveat=diffrax.SaveAt(ts=jnp.arange(t1-dt0*10, t1+dt0, dt0))
)
particles = sol.ys[-n_state_samples].reshape(-1, d)
evaluate_method(particles, "MCCube ULA | MC Kernel")

# [MCCube ULA | MC Kernel | weighted=False]
# 2-Wasserstein distance: (3.3503528099696664+0j)
```

Notice that the performance in this specific case is significantly worse than the standard ULA/MLA methods. This stands to highlight that the performance of MCC depends heavily on the choice of kernel, and the geometry of the underlying problem. For example, by simply partitioning the proposal particles before performing recombination/resampling, we can more than halve the 2-Wasserstein distance (shown below).

```python
n_parts = 256
kernel = PartitioningRecombinationKernel(
    BinaryTreePartitioningKernel(n_parts),
    MonteCarloKernel(n // n_parts, key=key)
)
solver = MCCSolver(diffrax.Euler(), kernel)

sol = diffrax.diffeqsolve(
    terms,
    solver,
    t0, 
    t1,
    dt0,
    y0,
    saveat=diffrax.SaveAt(ts=jnp.arange(t1-dt0*10, t1+dt0, dt0))
)
particles = sol.ys[-n_state_samples].reshape(-1, d)
evaluate_method(particles, "MCCube ULA | Partitioned MC Kernel")

# [MCCube ULA | Partitioned MC Kernel | weighted=False]
# 2-Wasserstein distance: (1.3772840507440691+0j)
```

### Weighted MCC
The above examples treat all particles as having equal mass/weight. That is to say, one can consider the particles as representing a discrete measure 
$$\mu = \sum_{i=1}^n \lambda_i \delta_{x_i},$$
where each $\lambda_i$ is a probability weight/mass and each $x_{i}$ is a particle.
In the above examples, the guassian cubature assigns equal weight to each proposal particle (update path), and the recombination kernels are weight invariant. However, in some cases the gaussian cubature will assign unequall weights, and the recombination kernel will be weight dependant. 

To utilise these weights in MCCube is relatively simple, requiring only a few minor modifications to the prior example.

```python
from mccube import pack_particles

weights = jnp.ones((y0.shape[0]))
y0_weighted = pack_particles(y0, weights)
n, d = y0_weighted.shape

solver = MCCSolver(diffrax.Euler(), kernel, weighted=True)
sol = diffrax.diffeqsolve(
    terms,
    solver,
    t0, 
    t1,
    dt0,
    y0_weighted,
    saveat=diffrax.SaveAt(ts=jnp.arange(t1-dt0*10, t1+dt0, dt0))
)
particles = sol.ys[-n_state_samples].reshape(-1, d)
evaluate_method(
    particles, "MCCube ULA | Partitioned MC Kernel", weighted=True
)

# [MCCube ULA | Partitioned MC Kernel | weighted=True]
# 2-Wasserstein distance: (1.3772840507440556+0j)
```

In the above case, the result is uchanged eventhough weight dependance has been enabled. This is because the [`mccube.Hadamard`][] formula used in the control path has homogeneous weights, along with the initial condition.


## Variational Methods
One may notice similarities between MCC and variational approaches such as [Stein variational gradient descent (SVGD)](https://arxiv.org/abs/1608.04471). Research is currently underway to better asses when and where MCC can provide superior performance to some of these methods (in addition to the non-variational approaches noted prior).

The below example demonstrates the case of SVGD for the same inference/sampling problem considered above.

```python
import optax

def svgd_inference_loop(kernel, initial_state, n_epochs):

    @jax.jit
    def one_step(_, state):
        _state = kernel(state)
        return _state

    return jax.lax.fori_loop(0, n_epochs, one_step, initial_state)

optimizer = optax.adam(0.1)
rbf_kernel = blackjax.vi.svgd.rbf_kernel
update_heuristic = blackjax.vi.svgd.update_median_heuristic

sampler = blackjax.vi.svgd.svgd(jax.grad(logdensity), optimizer, rbf_kernel, update_heuristic)
init_state = sampler.init(y0)
state = svgd_inference_loop(sampler.step, init_state, 128)

evaluate_method(state.particles, "SVGD")

# [SVGD | weighted=False]
# 2-Wasserstein distance: (1.7786431585886486+0j)
```

Like with MCC, the performance in this case is worse than ULA/MALA, again highlighting the importance of selecting appropriate kernels and (for SVGD) optimizers. Interested readers are encouraged to play around with the above examples and to identify parameterisations which yield enhanced performance.

## Next Steps
Equiped with the above knowledge, it should be possible to start experimenting with MCCube. 

API documentation can be found [here](api/), and please feel free to submit an issue if there are any tutorials or guides you would like to see added to the documentation.

!!! tip
    To get the most out of this package it is helpful to be familiar with all the         bells and whistles of [Diffrax](https://github.com/patrick-kidger/diffrax). 
