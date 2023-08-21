---
jupytext:
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: MCCDeploy
  language: python
  name: python3
---

# Markov chain cubature from scratch

The goal here is to introduce and explain, in a (hopefully) intuitive way, the concepts 
that underpin Markov chain cubature. A very basic understanding of stochastic calculus 
is assumed, but as no proofs are presented here, it should be accessible to non 
Mathematicians.

Code examples will be presented where appropriate, but the primary goal here is not to 
demonstrate operation of MCCube, but to instead explain its underpinning concepts.

## Quadrature and Cubature

Quadratures, and cubatures, are formulae for numerically integrating functions over 
weighted $n\text{-dimensional}$ regions. The cubature nomenclature implies that the 
integration region is expected to be of dimension $n \ge 2$, while quadrature implies 
dimension $n=1 $; in practice, many authors ignore the distinction and simply refer to 
any numerical integration formulae of the form

$$Q \left[f\right] := \sum_{i=1}^{k} B_i f(v_i) \approx \int\dotsi\int_{\mathbb{R}^d} w(x_1,\dots,x_d) f(x_1,\dots,x_d) dx_1,\dots,dx_d$$

as a quadrature/cubature formulae $Q$, where $B_i \in \mathbb{R}$ and $v_i \in \mathbb{R^d}$ are formula specific 
coefficients and vectors; $w:\mathbb{R}^d \to \mathbb{R}$ is a weight function/distribution; 
and $f:\mathbb{R}^d \to \mathbb{R}$ is the function to integrate/the integrand.

Such formulae $Q$ are said to be of degree $m$, with respect to a specific weight 
$w(\mathbf{x})$ and integration region $\Omega \in \mathbb{R}^d$, if they exactly 
integrate all $f \in \mathcal{P}^m(\Omega)$ (polynomials of degree at least $m$ over the specified region $\Omega$). That is to say, degree $m$ formulae $Q^m$ are exact for 
$f(\mathbf{x}) = \sum_{\alpha \in \mathcal{A}_m} c_{\alpha} \mathbf{x}^\alpha$
where $\mathbf{x}^{\alpha} = \prod_{j=1}^d x_j^{\alpha_j}$ are monomials and 
$c_\alpha \in \mathbb{R}$ are coefficients, for all multi-indexes $\alpha \in \mathcal{A}_m := \{(\alpha_1, \dots, \alpha_d) \in \mathbb{N}_0, \sum_{i=1}^d \alpha_i \le m\}$.

:::{admonition} A measure theoretic definition.
:class: note

For those familiar with Lebesgue-Steiltjies integration, the above can be more precisely denoted as

$$Q \left[f\right] := \int_\Omega f(d)\ d\hat{w}(x) \approx \int_\Omega f(x)\ dw(x),$$

where $f: \Omega \to \mathbb{R} \in L^1(\Omega, \mathcal{B}, w)$, and $\hat{w} = \sum_{i=1}^k B_i \delta_{x_i}$
is the quadrature/cubature measure. For all pratical formulae, where $\text{supp}(\hat{w}) < \text{supp}(w)$, 
one may interpret $\hat{w}$ as a compression of $w$ {cite:p}`satoshi2021`.

If you are not familiar with this notation then don't worry. Understanding the prior 
presentation, via improper Reimann Integrals, is sufficient for this article. 
:::

### Integrating non-polynoimial functions

While integrating polynomials is an important problem, in many practical cases one will
not have the luxury of dealing with such nice analytic functions. Thus, to be practical,
one must consider how the error of the integration formulae scales for more general 
$f \in C(\Omega)$.

The really neat thing to realize is that any formulae $Q^m$ will be a *good* integrator
for *any* continuous $f$ that can be well approximated by some $\hat{f} \in \mathcal{P}^{m}(\Omega)$,
such as the degree $m$ Taylor polynomial of $f$. In addition, when $f \in C^{m+1}(\Omega)$, 
one can show via the Peano kernel theorem that the formulae has an error bounded by 

$$
\left|\int_{\Omega}  w(x) f(x)\ dx - Q^m[f] \right| \le c\ \max_{x \in \Omega} \left| f^{(m+1)}(x)\right|,
$$

where the constant $c > 0$ is independent of $f$ {cite:p}`iserles2008` (note that the univariate 
case is shown for simplicty). This simple error bound is a result of $Q^m$, by definition, 
being an exact integrator of the degree $m$ Taylor polynomial $T^m[f](x)$ about any 
point in the domain. Noting that $f(x) = T^m[f](x) + r^m[f](x)$ and $r^m[f](x)$ is the Taylor remainder, the left hand side of the above inequality can 
be repesented as


$$
\left|\int_{\Omega} w(x) (T^m[f](x) + r^m[f](x))\ dx - Q^m[f]\right| = \left|\int_{\Omega} w(x) r^m[f](x)\ dx\right|,
$$
from which the right hand side of the inequality follows from the definiton of the 
Taylor expansion.

In the remainder of this article, only exact formulae are considered. Do not worry if 
the above error bound is confusing, the key takeaway is that $Q^m$ can still be useful 
for non-polynomial integrands. See {cite:p}`stroud1971` for an in depth consideration 
of the error analysis and other aspects of *approximate* cubature formulae.

### Constructing formulae

How exactly one constructs the coefficients $B_i$ and vectors $v_i$ that define a given formula $Q$, 
the existence of which is guaranteed by Tchakaloff (see {cite:t}`davis1967` and {cite:t}`mysovskikh1975`),
is a large topic in itself and will not be discussed in any depth here (interested readers 
are referred to {cite:p}`cools1997` and {cite:p}`stroud1971`). 

However, it is important to be aware of what makes a *good* construction/formula. That is, a 
construction with the fewest vectors $v_i$ to produce a formula $Q^m$ whose approximation 
error for polynomials of degree $> m$ is minimal. Such a construction can sometimes be 
further enhanced if the vectors are sparse or have other nice properties that can be 
leveraged to accelerate computation. 

### An example cubature

The notation used above is quite dense and on a first reading may be challenging to follow. 
However, it is just a rigorous presentation of a rather simple concept; multivariate 
polynomials can be exactly integrated, over weighted regions, by weighted sums of the 
polynomial evaluated at specially selected points.

To be concrete, consider the degree three Gaussian cubature formula from 
{cite:t}`stroudSecrest1963` ($E_n^{r^2} \text{3-1}$ in {cite:p}`stroud1971`), which 
exactly solves the following integral, for polynomial $f: \mathbb{R}^d \to \mathbb{R}$ of degree $m \le 3$,

$$\int_{\mathbb{R}^d} f(\mathbf{x})dP(\mathbf{x}) = \frac{1}{Z_c} 
\int\dotsi\int_{\mathbb{R}^d} f(x_1, \dots, x_d)\exp(-x_1^2 \dots -x_d^2)dx_1 \dots dx_d$$

where $P(x)$ is the probability measure of the $d\text{-dimensional}$ Gaussian 
distribution $X \sim \mathcal{N}(\mathbf{0}, \text{diag}(1/2))$ and $Z_c$ is a 
normalizing constant. If $Z_c$ is known, the above integral is the distribution's 
expectation $\operatorname{E}\left[f(X)\right]$; when $f(X) = X^\alpha$, a monomial, the
integral is the distribution's moment of degree $\sum_{i=1}^d \alpha_i$. Hence, the 
cubature formula can be used to compute all moments of $X$ of degree less than or equal 
to three. It is also exact for certain degree four moments, such as $E[X_i^2X_j^2]$.

Thus, consider the scenario where $d=2$ (two-dimensional Gaussian) and one wishes to 
compute the co-variance matrix $\operatorname{E}\left[X_iX_j\right]$ (moments of degree two) - using the above cubature 
formula. To perform the required computation one must:



1. **Look up the formula's vectors and coefficients:**<br>
    For $E_n^{r^2}$ they are as follows, where $r^2 = d/2$, $V=\pi^{d/2}$ is the unnormalized volume of the weighted integration region, and each row of the matrix $v$ is a cubature vector $v_i$.

    $$
    B_i = \frac{V}{2d}, \quad v=\begin{bmatrix}
    r  & 0 \\
    0  & r \\
    -r & 0 \\
    0  & -r\\
    \end{bmatrix}.
    $$

2. **Compute the normalization constant:**<br>
    The normalized volume must be one, the unnormalized volume is $V$, thus $Z_c=V$.

3. **Evaluate the formula:**<br>
    With all the pre-requisits now available, the cubature can computed as below,
    where $f(x_1, x_2) = x_i x_j$, and $k=2d$ (the number of cubature vectors).

    $$\frac{1}{Z_c}\sum_{i=1}^{k} B_i f(v_i).$$


Rather than doing this by hand, one can make use of {mod}`mccube.formulae` as 
shown below:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import numpy as np

from mccube.formulae import StroudSecrest63_31

jax.config.update("jax_platform_name", "cpu")  # Disable no GPU/TPU warning.

d = 2
expected_covariance = np.diag(np.ones(d)) / 2

# Step 1. Get the formula vectors and coefficients
cf = StroudSecrest63_31(dimension=d)
B_i = cf.coefficients
v_i = cf.vectors

# Step 2. Compute the normalization constant
z_c = cf.region.volume

# Step 3a. Evaluate the formula (Manual)
f_cov = lambda x: jnp.einsum("i, j -> ij", x, x) - x  # 2nd Central Moment
eval_vmap = jax.vmap(lambda v: B_i * f_cov(v), [0])
result = 1 / z_c * sum(eval_vmap(v_i))
assert np.isclose(expected_covariance, result).all()

# Step 3b. Evaluate the formula (mccube.formulae)
result, _ = cf(f_cov, normalize=True)
assert np.isclose(expected_covariance, result).all()

result_str = f"""
Expected result:\n {expected_covariance}\n
(Manual) Cubature result:\n {result}\n
(MCCube) Cubature result:\n {result}\n
"""
print(result_str)
```

Through a suitable affine transformation $\phi(x_1, \dots, x_n)$ of the vectors and 
coefficients, this cubature formula can be adapted to any parametrization of the above 
Gaussian distribution/weight (see pg 11 of {cite:p}`stroud1971` for details). Such a 
transform can be trivially applied in {mod}`mccube.formulae`:

```{code-cell} ipython3
expected_mean = np.ones(d)
expected_covariance = 5 * np.eye(d)
cf_affine = StroudSecrest63_31(d, mean=expected_mean, covariance=expected_covariance)
result, _ = cf_affine(f_cov, normalize=True)
result_str = f"""
Expected result:\n {expected_covariance}\n
(MCCube) Cubature result:\n {result}\n
"""
print(result_str)
```

## ODE Cubature
Readers are directed to the excelent {cite:p}`iserles2008` for a comprehensive 
introduction to the generalization of quadrature formulae to ODE solving.

:::{warning}
The remainder of this article is still under construction and should be assumed to be 
wrong for now!
:::

+++

## SDE Cubature
The equivalent SDE cubature of degree three extends the above example to the moments of an 
equivalent dimension Gaussian Process, $X = \{X_t | X_t \sim \mathcal{N}(\mu(t), 
\Sigma(t))\}_{t\ge0}$, that is described by the Itô integral equation

$$X_t^i = X^i_{t_0} + \int_{0}^{t}a^i(t, X_s)ds + \int_{0}^{t}b^{i,j}(t, X_s) dW^j_s$$

and alternatively the Itô SDE $dX^i_t = a^i(t, X_t)dt + b^{i,j}(t, X_t)dW^{j}_t$, 
where in this case $a^i(t, X_t) = 0$, $b^{i,j}(t, X_t) = \sqrt{\delta_{ij}/2}$, and 
$X^{i}_{t_0} \sim \mathcal{N}(\mathbf{0}, \delta_{ij}/2)$ is the initial condition. 

If one looks at the process at a single point in time $T$, and knows the corresponding
random variable $X_{t=T}$ is Gaussian, then its moments can be computed as per the 
standard cubature formulae above, the difficulty lies in extending this to moments of 
the full process

$$
\operatorname{E}\left[X_t^{\alpha} \right] = \operatorname{E}\left[\left(\int_0^t X_s ds \right)^{\alpha} \right]
$$ 

we need a way to compute the stochastic integral. The way to do this is via stochastic 
Taylor Expansion. As with many traditional/non-optimization based solution schemes 
one must discretise in time to actually solve numerically.

Nuance in the degree $m$... as a stochastic taylor "polynomial" degree.

While a cubature is valid for all algebraic polynomials, a Cubature on Wiener space is 
valid only for stochastic Taylor Polynomials (that is truncated stochastic taylor 
expansions, of degree m). The function f must be bounded and smooth up to degree l.

.. admonition:: A note on notation
    :class: note

    Strictly the above integral does not exist in that standard sense, as the measure 
    $dW^{j}_t$ is not sufficiently smooth w.r.t time. Instead, one must define a 
    *rougher* integral such as those of Itô and Stratanovich. See {cite:p}`allan2021` 
    for an overview and {cite:p}`kloeden1992` for an in depth discussion.

The salient limitation of SDE cubatures, constructed as per {cite:t}`lyons2004`,
is that the path count scales exponentially with the number of discrete time steps 
($\mathcal{O}(n^{m})$, where $n$ is the propagator expansion factor, and $m$ is the 
number of time-integration steps). MCC solves this problem by constructing the 
collection of paths as a markov chain, where the {class}`MCCubatureStep` acts as a 
transition kernel that employs recombination to maintain the path/particle count at 
every time step. Note that in MCCube the paths are usually interpreted as particle 
trajectories, as this provides a consistent physically analogy.

+++
