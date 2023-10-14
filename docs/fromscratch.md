---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: MCCDeploy
  language: python
  name: python3
---

# Markov chain cubature from scratch

The goal here is to introduce and explain the concepts that underpin Markov Chain Cubature. 
A very basic understanding of stochastic calculus is assumed, but as no proofs are presented 
here, it should be accessible to non Mathematicians.

Code examples will be presented where appropriate, but the primary goal here is not to 
demonstrate operation of MCCube, but to instead explain its underpinning concepts.

:::{warning}
This document is currently incomplete and is yet to be reviewed for accuracy and clarity.
:::

## Markov Chain Cubature
Before delving into the details, it is prudent to precisely define and identify the core 
underpinning components of Markov Chain Cubature. 

Markov Chain Cubature (MCC) is a technique for approximately (weakly) solving stochastic 
differential equations (SDEs) via Cubature on Wiener Space, where the so-called cubature
paths are constructed as a set of discrete Markov Chains. In a moderate number of 
dimensions, MCC should provide far greater accuracy with far fewer chains than an 
alternative Markov Chain Monte Carlo approximation of the (weak) solution to the SDE.

The remainder of this article will proceed as follows:
1. Define, explain, and demonstrate how and why quadrature and cubature formulae are used to 
integrate certain smooth functions.
2. Explain and demonstrate how Cubature on Wiener Space {cite:p}`lyons2004` extends 
these concepts to rougher functions (Wiener Space).
3. Highlight the limitations of conventional construction techniques for Cubatures on 
Wiener Space.

## Quadrature and Cubature

Quadratures, and cubatures, are formulae for numerically integrating functions over 
weighted $n\text{-dimensional}$ regions. The cubature nomenclature implies that the 
integration region is expected to be of dimension $n \ge 2$, while quadrature implies 
dimension $n=1 $; in practice, many authors ignore the distinction and simply refer to 
any numerical integration formulae of the form

$$
Q \left[f\right] := \sum_{i=1}^{k} B_i f(v_i) \approx \int_{\Omega} w(\boldsymbol{x}) f(\boldsymbol{x}) \operatorname{d}\!\boldsymbol{x},
$$

as a quadrature/cubature formulae $Q$, where $B_i \in \mathbb{R}$ and $v_i \in \mathbb{R^d}$ are formula specific 
coefficients and vectors; $w:\mathbb{R}^d \to \mathbb{R}$ is a weight function/distribution; 
and $f:\mathbb{R}^d \to \mathbb{R}$ is the function to integrate/the integrand.

Such formulae $Q$ are said to be of degree $m$, with respect to a specific weight 
$w(\boldsymbol{x})$ and integration region $\Omega \subseteq \mathbb{R}^d$, if they exactly 
integrate all $f \in \mathcal{P}^m(\Omega)$ (polynomials of degree at least $m$ over the specified region $\Omega$). That is to say, degree $m$ formulae $Q^m$ are exact for 
$f(\boldsymbol{x}) = \sum_{\alpha \in \mathcal{A}_m} c_{\alpha} \boldsymbol{x}^\alpha$
where $\boldsymbol{x}^{\alpha} = \prod_{j=1}^d x_j^{\alpha_j}$ are monomials and 
$c_\alpha \in \mathbb{R}$ are coefficients, for all multi-indexes $\alpha \in \mathcal{A}_m := \{(\alpha_1, \dots, \alpha_d) \in \mathbb{N}_0, \sum_{i=1}^d \alpha_i \le m\}$.

:::{admonition} What exactly is a polynomial?
:class: important
To aid in the simplicity of presentation, a somewhat loose definition of a polynomial 
was given above. A more precise definition is as follows: a (multi-variate) polynomial 
is an element of a polynomial ring $K[x_1,\dots,x_d]$ in the indeterminates $\{x_1, \dots, x_d\}$, 
over the field $K$. 

In the parlance of vectors spaces, a polynomial is an element $p$ of the finite 
dimensional $K\text{-vector}$ space $\mathcal{P}(\Omega)$, where $\{x_1, \dots, x_n\}$ 
forms a basis of the $K\text{-vector}$ space $\Omega$. One is typically interested in a 
subspace of degree $m$ polynomials $\mathcal{P}^m(\Omega)$. In this case, the monomial 
basis of $\mathcal{P}^m(\Omega)$ is canonically $\{x \mid \prod_{j=1}^d x_j^{\alpha_j}\}$ 
where the multi-indexes $\alpha \in \mathcal{A}_m := \{(\alpha_1, \dots, \alpha_d) \in \mathbb{N}_0, \sum_{i=1}^d \alpha_i \le m\}$.

The crucial thing to identify is that one can change the monomial basis, to say the 
trigonometric monomials (in the univariate case this is $\{1, \sin(x), \cos(x), \sin(2x), \cos(2x), \dots\}$), 
and providing a new suitable notion of the degree $m$ can be defined one may say 
these polynomials are defined in the trigonometric sense, rather than the algebraic sense.
As will be discussed later, this property is fundamental to Cubature on Wiener Space 
(the central building block of MCC), where the monomial basis is represented by the 
iterated integrals that arise in the Stratanovich stochastic Taylor expansion of a 
stochastic process.

An important point for clarification is that quadrature/cubature formulae are valid 
only for polynomials defined in a given monomial basis. I.E. quadrature/cubature 
formulae for standard algebraic polynomials need not be exact for trigonometric 
polynomials.
:::

:::{admonition} A measure theoretic definition.
:class: note

For those familiar with Lebesgue-Steiltjies integration, the above can be more precisely denoted as

$$
Q \left[f\right] := \int_\Omega f(\boldsymbol{x})\ \operatorname{d}\!\hat{w}(x) \approx \int_\Omega f(x)\ \operatorname{d}\!w(x),
$$

where $f: \Omega \to \mathbb{R} \in L^1(\Omega, \mathcal{B}, w)$, and $\hat{w} = \sum_{i=1}^k B_i \delta_{x_i}$
is the quadrature/cubature measure. 
If integration with respect to the cubature measure $\hat{w}$ is equivalent to integration 
against the target measure $w$ for all $f \in \mathcal{P}^m(\Omega)$, the formula is of 
degree $m$ (denoted $Q^m$ as above).
Note that when $\operatorname{card}(\operatorname{supp}(\hat{w})) < \operatorname{card}(\operatorname{supp}(w))$, 
one may interpret $\hat{w}$ as a *compression* of $w$ {cite:p}`satoshi2021`.
:::

### Integrating non-polynomial functions

While integrating polynomials is an important problem, in many practical cases one will
not have the luxury of dealing with such nice analytic functions. Thus, to be practical,
one must consider how the error of the integration formulae scales for more general 
$f$.

The really neat thing to realize is that any formulae $Q^m$ will be a *good* integrator
for *any* continuous $f$ that can be well approximated by some $\hat{f} \in \mathcal{P}^{m}(\Omega)$,
such as the degree $m$ (truncated) Taylor polynomial of $f$ about $\boldsymbol{x}_0$, given by

$$
T^m_{f(\boldsymbol{x}_0)}(x) = \sum_{k=0}^{m} f^{(k)}(\boldsymbol{x}_0) \frac{(\boldsymbol{x}-\boldsymbol{x}_0)^k}{k!},
$$

where $f(x) = T^m_{f(\boldsymbol{x}_0)}(\boldsymbol{x}) + r^m_{f(\boldsymbol{x}_0)}(\boldsymbol{x})$ and $r^m_{f(\boldsymbol{x}_0)}(\boldsymbol{x})$ is the Taylor remainder.
In practice, such a truncated polynomial will only be a good approximation in sufficiently small neighborhoods of $\boldsymbol{x}_0$.

When $f \in C^{m+1}(\Omega)$, one can show via the Peano kernel theorem that the 
formulae has an error bounded by 

$$
\left|\int_{\Omega}  w(x) f(x)\  \operatorname{d}\!x - Q^m[f] \right| \le c\ \max_{x \in \Omega} \left| f^{(m+1)}(x)\right|,
$$

where the constant $c > 0$ is independent of $f$ {cite:p}`iserles2008` (note that the univariate 
case is shown for simplicty). This simple error bound is a result of $Q^m$, by definition, 
being an exact integrator of the degree $m$ Taylor polynomial $T^m_{f(\boldsymbol{x}_0)}(x)$ about any 
point in the domain.

In the remainder of this article, only exact formulae are considered. Do not worry if 
the above error bound is confusing, the key takeaway is that $Q^m$ can still be useful 
for non-polynomial integrands. See {cite:p}`stroud1971` for an in depth consideration 
of the error analysis and other aspects of *approximate* cubature formulae.

### Constructing formulae

How exactly one constructs the coefficients $B_i$ and vectors $v_i$ that define a given formula $Q$, 
the existence of which is guaranteed by Tchakaloff (see {cite:t}`davis1967` and {cite:t}`mysovskikh1975`),
is a large topic in itself and will not be discussed in depth here (interested readers 
are referred to {cite:p}`cools1997` and {cite:p}`stroud1971`). 

**TODO: Give a brief outline of the construction problem in terms of dual spaces and 
the solution of systems of equations.**

However, it is important to be aware of what makes a *good* construction/formula. That is, a 
construction with the fewest vectors $v_i$ to produce a formula $Q^m$ whose approximation 
error for polynomials of degree $> m$ is minimal. Such a construction can sometimes be 
further enhanced if the vectors are sparse or have other nice properties that can be 
leveraged to accelerate computation. 

**TODO: Give examples of classes of formulae (Newton-Cotes, Gauss, etc...) and product and non-product rules etc...
and where they should be used**

### An example cubature

The notation used above is quite dense and on a first reading may be challenging to follow. 
However, it is just a rigorous presentation of a rather simple concept; multivariate 
polynomials can be exactly integrated, over weighted regions, by weighted sums of the 
polynomial evaluated at specially selected points.

To be concrete, consider the degree three Gaussian cubature formula from 
{cite:t}`stroudSecrest1963` ($E_n^{r^2} \text{3-1}$ in {cite:p}`stroud1971`), which 
exactly solves the following integral, for polynomial $f: \mathbb{R}^d \to \mathbb{R}$ 
of degree $m \le 3$,

$$\int_{\mathbb{R}^d} f(\boldsymbol{x}) \operatorname{d}\!P(\boldsymbol{x}) = \frac{1}{Z_c} 
\int\dotsi\int_{\mathbb{R}^d} f(x_1, \dots, x_d)\exp(-x_1^2 \dots -x_d^2) \operatorname{d}\!x_1 \dots  \operatorname{d}\!x_d$$

where $P(x)$ is the probability measure of the $d\text{-dimensional}$ Gaussian 
distribution $X \sim \mathcal{N}(\boldsymbol{0}, \text{diag}(1/2))$ and $Z_c$ is a 
normalizing constant. If $Z_c$ is known, the above integral is the distribution's 
expectation $\operatorname{E}\left[f(X)\right]$; when $f(X) = X^\alpha$ (a monomial) the
integral is the distribution's moment of degree $\sum_{i=1}^d \alpha_i$. Hence, the 
cubature formula can be used to compute all moments of $X$ of degree less than or equal 
to three. It is also exact for certain degree four moments, such as $E[X_i^2X_j^2]$.

Thus, consider the scenario where $d=2$ (two-dimensional Gaussian) and one wishes to 
compute the co-variance matrix $\operatorname{E}\left[X_iX_j\right]$ (moments of degree 
two) - using the above cubature formula. To perform the required computation one must:


1. **Look up the formula's vectors and coefficients:**<br>
    For $E_n^{r^2}$ they are as follows, where $r^2 = d/2$, $V=\pi^{d/2}$ is the 
    unnormalized volume of the weighted integration region, and each row of the 
    matrix $v$ is a cubature vector $v_i$.

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
coefficients, this cubature formula $Q$ can be adapted to any parametrization of the above 
Gaussian distribution/weight $Q^{*}$ (see pg 11 of {cite:p}`stroud1971` for details). 
Such a transform can be trivially applied in {mod}`mccube.formulae`:

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

Note that while the quadrature formula is in principle *exact*, due to practical limitations of finite-precision numerics, the cubature has an error of $5\times 10^{-7}$. While not critical in this example, there are certain cases where such imprecision can be very significant (such as the integration of high order moments). 
Thus, it is crucial to be cognizant of such issues when developing quadrature/cubature libraries or using them in practice.

Hopefully, this section has elucidated the core operation and utility of quadrature/cubature formulae for integrating "nice" continuous functions over weighted regions.
The presentation is, however, delibratley breif on the issue of formula construction and on intuting/prooving why such formulae work.
Such issues are better treated by dedicated texts, for which the reader is directed to {cite:p}`stroud1971`, {cite:p}`hildebrand1987`, {cite:p}`lanczos1988`, {cite:p}`cools1997`.

+++

# Cubature on Wiener Space
In the above section, it is stated that a quadrature/cubature formulae $Q^m_\Omega$ 
exactly integrates all *polynomials* $p \in \mathcal{P}^m(\Omega)$, where a polynomial 
is taken in the usual *algebraic* sense. However, it is possible to construct formulae 
for polynomials defined in a different sense - perhaps most notably the trigonometric 
polynomials. The fundamental result of Cubature on Wiener Space is that one can go a step 
further and construct formulae for *non-commutative* polynomials of degree $m$ (elements 
of a truncated free tensor algebra $\tau \in T^{(m)}(\Omega)$), and that the iterated 
integrals that appear in the Stratanovich stochastic Taylor expansion can be treated 
as the non-commutative monomial basis.

The benefit of such a result is that thanks to Chen, and Lyon's rough path theory, certain 
rough functions/paths can be represented as elements of the tensor algebra $T(\Omega)$,
and their *rough* Taylor expansions can be represented as elements of the truncated 
tensor algebra $T^{m}(\Omega)$. That is to say, one can construct quadrature/cubature 
formulae that integrate degree $m$ (non-commutative) rough Taylor polynomials.

:::{admonition} Non-commutative polynomials
:class: note
Loosely speaking, one can treat a free tensor algebra as defining the rules for the 
algebraic manipulation of non-commutative polynomials. That is, polynomials where the 
indeterminates do not commute. For example, given $\mathcal{T}(\mathbb{R}^2)$, and a 
standard monomial basis $(1,x,y,y^2,x^2,\dots)$, the non-commutative polynomials will 
satisfy the relation $x^iy^j \nRightarrow y^jx^i$ (for standard commutative polynomials 
$x^iy^j \Rightarrow y^jx^i$).
:::

This is all admittedly quite abstract. An immediate question may be to ask why exactly 
one would come across *non-commutative* polynomials that need to be integrated. 
This is where stochastic differential equations (SDEs) come in.

## Stochastic Differential Equations (SDEs)
<!-- Is there something special about Wiener Space, or can one construct cubature for 
any truncated free tensor algebras? JF: while one could indeed create a cubature for 
other truncated free tensor algebras, unless it has nice properties, such as being 
Markovian, there is not generally any reason to expect it to be possible to extend the 
concepts discussed here. Include this as an aside.-->
**TODO: outline this section**

+++

## ODE Quadrature
One may consider extending quadrature formulae, as described above, to the solution of 
systems of ordinary differential equations (ODEs); all ODEs can be represented as 
first-order systems of the form:

$$
\frac{\operatorname{d}\!\boldsymbol{f}(t)}{\operatorname{d}\!t} = \boldsymbol{g}(t, \boldsymbol{f}(t)),\quad t \ge t_0,\quad \boldsymbol{f}(t_0) = \boldsymbol{f}_0,
$$

where $\boldsymbol{g}\colon [t_0, \infty) \times \mathbb{R}^d \to \mathbb{R^d}$ is at 
least Lipschitz continuous, $\boldsymbol{f}\colon [t_0, \infty) \to \mathbb{R}^d$ is the
*solution*, and $\boldsymbol{f}_0 \in \mathbb{R}^d$ is a given *initial condition*. 
Using the fundamental theorem of calculus, one may attempt to solve this ODE with the 
following equation:

$$
\boldsymbol{f}(t) = \boldsymbol{f}(t_0) + \int_{t_0}^{t} \boldsymbol{g}(\tau, \boldsymbol{f}(\tau))\operatorname{d}\!\tau = \boldsymbol{f}(t_0) + h \int_{0}^{1} \boldsymbol{g}(t_0 + h\tau, \boldsymbol{f}(t_0 + h \tau))\operatorname{d}\!\tau.
$$

If $\boldsymbol{g}$ is analytic, then solving this equation is relatively straightforward. 
However, for many cases of practical interest, such nicities rarely exist, and it may only 
be possible to solve the above via numerical integration.
One may envision using quadrature for this purpose, as shown in the below itterative 
"pseudo-method":

$$
\boldsymbol{f}(t_{n+1}) \approx \boldsymbol{f}(t_n) + h_n Q[\boldsymbol{g}] = \boldsymbol{f}(t_n) + h_n \sum_{i=1}^k B_{i}\ \boldsymbol{g}(t_n + h_n v_i, \boldsymbol{f}(t_n + h_n v_i)), \quad n = 0,1,\dots,N
$$

where $Q$ is a quadrature formulae over the unit interval, and $h_n$ is the length of 
each (potentially non-uniform) region $[t_n, t_{n+1}]$. Unfortunately, because 
$\boldsymbol{f}(t)$ is unknown, it is not possible to evaluate $\boldsymbol{f}$ at the 
quadrature vectors $v_{i}$ - if $f(t)$ was known there would be no need to solve the ODE
in the first place. Thus, in order to make practical use of quadrature formulae for 
solving general ODEs, one must devise some means to approximate the solution at all 
$\boldsymbol{f}(t_n + h_n v_i)$. Such a problem is handled by the explicit and implicit 
Runge-Kutta methods. 

In the explicit approach, each $\boldsymbol{f}(t_n + h_n v_{i})$ is approximated 
sequentially as $\xi_{j}$, where

$$
\boldsymbol{\xi}_{j} = \boldsymbol{f}(t_n) + \sum_{i=1}^{j-1} A_{j,i}\ \boldsymbol{g}(t_n + h_n v_i, \boldsymbol{\xi_{i}}) \quad i = 1, \dots, \nu,
$$

and $\boldsymbol{A}$ is a lower triangular matrix of coefficients (called the RK matrix), and 
$\nu$ is the number of Runge-Kutta stages. The quadature vectors $\boldsymbol{v}$ may now be 
refered to as RK nodes, and the weights $\boldsymbol{B}$ as RK weights. 
These values can be organised into a so-called Butcher tableaux:

$$
\begin{array}{c|c}
\boldsymbol{v} & \boldsymbol{A} \\
\hline
& \boldsymbol{B^T}
\end{array}
$$

The entries in this tableaux are selected such that the Taylor polynomial for the RK 
approximation is identical to the truncated Taylor polynomial of the true solution up to
some desired degree $m$. Salient examples of the explicit Runge-Kutta methods include:
the (forward) Euler method of first order (ERK1), Heun's method of second order (ERK2), 
the "classic" Runge-Kutta method of third order (RK3), and the best "classic" method of 
fourth order (ERK4):

$$
\text{ERK1} \colon \quad
\begin{array}{c|c}
0 & 0\\
\hline & 1
\end{array}
\qquad
\text{ERK2} \colon \quad
\begin{array}{c|cc}
0 & & \\
1 & 1 & \\
\hline & \frac{1}{2} & \frac{1}{2}
\end{array}
\qquad
\text{ERK3} \colon \quad
\begin{array}{c|ccc}
0 & & & \\
\frac{1}{2} & \frac{1}{2} & & \\
1 & -1 & 2 & \\
\hline & \frac{1}{6} & \frac{2}{3} & \frac{1}{6}
\end{array}
\qquad
\text{ERK4} \colon \quad
\begin{array}{c|cccc}
0 & & & & \\
\frac{1}{2} & \frac{1}{2} & & & \\
\frac{1}{2} & 0 & \frac{1}{2} & & \\
1 & 0 & 0 & 1 & \\
\hline & \frac{1}{6} & \frac{1}{3} & \frac{1}{3} & \frac{1}{6}
\end{array}
$$

By inspection of the RK nodes and RK weights in the above examples, it can be seen that
in cases where $\boldsymbol{g}$ is independant of $\boldsymbol{f}$, the above methods 
are equivalent to Newton-Cotes quadratures (EK2 is equivalent to the Trapezoidal rule; 
ERK3 and ERK4 are equivalent to Simpson's rule). Hence, one can say that these specific 
ERK methods extend/generalize quadrature formulae to the solution of ODEs.

note that the degree of the quadrature formulae need not be the same as the degree of the ERK method.

Then explain how IRK methods and collocation and quadrature relate, then explain how it 
extends in a different way.

- yield backward Euler IRK1.
- midpoint IRK2 or Trapezoidal IRK2.
