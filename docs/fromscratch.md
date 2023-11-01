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

:::{warning}
This document is currently **🚧 under construction 🚧** incomplete and is yet to be 
reviewed for accuracy and clarity.
:::

## Quadrature and Cubature
Formulae for numerically integrating functions over (weighted) $d\text{-dimensional}$ 
regions $\Omega$, may be referred to as quadratures (when $d=1$) or cubatures (when $d \ge 2$) 
if they take the form

$$
Q \left[f\right] := \sum_{j=1}^{n} \lambda_j f(\boldsymbol{v}_j) \approx \int_{\Omega} w(\boldsymbol{x}) f(\boldsymbol{x}) \operatorname{d}\!\boldsymbol{x},
$$

where $f \colon \Omega \to \mathbb{R}$ is the integrand and $w \colon \Omega \to \mathbb{R}$ 
is the weighting function/distribution. Such a formulae is said to be of degree $m$,
with respect to a specific weight $w$ and region $\Omega$, if it exactly integrates all
(multi-variate) polynomials $f \in \mathcal{P}^m(\Omega)$ of degree $\le m$. The practical 
difficulty of constructing such formulae $Q^m \colon \mathcal{P}^m(\Omega) \to \mathbb{R}$ 
lies in identifying the smallest number, $n \in \mathbb{N}^+$, of coefficients $\lambda_j \in \mathbb{R}$ 
and vectors $\boldsymbol{v}_j \in \Omega$ such that the approximation above is replaced 
with equality.

:::{admonition} What exactly is a polynomial?
:class: important

A (multi-variate) polynomial is an element of a polynomial ring $K[x_1,\dots,x_d]$ in 
the indeterminates $\{x_1,\dots,x_d\}$, over the field $K$. In the parlance of vector
spaces, a polynomial is an element $f$ of the finite dimensional $K\text{-vector}$ space 
$\mathcal{P}^m(\Omega)$, with basis $p:=\{p_1, \dots, p_k\}$, where $(x_1, \dots, x_d) \in \Omega$ 
is an element (representing a value for the indeterminates), and $\operatorname{dim}(\Omega) = d$.
That is to say, each "polynomial" $f:\Omega \to \mathbb{R}$ is a linear combination of 
the following form, where $a_i \in K$,

$$
f(x_1, \dots, x_d) = \sum_{i=1}^{k} a_i p_i(x_1, \dots, x_d).
$$

Canonically, the field $K$ is the reals $\mathbb{R}$ and the basis of 
$\mathcal{P}^m(\Omega)$ is $p=\{(x_1, \dots, x_d) \mapsto \prod_{j=1}^d x_j^{\alpha} \mid \alpha \in \mathcal{A}_m\}$, 
for the multi-indices $\mathcal{A}_m := \{(\alpha_1, \dots, \alpha_d) \in \mathbb{N}_0, \sum_{j=1}^d \alpha_i \le m\}$.
As a concrete example, consider the case where $d=1$ and $m=2$, such that $f \in \mathcal{P}^2(\mathbb{R})$, 
and $x_1 \in \mathbb{R}$. The standard monomial basis simplifies to $p=\{x_1 \mapsto 1, x_1 \mapsto x_1, x_1 \mapsto x_1^2\}$, 
and each $f$ can be represented as the following linear combination, where $a_i \in \mathbb{R}$, 
and $n := \operatorname{card}(p) = 3$,

$$
\begin{align*}
f(x_1) &= \sum_{j=1}^{3} a_j p_j(x_1),\\
       &= a_1 p_1(x_1) + a_2 p_2(x_1) + a_3 p_3(x_1),\\
       &= a_1 + a_2 x_1 + a_3 x_1^2.
\end{align*}
$$

If one were to change the standard monomial basis to the trigonometric monomials 
(for the example above this is $p=\{x_1 \mapsto 1, x_1 \mapsto \sin(x_1), x_1 \mapsto \cos(x_1),  x_1 \mapsto \sin(2x_1), x_1 \mapsto \cos(2x_1)\}$), 
providing a new suitable notion of the degree $m$ can be defined, one may say 
these polynomials are defined in an alternative (trigonometric) sense. Hence, when one 
refers to a degree $m$ quadrature/cubature, this degree is with respect to polynomials 
in a specific sense/basis.

As will be discussed later, this ability to generalize polynomials to an alternative basis
is fundamental to Cubature on Wiener Space (the central building block of MCC), where 
the monomial basis is represented by the iterated integrals that arise in the Stratanovich 
stochastic Taylor expansion of a stochastic process.
:::

For those familiar with Lebesgue-Steiltjies integration, a nicer (and more general) 
notation for the definition of a quadrature/cubature is any formulae,

$$
Q \left[f\right] := \int_\Omega f(\boldsymbol{x})\ \operatorname{d}\!\hat{w}(x) \approx \int_\Omega f(x)\ \operatorname{d}\!w(x),
$$

where $f: \Omega \to \mathbb{R} \in L^1(\Omega, \mathcal{B}, w)$, and $\hat{w} = \sum_{j=1}^n \lambda_j \delta_{\boldsymbol{v}_j}$
is the quadrature/cubature measure. 
If integration with respect to the cubature measure $\hat{w}$ is equivalent to integration 
against the target measure $w$ for all $f \in \mathcal{P}^m(\Omega)$, the formula is of 
degree $m$ (denoted $Q^m$ as above).
Note that when $\operatorname{card}(\operatorname{supp}(\hat{w})) < \operatorname{card}(\operatorname{supp}(w))$, 
one may interpret $\hat{w}$ as a *compression* of $w$ {cite:p}`satoshi2021`.

Additional important aspects of quadrature/cubature are very briefly discussed in the
proceeding subsections. Due to the brevity of presentation, further interested readers 
are pointed to the dedicated works of {cite:t}`stroud1971`, {cite:t}`rabinowitz1984`,
{cite:t}`cools1997`, and {cite:t}`brass2011`.

### Constructing formulae
A formulae $Q^m$ defines a space of linear functionals that are dual to the space of 
polynomials $f \in \mathcal{P}^m(\Omega)$, with the condition that all functionals 
$Q^m[f] = \int_\Omega w(\boldsymbol{x}) f(\boldsymbol{x}) \operatorname{d}\!\boldsymbol{x}$. 
To construct a formulae is to find a basis $q = \{p_i \mapsto \sum_{j=1}^n \lambda_j p_i(\boldsymbol{v}_j) \mid i=1, \dots, k\}$,
where the integral condition is exactly satisfied for each element of the basis $q_i$.
The coefficients $\lambda_j$ and vectors $\boldsymbol{v}_j$ can be found by attempting 
to solve the following system of integral equations (other approaches are detailed in {cite:p}`cools1997`),

$$
q_i = \sum_{j=1}^n \lambda_j p_i(\boldsymbol{v}_j) = \int_\Omega w(\boldsymbol{x})p_i(\boldsymbol{x}) \operatorname{d}\!\boldsymbol{x}, \quad \text{for}\ i=1, \dots, k.
$$

For some $n$ (and arbitrary $m$), at least one solution/construction is guaranteed 
due to Tchakaloff (see {cite:p}`davis1967` and {cite:p}`mysovskikh1975`). Typically, 
one is interested in constructions where $n$ is minimal - minimizing the computational 
cost of formula evaluation.  

:::{admonition} What is the "best" construction?
:class: note

How one defines the "best" construction will depend on the desired application (class 
of functions to integrate), the available computational resources (CPU/GPU/TPU), and 
ones ability to control the points in the domain at which the integrand is sampled. 
However, when considering the optimality of a construction, it is prudent to keep the 
following properties in mind:

1. **Computational efficiency:** does the construction have properties that can be 
leveraged for accelerating computation such as sparse vectors, a minimal number of 
vectors, etc...?
2. **Numerical error:** when applied to non-polynomial functions (where the formulae are 
inexact) does the construction minimize the integration error?
3. **Control over sampling locations**: can you sample/evaluate the function at arbitrary 
locations, or are the outputs of the function to integrate already tabulated?

The issue of numerical error, ignoring precision loss, only arises if one whishes to 
integrate functions $f \notin \mathcal{P}^m(\Omega)$. Such a use case is very common 
and is discussed further in the next section.
:::

:::{admonition} Product and non-product constructions
:class: important

If one has constructed a formulae for the unit-interval, they can trivially construct a
formulae for any region that is the product of the unit-interval, such as the unit cube.
Formulae constructed in such a manner are called product formulae. The biggest downside 
of such formulae is that they are rarely minimal. For example, given a formulae for the 
unit interval with $n$ elements, the product formulae for the cube will have $n^3$ elements.

As will be shown later, the difference between product and non-product formulae is 
somewhat analogous to the difference between Cubature on Wiener space and Markov Chain 
Cubature.
:::

The above, while succinct, is also rather abstract. Thus, to aid in understanding, consider 
the concrete example of the univariate real polynomials of degree two 
$f \in \mathcal{P}^2(\mathbb{R})$ with the standard monomial basis 
$p = \{x_1 \mapsto 1, x_1 \mapsto x_1, x_1 \mapsto x_1^2\}$, where each polynomial $f$ 
can be represented in the following form, with $a_i \in \mathbb{R}$,

$$
\begin{align*}
f(x_1) &= a_1 p_1(x_1) + a_2 p_2(x_1) + a_3 p_3(x_1),\\
       &= a_1 + a_2 x_1 + a_3 x_1^2.
\end{align*}
$$

The weighted integral of any such polynomial $f(x_1)$ can be expressed as follows (due 
to the linearity of the integral operator and the expansion of $f$), 

$$
\int_\mathbb{R} w(x_1)f(x_1)\operatorname{d}\!x_1 = \sum_{i=1}^{3} a_i \int_\mathbb{R} w(x_1) p_i(x_1) \operatorname{d}\!x_1 = \sum_{i=1}^{3} a_i q_i,
$$

The right hand side follows from the definition of the quadrature construction. One 
need not know the values of the coefficients $a_j$, providing one is able to evaluate 
the function $f$, as shown below,

$$
\begin{align*}
Q^m[f] = \sum_{j=1}^{n} \lambda_j f(v_j) &= \sum_{j=1}^n \lambda_j \sum_{i=1}^{3} a_i p_i(v_j), \quad \forall f \in \mathcal{P}^2(\mathbb{R})\\
                                         &= \sum_{i=1}^{3} a_i \sum_{i=1}^k \lambda_i p_i(v_j),\\
                                         &= \sum_{i=1}^{3} a_i q_i.
\end{align*}
$$

The practical implication of the above is that one can implement a program which accepts
an arbitrary (potentially non-polynomial) function $f\colon \Omega \to \mathbb{R}$, 
apply the formulae to $f$, and if $f \in \mathcal{P}^m(\Omega)$, the result
will still be exact (ignoring any numerical precision loss).

### Integrating non-polynomial functions

In many practical cases, one is interested in integrating a much wider class of 
functions than the polynomials $f \in \mathcal{P}^m(\Omega)$. In such a case, the formulae 
will be inexact. Understanding how to minimize this inexactitude is crucial in selecting 
the best construction of formulae in practice.

One may assume that the most accurate formulae will always be of maximal degree $m$, 
given a fixed number of coefficients and vectors $n$. Using such logic, one may come to 
the conclusion that [Gauss type quadratures](https://en.wikipedia.org/wiki/Gaussian_quadrature) 
with degree $m=2n-1$ will always be optimal when one can choose the points at which to 
evaluate the integrand.

However, as shown by {cite:t}`Trefethen2022`, in practice this assumption is not always
true, with [Clenshaw-Curtis](https://en.wikipedia.org/wiki/Clenshaw%E2%80%93Curtis_quadrature),
composite [Newton-Cotes](https://en.wikipedia.org/wiki/Newton%E2%80%93Cotes_formulas), 
and other lower degree formulae (for a fixed number of elements $n$) in some cases 
showing better convergence rates than Gauss type formulae. This discrepancy arises 
from the fact that exactitude, while useful in algebraic arguments, is not the quantity 
of interest in applied cases. Instead, one is concerned with the analytical arguments 
of closeness (under some metric) of the formulae to the true solution. Thus, 
{cite:t}`Trefethen2022` makes the argument that one should take care to consider the 
nuances of the specific class of non-polynomial functions one is wishing to integrate 
before accepting a formulae as optimal.

### An example cubature

Consider the degree three Gaussian cubature formula from {cite:t}`stroudSecrest1963` 
($E_n^{r^2} \text{3-1}$ in {cite:p}`stroud1971`), which exactly solves the following 
integral, for polynomial $f: \mathbb{R}^d \to \mathbb{R}$ of degree $m \le 3$,

$$\int_{\mathbb{R}^d} f(\boldsymbol{x}) \operatorname{d}\!P(\boldsymbol{x}) = \frac{1}{Z_c} 
\int\dotsi\int_{\mathbb{R}^d} f(x_1, \dots, x_d)\exp(-x_1^2 \dots -x_d^2) \operatorname{d}\!x_1 \dots  \operatorname{d}\!x_d$$

where $P$ is the probability measure of the $d\text{-dimensional}$ Gaussian 
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
    \lambda_i = \frac{V}{2d}, \quad \boldsymbol{v}=\begin{bmatrix}
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

    $$\frac{1}{Z_c}\sum_{i=1}^{k} \lambda_i f(\boldsymbol{v}_i).$$


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
lambda_i = cf.coefficients
v_i = cf.vectors

# Step 2. Compute the normalization constant
z_c = cf.region.volume

# Step 3a. Evaluate the formula (Manual)
f_cov = lambda x: jnp.einsum("i, j -> ij", x, x) - x  # 2nd Central Moment
eval_vmap = jax.vmap(lambda v: lambda_i * f_cov(v), [0])
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

Note that while the quadrature formula is in principle *exact*, due to practical 
limitations of finite-precision numerics, the cubature has an error of $5\times 10^{-7}$. 
While not critical in this example, there are certain cases where such imprecision can 
be very significant (such as the integration of high order moments). 
Thus, it is crucial to be cognizant of such issues when developing quadrature/cubature 
libraries or using them in practice.

This concludes the section on conventional quadrature/cubature. Further interested 
readers are again direct to the dedicated texts of {cite:t}`stroud1971`, 
{cite:t}`rabinowitz1984`, {cite:t}`cools1997`, and {cite:p}`brass2011`. In addition to 
the above dedicated works, one may find the appropriate sections in the (non-dedicated) 
works of {cite:t}`hildebrand1987` and {cite:t}`lanczos1988` helpful in the pursuit of a
complete comprehension of quadrature/cubature formulae.

+++

## Stochastic Differential Equations
Stochastic Differential Equations (SDEs) are stochastic analogues of ODEs. As such, 
they are useful for modelling systems that evolve (usually in time) and are subject to 
stochasticity due to a lack of information about the system (epistemic uncertainty) 
and/or a fundamental mechanism driving the system (aleatoric uncertainty). 

SDEs can be used to model a very wide variety of systems. A few examples from 
{cite:t}`kloeden1992` include but are not limited to: population dynamics, protein kinetics, 
genetics, experimental psychology, investment finance and option pricing, turbulent diffusion, 
satellite orbit stability, hydrology, seismology, fatigue cracking and optical bistability. 
In addition any system modelled via a parabolic PDE can be equivalently modelled via an 
SDE (thanks to the [Feynman-Kac formula](https://en.wikipedia.org/wiki/Feynman%E2%80%93Kac_formula)).

:::{warning}
Readers with no prior knowledge of stochastic processes and stochastic calculus may 
struggle to follow the remainder of this section. For those readers who are interested 
in understanding the details, the following lecture notes and textbook may be useful 
references {cite:p}`tehranchi2016`, {cite:p}`Siegrist2023`, {cite:p}`kloeden1992`.

For all other readers, only the following key concept must be understood: the (weak) solution 
of an SDE (a stochastic process) can be approximated by (truncated) stochastic Taylor 
expansion, in much that same way an ODE can be approximated by (truncated) Taylor expansion.
:::

### Stratonovich SDEs
The general form of a (Stratonovich) SDE (driven by a $d\text{-dimensional}$ Brownian motion)
is given in the following equivalent differential and integral forms,

$$
\begin{gather*}
\operatorname{d}\!X_{t} = a(X_t) \operatorname{d}\!t + \sum_{i=1}^{d} b^i(X_t) \circ \operatorname{d}\!W^i_t,\\
X_t = X_0 + \int_0^t a(X_s) \operatorname{d}\!s + \int_0^t \sum_{i=1}^d b^i(X_s) \circ \operatorname{d}\!W^i_s,
\end{gather*}
$$

where $a, b^i \colon \mathbb{R^N} \to \mathbb{R}^N$ are drift and diffusion terms (vector 
fields with sufficient smoothness such that the above integrals exist), $X_0$ is the 
initial condition, and $\{W^i_t\}_{t \in [0,T]}$ is the standard $d\text{-dimensional}$ 
Wiener process defined over the probability space $(\Omega, \mathcal{F}, \mathbb{P})$.

:::{admonition} Understanding the Wiener process.
:class: important

A standard $d\text{-dimensional}$ Wiener process is a stochastic process $W = \{W_t\}_{t\in[0,T]}$ with state space $\mathbb{R}^d$ that obeys the following properties:
1. $W_0 = 0$ almost surely,
2. $W$ has stationary and independent increments,
3. $W_t$ is normally distributed with mean zero and variance $t$ for all $t \in (0, T]$,
4. The co-ordinate map $t \mapsto W_t$ is almost surely continuous on $[0,T]$.

Such a process, that obeys the above properties, can be interpreted in two equivalent ways: 
- a collection of random-variables, with index-set $[0,T]$, defined over the common 
(abstract) probability space $(\Omega, \mathcal{F}, \mathbb{P})$, where $\mathcal{F}$ is 
the Borel $\sigma\text{-algebra}$ and $\mathbb{P}$ is the common probability measure, 
that maps to the state space $(\mathbb{R}^d, \Sigma)$ where $\Sigma$ is the Borel 
$\sigma\text{-algebra}$ for $\mathbb{R^d}$ (a standard definition for a stochastic 
process);
- a function-valued random variable defined over the probability space, also known as the
Wiener space, $(C_0^0([0, T], \mathbb{R}^d), \mathcal{F}, \mathbb{P})$, where $C_0^0([0,T], \mathbb{R}^d)$ 
is the space of continuous functions starting at zero that map from the index-set to the 
state space, $\mathcal{F}$ is the Borel $\sigma\text{-field}$ and $\mathbb{P}$ is the 
classical Wiener measure (for the purposes here, a more helpful definition of a stochastic 
process).

To understand the validity of the later interpretation, consider that for all 
$\omega \in C_0^0([0,T], \mathbb{R}^d)$ properties one and four are satisfied and that 
properties two and three can be satisfied by selection of an appropriate measure 
$\mathbb{P}$ (by definition, the classical Wiener measure). Note that in this 
interpretation the measure is often called the *law* of the process. 
:::

Through a bit of notational trickery, the above SDE can be represented 
in the following more compact (non-standard) form, where the new zeroth dimension is 
defined by $b^0 := a$ and $W^0_t(\omega) := t$,

$$
\begin{gather*}
\operatorname{d}\!X_t = \sum_{i=0}^{d} b^i(X_t) \circ \operatorname{d}\!W^i_t,\\
X_t = X_0 + \int_0^t \sum_{i=0}^d b^i(X_s) \circ \operatorname{d}\!W^i_s.
\end{gather*}
$$

:::{admonition} Stratonovich vs Itô SDEs.
:class: note

An SDE can be presented in either Stratonovich or Itô form. The difference between the 
forms lies in how one defines the stochastic integral term. However, the choice between 
one form and another, is in practice, mostly a matter of convenience as one can freely 
convert between the two. The choice of the Stratonovich here is to provide consistency 
with {cite:p}`lyons2004`. 
:::

### Solving an SDE
The *solution* stochastic process $X_t$ is the almost surely defined measurable map from 
the Wiener space into the state space $(\mathbb{R}^N, \mathcal{B}(\mathbb{R}^N))$, where
$\mathcal{B}(\mathbb{R}^N)$ is the Borel $\sigma\text{-algebra}$ on $\mathbb{R}^N$, that 
satisfies the SDE almost everywhere. A solution is called *weak* if it holds for some 
suitable probability space (I.E. any of the many possible equivalent in law/distribution 
Wiener processes); a solution is called *strong* if it holds for some specific given 
probability space (I.E. a given version of the Wiener process). The *weak* solution is 
typically more natural.

In general, one must resort to finding the (weak) solution $X_t$ via numerical means (in 
the absence of any specific closed-form solutions). A natural approach to this problem 
is to approximate the solution via a suitable degree truncated stochastic Taylor 
expansion. A weak form of the (Stratonovich) stochastic Taylor expansion of degree $m$, 
which is consistent with {cite:p}`lyons2004` is presented below,

$$
f(X_t) = \sum_{(i_1, \dots, i_k) \in \mathcal{A_m}} b^{i_1}\dots b^{i_k} f(X_0) \int_{t_0 < t_1 < \dots < t_k < t} \circ \operatorname{d}\!W_{t_1}^{i_1}\dots\circ \operatorname{d}\!W_{t_k}^{i_k} + \mathcal{O}(h^{(m+1)/2}),
$$

where $h = t - t_0$ is the step size, $b^i$ are treated as operators that act on the 
right hand arguments, and $\mathcal{A}_m = \{(i_1, \dots, i_k) \in \{0, \dots, d\}^k, k + \text{card}\{j, i_j=0\} \le m\}$. 
Note that the lower integration limit is simply a convenient notation for the iterated 
integration $\int_{t_0}^{t_1} \int_{t_1}^{t_2}\dots\int_{t_{k-1}}^{t_k} \circ \operatorname{d}\!W_{t_1} \circ \operatorname{d}\!W_{t_2} \dots \circ\operatorname{d}\!W_{t_k}$.

The above notation, while nice and compact, may be confusing on a first reading. To aid 
with comprehension, consider the following example cases where $m=1$ (Euler solver) 
and $m=2$ (Huen solver) and the dimensionality of the Wiener process $d=2$. In the first 
case, the set $A_{1} = \{(1),(2)\}$. This is a result of the condition $k + \text{card}(j, i_j =0) \le m$, 
which eliminates and $k > 1$ and the zero element $(0)$. The resulting expansion is thus,

$$
f(X_t) = b^1 f(X_0) \int_{t_0 < t_1 < t} \circ \operatorname{d}\!W_{t_1}^1 + b^2 f(X_0) \int_{t_0 < t_1 < t} \circ \operatorname{d}\!W_{t_1}^2 + \mathcal{O}(h).
$$

The in second case, where $A_{2} = \mathcal{A}_1 \cup \{(0), (1,1), (1,2), (2,1), (2,2)\}$, 
the resulting expansion is,

$$
\begin{align*}
f(X_t) &= b^0 f(X_0) \int_{t_0 < t_1 < t} \circ \operatorname{d}\!W_{t_1}^0
+ b^1 f(X_0) \int_{t_0 < t_1 < t} \circ \operatorname{d}\!W_{t_1}^1
+ b^2 f(X_0) \int_{t_0 < t_1 < t} \circ \operatorname{d}\!W_{t_1}^2\\
&+ b^1 b^1 f(X_0)\int_{t_0 < t_1  < t_2 < t} \circ \operatorname{d}\!W_{t_1}^1\circ\operatorname{d}\!W_{t_2}^1
+ b^1 b^2 f(X_0)\int_{t_0 < t_1  < t_2 < t} \circ \operatorname{d}\!W_{t_1}^1\circ\operatorname{d}\!W_{t_2}^2\\
&+ b^2 b^1 f(X_0)\int_{t_0 < t_1  < t_2 < t} \circ \operatorname{d}\!W_{t_1}^2\circ\operatorname{d}\!W_{t_2}^1
+ b^2 b^2 f(X_0)\int_{t_0 < t_1  < t_2 < t} \circ \operatorname{d}\!W_{t_1}^2\circ\operatorname{d}\!W_{t_2}^2 + \mathcal{O}(h^{3/2}).
\end{align*}
$$

The above unwieldy equation exemplifies the elegance of the more compact notation used 
to introduce the expansion.

### Stochastic Taylor "polynomials"
The truncated (degree $m$) deterministic Taylor expansion of a smooth function $g$, 
about a point $x \in \Omega$, is a polynomial in $\mathcal{P}^m(\Omega)$ of the form,

$$
\sum_{k \le m} g^{(m)}(x_0) \frac{(x - x_0)^m}{m!}.
$$

The keen eyed reader will note that the stochastic Taylor expansion presented above has 
a very similar form, with the exception that the monomials $(x-x_0)^m$ in the 
deterministic formula have been replaced with the iterated Stratonovich integrals, and 
the coefficients $g^(m)(x_0)$ have been replaced with $b^{i_1}\dots b^{i_k} f(X_0)$.
As such, a natural question is to ask if the truncated stochastic Taylor expansion is 
also a polynomial in $\mathcal{P}^m(\Omega)$?

For now, lets assume that the truncated stochastic Taylor expansion is a polynomial in 
the indeterminates $(W^0, W^{1}, \dots, W^{d})$ span the Wiener space, which takes 
the place of $\Omega$. The monomial basis of the space of stochastic "polynomials" of 
degree $m$ is then the set,

$$
\left\{(W^{0}, W^{1}, \dots W^{d}) \mapsto \int_{t_0 < t_1 < \dots < t_k < t} \circ \operatorname{d}\!W_{t_1}^{i_1}\dots\circ \operatorname{d}\!W_{t_k}^{i_k} \mid (i_1, \dots, i_k) \in \mathcal{A}_m \right\}.
$$

There is, however, a slight problem. Polynomials, in the usual sense, place an implicit 
symmetry condition on the basis functions. For example, it is implied that the basis 
$(x_1,x_2) \mapsto x_1 x_2$ is identical $(x_1,x_2) \mapsto x_2 x_1$. That is to say, 
each basis function is commutative in the indeterminates. This commutativity condition 
is not (in general) met for the stochastic "polynomials". For example, the basis 
$(W^1, W^2) \mapsto \int_{t_0<t_1<t_2<t}\circ \operatorname{d}\!W_{t_1}^1\circ \operatorname{d}\!W_{t_2}^2$
is not identical to $(W^1, W^2) \mapsto \int_{t_0<t_1<t_2<t}\circ \operatorname{d}\!W_{t_1}^2\circ \operatorname{d}\!W_{t_2}^1$.

Thankfully, one can abandon the commutativity requirement and still retain many of the 
useful/intuitive properties of standard polynomials. This is done by generalizing from a 
polynomial algebra to a free tensor algebra. The ability to represent degree $m$ 
non-commutative stochastic Taylor polynomials as elements of the free tensor algebra $\mathcal{T}^m(\Omega)$
is central to enabling cubature on Wiener space. 

This concludes the section on stochastic differential equations. As noted above, this 
section is not intended to be a comprehensive introduction to stochastic calculus or SDEs, 
but simply an overview of the notation and core concepts required to understand Cubature
on Wiener Space, and Markov Chain Cubature. Further interested readers are again directed 
to the notes of {cite:t}`tehranchi2016` and {cite:t}`Siegrist2023`, along with the 
textbook of {cite:t}`kloeden1992`.

## Cubature on Wiener Space
Consider the scenario where one is interested in computing the unique expected value 
$\operatorname{E}(f(X_{t,x}))$ of some smooth function $f$ of a stochastic process $X_{t,x}$,
with the initial condition $X_0 = x$, where $X_{t,x}$ is the strong path-wise solution of 
an SDE defined over the standard Wiener space that coincides with bounded variation paths.
That is to say, one needs to compute a unique in expectation (weak) solution of the SDE 
describing $X_{t,x}$. This exact scenario arises when one is faced with solving a 
parabolic PDE.

:::{admonition} Solving a Parabolic PDE as the expectation of an SDE. 
:class: note
**TODO: 🚧 Under Construction 🚧**
:::

One may attempt to approximate the expected value (weak solution of the SDE) to some 
desired error order $\mathcal{O}(h^{(m+1)/2})$ by exactly computing the expectation (over 
Wiener space) of the degree $m$ stochastic Taylor expansion of $f(X_{t,x})$. To do so,
on requires some formulae/method that can exactly integrate the degree $m$ non-commutative
stochastic Taylor polynomials $\mathcal{T}^m(\Omega)$ generated by the expansion.
Recalling the definition of a cubature (a formulae that exactly integrates all degree 
$m$ polynomials over a given weighted region), it should be apparent that what one needs,
to compute this approximate solution, is a degree $m$ cubature formulae for the 
non-commutative polynomials $\mathcal{T}^m(\Omega)$.

Such a formulae is called a Cubature on Wiener Space if it exactly computes the expectation 
over Wiener space for the set of iterated integrals that form a basis of $\mathcal{T}^m(\Omega)$.
That is, an formulae of the following form, that holds for all $(i_1, \dots, i_k) \in \mathcal{A}_m$,

$$
\operatorname{E}\left[\int_{t_0 < t_1 < \dots < t_k < T} \circ \operatorname{d}\!W_{t_1}^{i_1}\dots\circ\operatorname{d}\!W_{t_k}^{i_k}\right] = \sum_{j=1}^n \lambda_i \int_{0 < t_1 < \dots < t_k < T} \operatorname{d}\!\omega_j^{i_1}(t_1)\dots\operatorname{d}\!\omega_j^{i_k}(t_k),
$$

where $\omega_j \in C_{0, bv}^0([0,T], \mathbb{R^d})$ and $\lambda_j \in \mathbb{R}_+$ 
are the formula specific cubature paths and coefficients respectively. The above can 
alternatively be interpreted in a Lebesgue sense, where the cubature formulae defines
the new measure $\mathbb{Q}^m = \sum_{j=1}^n \lambda_j \delta_{w_j}$, and the above can 
be equivalently denoted as,

$$
\operatorname{E}\left[\int_{t_0 < t_1 < \dots < t_k < T} \circ \operatorname{d}\!W_{t_1}^{i_1}\dots\circ\operatorname{d}\!W_{t_k}^{i_k}\right] = \operatorname{E_{\mathbb{Q}^m}}\left[\int_{t_0 < t_1 < \dots < t_k < T} \circ \operatorname{d}\!W_{t_1}^{i_1}\dots\circ\operatorname{d}\!W_{t_k}^{i_k}\right].
$$

:::{admonition} Cubature on Wiener Space as solving weighted ODEs
:class: important

An important thing to notice is that, for a given path $\omega$, the solution of the SDE 
$X_{t,x}(\omega)$ reduces to an ODE (with respect to the selected path $\omega$), such 
that the following holds,

$$
\operatorname{d}\!X_{t,x}(\omega) = \sum_{j=1}^n b^i (X_{t,x}(\omega))\operatorname{d}\! \omega^i(t), 
$$

where the initial condition $X_{0,x}(\omega)$ is given. Consequently, the expectation 
with respect to the cubature measure (cubature paths), simply reduces to a weighted sum 
of the solution of specially selected ODEs,

$$
\operatorname{E_{\mathbb{Q}^m}}[f(X_{t,x})] = \sum_{j=1}^n \lambda_j f(X_{t,x}(w_j)).
$$
:::

The beauty of the above cubature formulae is that it removes the difficulty of 
computing the expectation of a stochastic process $X_{t,x}$ over an infinite-dimensional 
region, and replaces it with the familiar problem of constructing a cubature formulae, 
that is exact for some finite-dimensional space of "polynomials", and subsequently 
solving a set of $n$ specially selected ODEs.

### Constructing a Cubature on Wiener Space
While the algebraic construction of cubature formulae on Wiener space is significantly 
complicated by the generalization to non-commutative polynomials (see {cite:p}`lyons2004`), 
for the purposes of constructing formulae by solving a system of equations (as described 
in the above section on standard quadrature/cubature construction), the distinction only 
complicates thing in that, for a given degree, the number of equations the formulae must 
exactly satisfy is larger (due to the non-commutative polynomials requiring a larger basis).

### Limitations of Cubature on Wiener Space
There is a significant problem with the cubature approach as currently described. The 
error scales on the order $\mathcal{O}(h^{(m+1)/2})$, which even for large $m$, will 
grow large as the time of interest $T$ moves further away from the initial condition 
$t_0$ (the step size $h = T - t_0$ grows large). The result is that for many time scales 
of interest, the error will grow too large to be of practical utility.

The solution is to subdivide the time domain $[t_0, T]$ into suitably small sub-intervals, 
and to compute the solution step by step, taking the result of the prior step as the 
initial condition for the subsequent step (this time-stepping scheme is standard practice 
when solving ODEs). For example, the time domain $[0, T]$ can be split into the intervals,

$$
0 = t_0 < t_1 < t_2 < \dots < t_M = T,
$$

where the interval sizing may be fixed or adaptive (as is again common in methods for 
solving ODEs) and is defined as $h_l = t_l - t_{l-1}$. The sub-division can be modelled 
as a Markov process, $\{Y_i\}_{0 \le i \le M}$, where $i$ is the step count and the 
process is defined by the transition kernel,

$$
\mathbb{P}(Y_{l+1} = X_{h_l,x}(\omega_j(h_l)) \mid Y_l = x) = \lambda_j.
$$

Then, for any number of steps $M$, presuming one uses a cubature formulae that is valid
for each step $h_l$ (due to the scaling property of Brownian motion, any cubature valid 
over the unit time step can be rescaled for arbitrary step $h_l$ {cite:p}`lyons2004`),

$$
\operatorname{E}[f(X_{T,x})] = \operatorname{E}[f(Y_M) / Y_0 = x] + \mathcal{O}(\sum_{l=1}^M h_l^{(m+1)/2}),
$$

where $h_l$ can be chosen to be arbitrarily small and $m$ as large as feasible for the 
construction of a cubature formulae. 

What may not be immediately clear from the above, is that the number of cubature 
paths and weights required at each step $l$ increases exponentially. For example, given 
an $n$ path single step cubature formulae, repeated application of this formula for $m$
steps will lead to $n^m$ (potentially unique) paths and weights. Intuitively, this can 
be understood from the fact that if one starts with an initial state at $t_0$, the 
cubature formulae implies that at $t_1$ there exist $n$ states which evolve along each
path $\omega_j$ with probability $\lambda_j$. At $t_2$ there then exist $n$ initial states
from which each lead to $n$ states with probability $\lambda_i \lambda_j$, where $\lambda_i$
is the probability of the "initial state" as determined by the previous cubature step. 
Numerically, this can be understood by the following equation,

$$
\begin{align*}
\operatorname{E}[f(Y_k)/Y_0 = x] &= \sum_{j_1=1}^n \dots \sum_{j_M=1}^n \lambda_{j_1} \dots \lambda_{j_M} f(X_{T,x}(\omega_{j_1}(s_1) \otimes \dots \otimes \omega_{j_M}(s_M))),\\
&= \operatorname{E_{\mathbb{Q}^m_T}}(f(X_{t,x})),
\end{align*}
$$

where the measure $\mathbb{Q}^m_T$ is defined as follows,

$$
\mathbb{Q}^m_T = \sum_{j_1=1}^n \dots \sum_{j_M=1}^n \lambda_{j_1} \dots \lambda_{j_M} \delta_{\omega_{j_1}(s_1) \otimes \dots \otimes \omega_{j_M}(s_M)}.
$$

The good news is that the subdivision doesn't change the fundamentals. To solve 
$\operatorname{E(f(X_{t,x}))}$, one need only compute the weighted sum of the solution 
of $n^M$ specially selected ODEs (where the choice of ODEs is determined exactly by the
cubature formula). The bad news is that, at some point, the number of steps $M$ will 
cause the number of ODEs to exploded and exceed computational tractability. 
Ameliorating this problem is the premise of Markov Chain Cubature (MCC).

This concludes the section on Cubature on Wiener Space. Further interested readers are 
pointed to the seminal paper of {cite:t}`lyons2004`. Attempting to write this section in 
a somewhat accessible manner has been a challenge. If you have any suggestions for
simplifying the presentation or find anything confusing please open an issue. 

## Markov Chain Cubature (MCC)
**TODO: 🚧 Under Construction 🚧**

+++

<!-- ## ODE Quadrature
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
- midpoint IRK2 or Trapezoidal IRK2. -->
