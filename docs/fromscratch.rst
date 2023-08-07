Markov chain cubature from scratch
==================================
The goal here is to introduce and explain in a (hopefully) intuitive way, the concepts 
that underpin Markov chain cubature. A very basic understanding of stochastic calculus 
is assumed, but as no proofs are presented here, it should be accessible to non 
Mathematicians.

Code examples will be presented where appropriate, but the primary goal here is not to 
demonstrate operation of MCCube, but to instead explain its underpinning concepts.

Cubature and SDE Cubature
-------------------------
Quadratures, and cubatures, are formulae for numerically integrating functions over 
weighted $n\text{-dimensional}$ regions. The cubature nomenclature implies that the 
integration region is expected to be of dimension $n \ge 2$, while quadrature implies 
dimension $n=1 $; in practice, many authors ignore the distinction and simply refer to 
any numerical integration formulae of the form

$$Q \left[f\right] := \sum_{i=1}^{k} B_i f(v_i) \approx \idotsint_{\mathbb{R}^d} w(x_1,\dots,x_d) f(x_1,\dots,x_d) dx_1,\dots,dx_d$$

as a quadrature/cubature formulae $Q$, where $B_i \in \mathbb{R}$ and $v_i \in \mathbb{R^d}$ are formula specific 
coefficients and vectors; $w:\mathbb{R}^d \to \mathbb{R}$ is a weight function/distribution; 
and $f:\mathbb{R}^d \to \mathbb{R}$ is the function to integrate/the integrand.

Such formulae $Q$ are said to be of degree $m$, with respect to a specific weight 
$w(\mathbf{x})$ and integration region $\Omega \in \mathbb{R}^d$, if they exactly 
integrate all $d$ dimensional polynomials of degree at least $m$ over the specified 
region. That is to say, degree $m$ formulae $Q^m$ are exact for $f(\mathbf{x}) = \sum_{\alpha \in \mathcal{A}_m} {c}_{\alpha} \mathbf{x}^\alpha$
where $\mathbf{x}^{\alpha} = \prod_{j=1}^d x_j^{\alpha_j}$ are monomials and ${c}_\alpha \in \mathbb{R}$ are coefficients, for all multi-indexes 
$\alpha \in \mathcal{A}_m := \{(\alpha_1, \dots, \alpha_d) \in \mathbb{N}_0, \sum_{i=1}^d \alpha_i \le m\}$.

.. admonition:: What about non-polynomial functions?
    :class: note

    The really neat thing to realize is that such formulae $Q^m$ are consequently good 
    integrators for *any* continuous function $f(\cdot)$ that can be well approximated by 
    the same polynomials for which $Q^m$ is an exact integrator. Thus, due to the Stone-Weierstrass
    theorem, a hypothetical "infinite" degree formula would approximately integrate all 
    $f \in C(\mathbb{R}^d)$ to within any desired error $\epsilon > 0$.

    In this article only exact formulae are considered; see :cite:p:`stroud1971` for 
    error analysis of approximate cubature formulae.

The final thing to consider, before presenting an example, is how exactly one finds 
values for the coefficients $B_i$ and vectors $v_i$ that define a given formula $Q$ 
(existence of such a formula is guaranteed Tchakaloff, see :cite:t:`davis1967` and :cite:t:`mysovskikh1975`). 
The issue of constructing formulae will not be discussed in any depth here and interested 
readers are referred to :cite:p:`cools1997` and :cite:p:`stroud1971`. However, it is 
important to be aware of what makes a *good* construction/formula. That is, a 
construction that requires the fewest vectors $v_i$ to produce a formula $Q^m$ (Such a 
construction can sometimes be further enhanced if the vectors are sparse or have other 
nice properties that can be leveraged to accelerate computation).

An example cubature
~~~~~~~~~~~~~~~~~~~
Consider the degree three Gaussian cubature formula from 
:cite:t:`stroudSecrest1963` ($E_n^{r^2} \text{3-1}$ in :cite:p:`stroud1971`), which 
exactly solves the following integral, for polynomial $f: \mathbb{R}^d \to \mathbb{R}$ of degree $m \le 3$,

$$\int_{\mathbb{R}^d} f(\mathbf{x})dP(\mathbf{x}) = \frac{1}{Z_c} 
\idotsint_{\mathbb{R}^d} f(x_1, \dots, x_d)\exp(-x_1^2 \dots -x_d^2)dx_1 \dots dx_d$$

where $P(x)$ is the probability measure of the $d\text{-dimensional}$ Gaussian 
distribution $X \sim \mathcal{N}(\mathbf{0}, \text{diag}(1/2))$ and $Z_c$ is a 
normalizing constant. If $Z_c$ is known, the above integral is the distribution's 
expectation $\operatorname{E}\left[f(X)\right]$; when $f(X) = X^\alpha$, a monomial, the
integral is the distribution's moment of order $\sum_{i=1}^d \alpha_i$. Hence, the 
cubature formula can be used to compute all moments of $X$ of order less than or equal 
to three.

Thus, consider the scenario where $d=2$ (two-dimensional Gaussian) and one wishes to 
compute the co-variance matrix (moments of order two) for the distribution - using the 
above cubature formula. To perform the computation one must:

1. Look up the formula's vectors and coefficients.
    For $E_n^{r^2}$ they are
    $$
    B_i = \frac{V}{2d}, \quad
    v=\begin{bmatrix}
    r  & 0 \\
    0  & r \\
    -r & 0 \\
    0  & -r\\
    \end{bmatrix},$$
    where $r^2 = d/2$, $V=\pi^{d/2}$ is the unnormalized volume of the weighted integration 
    region, and each row of the matrix $v$ is a cubature vector $v_i$.

2. Compute the normalization constant.
    The normalized volume must be one, the unnormalized volume is $V$, thus $Z_c=V$.

3. Evaluate the formula.
    $$\frac{1}{Z_c}\sum_{i=1}^{k} B_i f(v_i),$$
    where $f(x_1, x_2) = x_i x_j$, and $k=2d$ (the number of cubature vectors). Note 
    that the covariance matrix constists of d**2 monomials

One can either perform this computation by hand, or utilize :mod:`mccube.formulae` as 
shown below:

.. code-block:: python
    
    import jax
    import jax.numpy as jnp
    import numpy as np

    from mccube.formulae import StroudSecrest63_31

    d = 2
    expected_covariance = np.diag(np.ones(d)) / 2

    # Step 1. Get the formula vectors and coefficients
    cf = StroudSecrest63_31(dimension=d)
    B_i = cf.coefficients
    v_i = cf.vectors

    # Step 2. Compute the normalization constant
    z_c = cf.region.volume

    # Step 3a. Evaluate the formula (Manual)
    f_cov = lambda x: jnp.einsum("i, j -> ij", x, x)
    eval_vmap = jax.vmap(lambda b, v: b * f_cov(v), [0, 0])
    result = 1 / z_c * sum(eval_vmap(B_i, v_i))
    assert np.isclose(expected_covariance, result).all()
    print(f"(Manual) Expected result: {expected_covariance}\n Cubature result: {result}")

    # Step 3b. Evaluate the formula (mccube.formulae)
    result, _ = cf(f_cov, normalize=True)
    assert np.isclose(expected_covariance, result).all()
    print(f"(MCCube) Expected result: {expected_covariance}\n Cubature result: {result}")

Through a suitable affine transformation $\phi(x_1, \dots, x_n)$ of the vectors and 
coefficients, this cubature formula can be adapted to any parametrization of the above 
Gaussian distribution (see pg 11 of :cite:p:`stroud1971`). 

.. TODO: ADD A code example here.

Extending the example to SDE cubature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SDE cubature is an extension of conventional cubature/quadrature to infinite 
dimensional spaces. 

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
\operatorname{E}\left[X_t^{\alpha} \right] 
&= \operatorname{E}\left[(\int_0^t X_s\right)^{\alpha}] ds\\

$$ 

we need a way to compute the stochastic integral. The way to do this is via stochastic 
Taylor Expansion. As with many traditional/non-optimization based solution schemes 
one must discretise in time to actually solve numerically.

Nuance in the order $m$... nolonger a polynomial order per say, as much as a stochastic 
taylor polynomials order.

While a cubature is valid for all algebraic polynomials, a Cubature on Wiener space is 
valid only for stochastic Taylor Polynomials (that is truncated stochastic taylor 
expansions, of degree m). The function f must be bounded and smooth up to degree l.

.. admonition:: A note on notation
    :class: note

    Strictly the above integral does not exist in that standard sense, as the measure 
    $dW^{j}_t$ is not sufficiently smooth w.r.t time. Instead, one must define a 
    *rougher* integral such as those of Itô and Stratanovich. See :cite:p:`allan2021` 
    for an overview and :cite:p:`kloeden1992` for an in depth discussion.

The salient limitation of SDE cubatures, constructed as per :cite:t:`lyons2004`,
is that the path count scales exponentially with the number of discrete time steps 
($\mathcal{O}(n^{m})$, where $n$ is the propagator expansion factor, and $m$ is the 
number of time-integration steps). MCC solves this problem by constructing the 
collection of paths as a markov chain, where the :class:`MCCubatureStep` acts as a 
transition kernel that employs recombination to maintain the path/particle count at 
every time step. Note that in MCCube the paths are usually interpreted as particle 
trajectories, as this provides a consistent physically analogy.