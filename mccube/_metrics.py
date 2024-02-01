"""Defines helpful metrics and dissimilarity measures."""
from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.scipy.linalg import sqrtm
import lineax as lx
from jaxtyping import Array, PyTree, Real, Shaped

from ._custom_types import Particles, RealScalarLike

Mean = Real[Array, "d"]
"""Mean vector for a `d` dimensional distribution."""

Cov = Real[Array, "d d"]
"""Square positive definite (covariance) matrix for a `d` dimensional distribution."""

T = TypeVar("T")


def lp_metric(
    xs: PyTree[Array], ys: PyTree[Array], p: RealScalarLike = 2.0
) -> PyTree[RealScalarLike, "T"]:
    r"""$\ell^p$ metric; $\|x_s - y_s\|_p$."""
    return jtu.tree_map(lambda x, y: jnp.linalg.norm(x - y, ord=p, axis=-1), xs, ys)


def lpp_metric(
    xs: PyTree[Array], ys: PyTree[Array], p: RealScalarLike = 2.0
) -> PyTree[RealScalarLike, "T"]:
    r"""$\ell^p$ metric; $\|x_s - y_s\|_p^p$. [`mccube.lp_metric`][] to the power `p`."""
    return jtu.tree_map(
        lambda x, y: jnp.linalg.norm(x - y, ord=p, axis=-1) ** p, xs, ys
    )


def pairwise_metric(
    xs: Particles,
    ys: Particles,
    metric: Callable[[Mean, Mean], RealScalarLike] = lp_metric,
) -> PyTree[Shaped[Array, "?n ?n"], "Particles"]:
    """Pairwise metric between two PyTrees of `n` vectors of dimension `d`.

    Example:
        ```python
        x1 = jnp.ones((3, 1), 3),
        x2 = jnp.full((5, 1), 3),
        result = pairwise_metric(x1, x2, metric=mccube.euclidean_metric)
        # jnp.full((3, 3), 2)
        ```
    """
    return jtu.tree_map(jax.vmap(jax.vmap(metric, [0, None]), [None, 0]), xs, ys)


# Closed form metrics between multi-variate Gaussian distributions.
def gaussian_squared_bures_distance(
    cov1: PyTree[Cov, "T"],
    cov2: PyTree[Cov, "T"],
    sigma: RealScalarLike,
) -> PyTree[RealScalarLike, "T"]:
    r"""Entropy-regularized squared Bures distance between two multi-variate Gaussian
    distributions, based on the closed form defined in equation 14 of [`@janati2020`].

    $B_\sigma^2(\Sigma_1, \Sigma_2) = \text{Tr}(\Sigma_1 + \Sigma_2 - D_\sigma) +
    d\sigma^2(1-\log(2\sigma^2)) + \sigma^2 \log\det(D_\sigma + \sigma^2 \text{Id})$

    where:

    - $D_\sigma = (4\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2} + \sigma^4 \text{Id})^{1/2}$;
    - $d$ is the dimension of the mutli-variate Gaussian distributions, parameterised
    by the covariances $\Sigma_1$ and $\Sigma_2$;
    - $\sigma \ge 0$ is the regularization parameter.

    Special handling is provided for the following limits:

    - $\lim_{\sigma \to 0}B_\sigma^2(\Sigma_1, \Sigma_2) = \text{Tr}(\Sigma_1 + \Sigma_2
    -D_\sigma)$;
    - $\lim_{\sigma \to \infty}B_\sigma^2(\Sigma_1, \Sigma_2) = \text{Tr}(\Sigma_1 +
    \Sigma_2)$

    ??? cite "Reference: [`@janati2020`]"

        ```bibtex
        @inproceedings{janati2020,
         title     = {Entropic Optimal Transport between Unbalanced Gaussian Measures
                      has a Closed Form},
         author    = {Janati, Hicham and Muzellec, Boris and Peyr\'{e}, Gabriel and
                      Cuturi, Marco},
         year      = {2020},
         booktitle = {Advances in Neural Information Processing Systems},
         editor    = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and
                      H. Lin},
         pages     = {10468--10479},
         publisher = {Curran Associates, Inc.},
         url       = {https://proceedings.neurips.cc/paper_files/paper/2020/file/766e428d1e232bbdd58664b41346196c-Paper.pdf},
         volume    = {33},
        }
        ```
    """

    def _fn(A: Array, B: Array):
        assert A.ndim == 2
        assert A.shape[0] == A.shape[1]
        assert A.shape == B.shape

        d = A.shape[0]

        sigma_is_lower_limit = sigma == 0.0
        sigma_is_upper_limit = abs(sigma) == float("inf")
        sigma_is_limit = sigma_is_lower_limit or sigma_is_upper_limit

        A_sqrt = sqrtm(A)
        D_sigma = sqrtm(4 * A_sqrt @ B @ A_sqrt + sigma**4 * jnp.identity(d))
        trace_terms = jnp.trace(A + B - jnp.where(sigma_is_upper_limit, 0.0, D_sigma))
        regularization_terms = sigma**2 * (
            d * (1 - jnp.log(2 * sigma**2))
            + jnp.linalg.slogdet(D_sigma + sigma**2 * jnp.identity(d))[1]
        )
        squared_bures_distance = trace_terms + jnp.where(
            sigma_is_limit, 0.0, regularization_terms
        )
        return jnp.abs(squared_bures_distance)

    return jtu.tree_map(_fn, cov1, cov2)


def gaussian_optimal_transport(
    means: tuple[Mean, Mean],
    covs: tuple[Cov, Cov],
    sigma: RealScalarLike,
    cost: Callable[[Mean, Mean], RealScalarLike] = lpp_metric,
) -> RealScalarLike:
    r"""Entropy-regularized optimal transport between two multi-variate Gaussian
    distributions, based on the closed form in eqautions 13 and 14 of [`@janati2020`].
    $OT_{\sigma,c}(x_1, x_2) = c(\mu_1, \mu_2) + B_\sigma^2(\Sigma_1, \Sigma_2)$;

    where:

    - $x_1 \sim \text{Normal}(\mu_1, \Sigma_1)$ and $x_2 \sim \text{Normal}(\mu_2, \Sigma_2)$;
    - $B_\sigma^2(\Sigma_1, \Sigma_2)$ is the [`mccube.gaussian_squared_bures_distance`][];
    - $\sigma \ge 0$ is the regularization parameter.
    - $c(\cdot, \cdot)$ is the "ground cost" to move a unit of mass.

    Special cases occur in the following limit:

    - If $\sigma \to 0$, and $c(\mu_1, \mu_2) = \|\mu_1 - \mu_2\|_p^p$, the optimal
    transport is the p-Wasserstein metric raised to the power of $p$; $W_p^p(x_1, x_2)$.

    ??? cite "Reference: [`@janati2020`]"

        ```bibtex
        @inproceedings{janati2020,
         title     = {Entropic Optimal Transport between Unbalanced Gaussian Measures
                      has a Closed Form},
         author    = {Janati, Hicham and Muzellec, Boris and Peyr\'{e}, Gabriel and
                      Cuturi, Marco},
         year      = {2020},
         booktitle = {Advances in Neural Information Processing Systems},
         editor    = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and
                      H. Lin},
         pages     = {10468--10479},
         publisher = {Curran Associates, Inc.},
         url       = {https://proceedings.neurips.cc/paper_files/paper/2020/file/766e428d1e232bbdd58664b41346196c-Paper.pdf},
         volume    = {33},
        }
        ```
    """
    mean1, mean2 = means
    cov1, cov2 = covs
    return cost(mean1, mean2) + gaussian_squared_bures_distance(cov1, cov2, sigma)


def gaussian_wasserstein_metric(
    means: tuple[Mean, Mean], covs: tuple[Cov, Cov], p: RealScalarLike = 2
):
    r"""p-Wasserstein metric between two multi-variate Gaussian distributions. Alias
    for the $p\text{-th}$ root of [`mccube.gaussian_optimal_transport`][] at the limit
    $\sigma \to 0$, with "ground cost" $c(\mu_1, \mu_2) = \|\mu_1 - \mu_2\|_p^p$.
    """
    cost = jtu.Partial(lpp_metric, p=p)
    return gaussian_optimal_transport(means, covs, 0.0, cost) ** (1 / p)


def gaussian_sinkhorn_divergence(
    means: tuple[Mean, Mean],
    covs: tuple[Cov, Cov],
    sigma: RealScalarLike,
    cost: Callable[[Mean, Mean], RealScalarLike] = lpp_metric,
) -> RealScalarLike:
    r"""Sinkhorn divergence between two multi-variate Gaussian distributions, based on
    the closed form in [`@janati2020`];
    $S_{\sigma,c}(x_1, x_2) = \text{OT}_{\sigma,c}(x_1, x_2) - \frac{1}{2}(
    \text{OT}_{\sigma,c}(x_1, x_1) + \text{OT}_{\sigma,c}(x_2, x_2))$;

    where:

    - $x_1 \sim \text{Normal}(\mu_1, \Sigma_1)$ and $x_2 \sim \text{Normal}(\mu_2, \Sigma_2)$;
    - $\text{OT}_{\sigma,c}(x_1, x_2)$ is the [`mccube.gaussian_optimal_transport`][];
    - $\sigma \ge 0$ is the regularization parameter.
    - $c(\cdot, \cdot)$ is the "ground cost" to move a unit of mass.

    Special cases occur in the following limits:

    - If $\sigma \to 0$, and $c(\mu_1, \mu_2) = \|\mu_1 - \mu_2\|_p^p$, the sinkhorn
    divergence is the optimal transport, which is itself the p-Wasserstein metric raised
    to the power $p$; $W_p^p(x_1, x_2)$.
    - If $\sigma = \infty$, the sinkhorn divergence is the maximum mean discrepancy
    (MMD) with a $-c$ kernel. If, additionally, $c(\mu_1, \mu_2) = \|\mu_1 - \mu_2\|_p^p$
    and $1 < p < 2$, the MMD is the energy distance [`@genevay2018`].

    ??? cite "Reference: [`@janati2020`]"

        ```bibtex
        @inproceedings{janati2020,
         title     = {Entropic Optimal Transport between Unbalanced Gaussian Measures
                      has a Closed Form},
         author    = {Janati, Hicham and Muzellec, Boris and Peyr\'{e}, Gabriel and
                      Cuturi, Marco},
         year      = {2020},
         booktitle = {Advances in Neural Information Processing Systems},
         editor    = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and
                      H. Lin},
         pages     = {10468--10479},
         publisher = {Curran Associates, Inc.},
         url       = {https://proceedings.neurips.cc/paper_files/paper/2020/file/766e428d1e232bbdd58664b41346196c-Paper.pdf},
         volume    = {33},
        }
        ```

    ??? cite "Reference: [`@genevay2018`]"

        ```bibtex
        @inproceedings{genevay2018,
          title     = {Learning Generative Models with Sinkhorn Divergences},
          author    = {Genevay, Aude and Peyre, Gabriel and Cuturi, Marco},
          year      = {2018},
          booktitle = {Proceedings of the Twenty-First International Conference on
                       Artificial Intelligence and Statistics},
          pages     = {1608--1617},
          editor    = {Storkey, Amos and Perez-Cruz, Fernando},
          volume    = {84},
          series    = {Proceedings of Machine Learning Research},
          month     = {09--11 Apr},
          publisher = {PMLR},
          pdf       = {http://proceedings.mlr.press/v84/genevay18a/genevay18a.pdf},
          url       = {https://proceedings.mlr.press/v84/genevay18a.html},
        }
        ```
    """
    mean1, mean2 = means
    cov1, cov2 = covs
    ot12 = gaussian_optimal_transport(means, covs, sigma, cost)
    ot11 = gaussian_squared_bures_distance(cov1, cov1, sigma)
    ot22 = gaussian_squared_bures_distance(cov2, cov2, sigma)
    return ot12 - (ot11 + ot22) / 2


def gaussian_maximum_mean_discrepancy(
    mean1: Mean,
    mean2: Mean,
    cost: Callable[[Mean, Mean], RealScalarLike] = lpp_metric,
) -> RealScalarLike:
    r"""Maximum mean discrepancy ($\text{MMD}_{-c}$) between two multi-variate Gaussian
    distributions. Equivalent to [`mccube.gaussian_sinkhorn_divergence`][] at the limit
    $\sigma \to \infty$, with arbitrary covariances and $-c$ "ground cost".
    """
    return cost(mean1, mean2)


def gaussian_kl_divergence(
    means: tuple[Mean, Mean], covs: tuple[Cov, Cov]
) -> RealScalarLike:
    """Kullback-Leibler divergence between two multi-variate Gaussian distributions,
    based on the closed-form derived in §9 of [`@duchi2007`].

    ??? cite "Reference: [`@duchi2007`]"

        ```bibtex
        @misc{duchi2007,
          title = {Derivations for Linear Algebra and Optimization},
          author = {John Duchi},
          month = {February},
          year = {2007},
          publisher = {Stanford University},
          url = {https://web.stanford.edu/~jduchi/projects/general_notes.pdf}
        }
        ```
    """
    mean1, mean2 = means
    cov1, cov2 = covs

    assert cov1.ndim == 2
    assert cov1.shape[0] == cov2.shape[1]
    assert cov1.shape == cov2.shape

    d = cov1.shape[0]

    sign_1, logdet_cov1 = jnp.linalg.slogdet(cov1)
    sign_2, logdet_cov2 = jnp.linalg.slogdet(cov2)
    multi_linear_solve = jax.vmap(lx.linear_solve, (None, 1))

    cov2 = lx.MatrixLinearOperator(cov2, lx.positive_semidefinite_tag)
    cov2_inv_cov1 = multi_linear_solve(cov2, cov1).value
    mean_delta = mean2 - mean1
    cov2_inv_mean_delta = lx.linear_solve(cov2, mean_delta).value

    return (
        (logdet_cov2 - logdet_cov1)
        - d
        + jnp.trace(cov2_inv_cov1)
        + mean_delta.T @ cov2_inv_mean_delta
    ) / 2
