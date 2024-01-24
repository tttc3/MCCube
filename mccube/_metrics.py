"""Defines helpful metrics and dissimilarity measures."""
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.scipy.linalg import sqrtm
from jaxtyping import ArrayLike, Complex, PyTree, Shaped

from ._custom_types import Particles, RealScalarLike, Weights

Mean = Shaped[ArrayLike, "d"]
"""Mean vector for a `d` dimensional distribution."""

Cov = Shaped[ArrayLike, "d d"]
"""Covariance matrix for a `d` dimensional distribution."""


def gaussian_wasserstein_metric(
    means: tuple[Mean, Mean], covs: tuple[Cov, Cov]
) -> Complex[ArrayLike, ""]:
    """2-Wasserstein metric between two multi-variate Gaussian distributions.

    Example:
        ```python
        mean = jnp.array([1,2])
        cov = jnp.array([[1,0],[0,1]])
        result = gaussian_wasserstein_metric((mean, mean), (cov, cov))
        # Array(0.+0.j, dtype=complex64)
        ```

    Args:
        means: is a tuple of two `d` dimensional mean vectors.
        covs: is a tuple of two `d x d` dimensional covariance matrices.

    Returns:
        The 2-Wasserstein metric betwen the `d` dimensional Gaussian distributions
            parameterised by the `means` and `covariances`.

    """
    m1, m2 = means
    c1, c2 = covs
    root_c2 = sqrtm(c2)
    mean_dist = jnp.linalg.norm(m1 - m2) ** 2  # pyright: ignore
    cov_dist = jnp.trace(c1 + c2 - 2 * sqrtm(root_c2 @ c1 @ root_c2))
    return jnp.asarray(mean_dist + cov_dist)


def euclidean_metric(
    xs: PyTree[ArrayLike, "Particles"],
    ys: PyTree[ArrayLike, "Particles"],
) -> PyTree[RealScalarLike, "Particles"]:
    """Euclidean metric."""
    return jtu.tree_map(lambda x, y: jnp.linalg.norm(x - y, axis=-1), xs, ys)


def squared_euclidean_metric(
    xs: PyTree[ArrayLike, "Particles"],
    ys: PyTree[ArrayLike, "Particles"],
) -> PyTree[RealScalarLike, "Particles"]:
    """Squared Euclidean metric."""
    return jtu.tree_map(lambda _xs, _ys: jnp.square(euclidean_metric(_xs, _ys)), xs, ys)


def pairwise_metric(
    xs: Particles, ys: Particles, metric=euclidean_metric
) -> PyTree[Shaped[ArrayLike, "?n ?n"], "Particles"]:
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


def center_of_mass(
    particles: Particles, weights: Weights | None = None
) -> PyTree[Mean, "Particles"]:
    """Compute the weighted mean/center of mass of a [`Particles`][mccube._custom_types.Particles]
    PyTree. If `weights==None` then all particles are equally weighted.

    Example:
        ```python
        particles = {"y": jnp.array([1,2,3])}
        weights = {"y": jnp.array([1,2,2])}
        result = center_of_mass(particles, weights)
        # {"y": Array(2.2, dtype=float32)}

        result = center_of_mass(particles, None)
        # {"y": Array(2.0, dtype=float32)}
        ```
    """
    return jtu.tree_map(jnp.average, particles, 0, weights)
