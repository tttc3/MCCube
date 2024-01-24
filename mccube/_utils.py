"""Helpful utilities used throughout MCCube."""
import jax.numpy as jnp
import jax.tree_util as jtu

from ._custom_types import PackedParticles, Particles, UnpackedParticles, Weights


def nop(*args, **kwargs) -> None:
    """Callable which accepts any arguments and does nothing.

    Example:
        ```python
        result = nop(123, test="string")
        # None
        ```
    """
    ...


def pack_particles(
    particles: Particles, weights: Weights | None
) -> Particles | PackedParticles:
    """Packs `particles` and normalized `weights` into a single
    [`Particles`][mccube._custom_types.Particles] PyTree.

    Example:
        ```python
        particles = {"y1": jnp.ones((10, 2))} # Shape: [n=10, d=2]
        weights = {"y1": jnp.ones(10)} # Shape: [n=10]
        packed_particles = mccube.pack_particles(particles, weights)
        # {"y1": jnp.c_[particles["y1"], weights["y1"]i / 10]} Shape: [n=n, d=d+1]

        weights = {"y1": None}
        packed_particles = mccube.pack_particles(particles, weights)
        # {"y1": particles["y1"]} Shape: [n=n, d=d]
        ```

    Args:
        particles: is a [`Particles`][mccube._custom_types.Particles] PyTree.
        weights: is either a [`Weights`][mccube._custom_types.Weights] PyTree or [`None`][].

    Returns:
        A [`Particles`][mccube._custom_types.Particles] PyTree where each leaf has
            dimension `d+1` and this new trailing dimension represents the [`Weights`][mccube._custom_types.Weights].
            If `weights` is [`None`][], then `particles` is returned unmodified.
    """
    return jtu.tree_map(
        lambda p, w: jnp.c_[p, jnp.squeeze(w) / jnp.sum(w)] if w is not None else p,
        particles,
        weights,
    )


def unpack_particles(
    particles: Particles, weighted: bool = False
) -> tuple[Particles, None] | tuple[UnpackedParticles, Weights]:
    """Unpacks [`PackedParticles`][mccube._custom_types.PackedParticles] into separate
    PyTrees of [`Particles`][mccube._custom_types.Particles] and [`Weights`][mccube._custom_types.Weights].

    Example:
        ```python
        particles = jnp.ones((10,2)) # Shape: [n=10, d=2]
        weights = jnp.full(10, 1/10) # Shape: [n=10]
        packed_particles = pack_particles(particles, weights) # Shape: [n=n, d=d+1]
        result = unpack_particles(packed_particles, weighted=True)
        # (particles, weights) Shape: [n=n, d=d], [n=n]

        result = unpack_particles(packed_particles)
        # (jnp.c_[particles, weights], None) Shape: [n=n, d=d+1], None
        ```

    Args:
        particles: is a [`Particles`][mccube._custom_types.Particles] PyTree.
        weighted: indicates if to treat the `particles` as [`PackedParticles`][mccube._custom_types.PackedParticles]
            (`weighted==True`), or standard [`Particles`][mccube._custom_types.Particles]
            (`weighted==False`).

    Returns:
        A tuple of the unpacked `particles` and weights (`weighted==True`), or a tuple
            of `particles` and `None` (`weighted==False`).
    """
    _particles = jtu.tree_map(lambda p: p[..., :-1] if weighted else p, particles)
    _weights = jtu.tree_map(lambda p: p[..., -1] if weighted else None, particles)
    return _particles, _weights


def all_subclasses(cls: type) -> set[type]:
    """Recursively identifies all subclasses of a class in the current scope.

    Example:
        ```python
        result = all_subclasses(mccube.AbstractRegion)
        # {mccube.GaussianRegion}
        ```
    """
    subclasses = set()

    for subclass in cls.__subclasses__():
        subclasses.add(subclass)
        subclasses.update(all_subclasses(subclass))
    return subclasses


def requires_weighing(is_weighted: bool) -> None:
    if not is_weighted:
        raise ValueError("Kernel requires `weighted=True`; got {`weighted=False`}.")
    return
