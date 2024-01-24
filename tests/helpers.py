import mccube

gaussian_formulae = {
    f
    for f in mccube._formulae.builtin_cubature_registry
    if issubclass(f, mccube.AbstractGaussianCubature)
    and f is not mccube.AbstractGaussianCubature
}
